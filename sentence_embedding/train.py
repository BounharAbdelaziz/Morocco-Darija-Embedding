import random
import logging
from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
random.seed(12)

from pprint import pprint
import numpy as np
import wandb
import yaml
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_seed(seed):
    """ Sets the seed for reproducibility """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def load_train_eval_datasets(dataset_path):
    """
    Either load the train and eval datasets from disk or load them from the datasets library & save them to disk.
    Upon saving to disk, we quit() to ensure that the datasets are not loaded into memory before training.
    """

    print("Loading dataset...")
    dataset_1 = load_dataset(dataset_path, split="train")
    dataset_1 = dataset_1.map(
        lambda x: {"anchor": x["anchor"], "positive": x["positive"], "negative": x["negative"]}, 
        remove_columns=["original_source", "translation_model", "metadata"]
    )
    dataset_dict_1 = dataset_1.train_test_split(test_size=250, seed=12)
    train_dataset_1: Dataset = dataset_dict_1["train"]
    eval_dataset_1: Dataset = dataset_dict_1["test"]
    
    dataset_2 = load_dataset("BounharAbdelaziz/Terjman-v2-English-Darija-Dataset-350K", split="train")
    dataset_2 = dataset_2.map(
        lambda x: {"english": x["english"], "non_english": x["darija_Arab"]}, 
        remove_columns=["darija_Latn", "darija_Arab", "dataset_source", "id", "role", "darija_tokens", "subtopic", "topic", "annotator_dialect"]
    )
    dataset_dict_2 = dataset_2.train_test_split(test_size=250, seed=12)
    train_dataset_2: Dataset = dataset_dict_2["train"]
    eval_dataset_2: Dataset = dataset_dict_2["test"]
    
    train_dataset = DatasetDict({
        "anchor_trans": train_dataset_1,
        "terjman_v2_350K": train_dataset_2,
    })
    
    eval_dataset = DatasetDict({
        "anchor_trans": eval_dataset_1,
        "terjman_v2_350K": eval_dataset_2,
    })
   
    print("Loaded dataset.")
    
    return train_dataset, eval_dataset

    
if __name__ == "__main__":
    
    # Set up logging and tracking
    wandb.login()
    
    # get training configuration
    with open('training_config.yaml') as file:
        config = yaml.safe_load(file)
    
    print('-'*50)
    print("Training configuration:")
    pprint(config)
    print('-'*50)
    
    # Training hyperparameters
    num_train_epochs = config['hyperparameters']['num_train_epochs']
    lr = config['hyperparameters']['lr']
    batch_size = config['hyperparameters']['batch_size']
    gradient_accumulation_steps = config['hyperparameters']['gradient_accumulation_steps']
    max_grad_norm = config['hyperparameters']['max_grad_norm']
    warmup_steps = config['hyperparameters']['warmup_steps']
    warmup_ratio = config['hyperparameters']['warmup_ratio']
    
    # Logging and saving
    logging_steps = config['hyperparameters']['logging_steps']
    save_steps = config['hyperparameters']['save_steps']
    eval_steps = config['hyperparameters']['eval_steps']

    # Training data path
    TRAIN_DATA_PATH = config['DATASET_PATH']
    
    # base model path
    MODEL_PATH = config['BASE_MODEL']
    FP16_TRAINING = config['FP16_TRAINING']
    
    if FP16_TRAINING:
        torch_dtype=torch.bfloat16 # bfloat16 has better precission than float16 thanks to bigger mantissa. Though not available with all GPUs architecture.
    else:
        torch_dtype=torch.float32
    
    # set seed
    SEED = config['SEED']
    set_seed(SEED)
    
    # build and check run name    
    run_name = f'{MODEL_PATH.split("/")[-1]}-bs-{batch_size}-lr-{lr}-ep-{num_train_epochs}-wp-{warmup_ratio}-gacc-{gradient_accumulation_steps}-gnm-{max_grad_norm}-{config['version']}'
    assert '--' not in run_name, f"[WARN] Detected -- in run_name. This will cause a push_to_hub error! Found run_name={run_name} "
    assert len(run_name) < 96, f"[WARN] run_name too long, found len(run_name)={len(run_name)} > 96. This will cause a push_to_hub error! Consider squeezing it. Found run_name={run_name}"

    MODEL_RUN_SAVE_PATH = f"atlasia/{run_name}"

    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged, all runs will be under this project
        project=config['project_name'],   
        # Group runs by model size
        group=MODEL_PATH,       
        # Unique run name
        name=run_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "warmup_ratio": warmup_ratio,
            # "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            # "weight_decay": weight_decay,
            "dataset": TRAIN_DATA_PATH,
        }
    )

    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        MODEL_PATH,
        model_card_data=SentenceTransformerModelCardData(
            license="apache-2.0",
            model_name="Sentence embeddings for finetuned on Moroccan Darija.",
        ),
    )

    # 3. Set up training & evaluation datasets - each dataset is trained with MNRL (with MRL)
    train_dataset, eval_dataset = load_train_eval_datasets(TRAIN_DATA_PATH)
    
    print(f'train_dataset: {train_dataset}')
    print(f'eval_dataset: {eval_dataset}')

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, loss, matryoshka_dims=[32, 64, 128, 256, 512, 768])

    # 5. Specify training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=MODEL_RUN_SAVE_PATH,
        eval_strategy="steps",
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
        bf16=config['FP16_TRAINING'],
        fp16_full_eval=config['FP16_TRAINING'],
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        report_to="wandb",
        push_to_hub=True,
        metric_for_best_model=config['METRIC_FOR_BEST_MODEL'],
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        logging_first_step=True,
    )
    
    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset['anchor_trans']["anchor"],
        positives=eval_dataset['anchor_trans']["positive"],
        negatives=eval_dataset['anchor_trans']["negative"],
        name="triplet-evaluator-dev",
    )
    dev_evaluator(model)

    # 6. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. Save the trained model
    model.save_pretrained(MODEL_RUN_SAVE_PATH)

    # 8. Push it to the Hugging Face Hub
    model.push_to_hub(MODEL_RUN_SAVE_PATH, private=True)