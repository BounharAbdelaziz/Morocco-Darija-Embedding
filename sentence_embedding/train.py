import random
import logging
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
)
from transformers import AutoTokenizer
from sentence_transformers.models.StaticEmbedding import StaticEmbedding

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)

from sentence_transformers.training_args import (
    BatchSamplers, 
    MultiDatasetBatchSamplers,
)

from sentence_transformers.losses import (
    CoSENTLoss, 
    MultipleNegativesRankingLoss,
    MatryoshkaLoss,
)

from sentence_transformers.evaluation import (
    TripletEvaluator,
    TranslationEvaluator,
    EmbeddingSimilarityEvaluator,
)

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
        
def load_train_eval_datasets(train_dataset, bench_dataset="atlasia/Morocco-Darija-Sentence-Embedding-Benchmark"):
    """ Load and prepare training and evaluation datasets. """

    print("Loading dataset...")
    
    ################################# Triplet dataset #################################
    
    dataset_triplet = load_dataset(train_dataset, "triplet", split="train")
    dataset_triplet = dataset_triplet.map(
        lambda x: {"anchor": x["anchor"], "positive": x["positive"], "negative": x["negative"]}, 
        remove_columns=["original_source", "translation_model", "metadata"]
    )
    dataset_dict_triplet = dataset_triplet.train_test_split(test_size=250, seed=12)
    train_dataset_triplet: Dataset = dataset_dict_triplet["train"]
    eval_dataset_triplet: Dataset = dataset_dict_triplet["test"]
    
    ################################# Negation Triplet dataset #################################
    
    dataset_negation_triplet = load_dataset(train_dataset, "negation-triplet", split="train")
    dataset_negation_triplet = dataset_negation_triplet.map(
        lambda x: {"anchor": x["anchor"], "entailment": x["entailment"], "negative": x["negative"]}, 
        remove_columns=["original_source", "translation_model", "metadata"]
    )
    dataset_dict_negation_triplet = dataset_negation_triplet.train_test_split(test_size=250, seed=12)
    train_dataset_negation_triplet: Dataset = dataset_dict_negation_triplet["train"]
    eval_dataset_negation_triplet: Dataset = dataset_dict_negation_triplet["test"]
    
    ################################# Pair Score dataset #################################
    
    dataset_pair_score = load_dataset(train_dataset, "pair-score", split="train") #.shuffle(seed=12).select(range(75_00))
    dataset_pair_score = dataset_pair_score.map(
        lambda x: {"sentence1": x["sentence1"], "sentence2": x["sentence2"], "score": x["score"]}, 
        remove_columns=["original_source", "translation_model", "metadata"]
    )
    dataset_dict_pair_score = dataset_pair_score.train_test_split(test_size=250, seed=12)
    train_dataset_pair_score: Dataset = dataset_dict_pair_score["train"]
    eval_dataset_pair_score: Dataset = dataset_dict_pair_score["test"]
    
    ################################# (English, Non english) dataset #################################
    
    dataset_english_non_english = load_dataset(train_dataset, "english-non_english", split="train")
    dataset_english_non_english = dataset_english_non_english.map(
        lambda x: {"english": x["english"], "non_english": x["non_english"]}, 
        remove_columns=["original_source"]
    )
    dataset_dict_english_non_english = dataset_english_non_english.train_test_split(test_size=250, seed=12)
    train_dataset_english_non_english: Dataset = dataset_dict_english_non_english["train"]
    eval_dataset_english_non_english: Dataset = dataset_dict_english_non_english["test"]
    
    ############################### AtlasIA Sentence Embedding Benchmark ##############################
    
    # already prepared
    atlasia_sent_embd_bench = load_dataset(bench_dataset, split="test")
    
    ##################################################################################################
    
    # combine all datasets
    train_dataset = DatasetDict({
        "triplet": train_dataset_triplet,
        "negation_triplet": train_dataset_negation_triplet,
        "pair_score": train_dataset_pair_score,
        "english_non_english": train_dataset_english_non_english,
    })
    
    eval_dataset = DatasetDict({
        "triplet": eval_dataset_triplet,
        "negation_triplet": eval_dataset_negation_triplet,
        "pair_score": eval_dataset_pair_score,
        "english_non_english": eval_dataset_english_non_english,
        "atlasia_sent_embd_bench": atlasia_sent_embd_bench,
    })
   
    print("[INFO] Loaded train and eval datasets.")
    
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
    
    # Benchmark data path
    BENCH_DATASET_PATH = config['BENCH_DATASET_PATH']
    
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
    if config['STATIC_EMBEDDINGS']:
        static_embedding = StaticEmbedding(AutoTokenizer.from_pretrained(MODEL_PATH), embedding_dim=1024)
        model = SentenceTransformer(
            modules=[static_embedding],
            model_card_data=SentenceTransformerModelCardData(
                license="apache-2.0",
                model_name="Static Embeddings with BERT Multilingual uncased tokenizer finetuned on Moroccan Darija.",
            ),
        )
    else:
        model = SentenceTransformer(
            MODEL_PATH,
            model_card_data=SentenceTransformerModelCardData(
                license="apache-2.0",
                model_name="Sentence embeddings for finetuned on Moroccan Darija.",
            ),
        )

    # 3. Set up training & evaluation datasets - each dataset is trained with MNRL (with MRL)
    train_dataset, eval_dataset = load_train_eval_datasets(
        train_dataset=TRAIN_DATA_PATH, 
        bench_dataset=BENCH_DATASET_PATH
    )
    
    print(f'train_dataset: {train_dataset}')
    print(f'eval_dataset: {eval_dataset}')
    
    # 4. Load several loss functions to train with
    
    # (anchor, positive), (anchor, positive, negative)
    mnrl_loss = MultipleNegativesRankingLoss(model)
    mnrl_loss = MatryoshkaLoss(model, mnrl_loss, matryoshka_dims=[32, 64, 128, 256, 512, 1024])
    
    # (sentence_A, sentence_B) + score
    cosent_loss = CoSENTLoss(model)
    
    # Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where.
    # Note that you can also just use one loss if all of your training/evaluation datasets use the same loss
    losses = {
        "triplet": mnrl_loss,
        "negation_triplet": mnrl_loss,
        "pair_score": cosent_loss,
        "english_non_english": mnrl_loss,
        "atlasia_sent_embd_bench": cosent_loss, # is not used for trainning but must be included in the losses dict
    } 
   
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
    # Evaluators
    dev_evaluators = []

    ## triplet dataset
    triplet_evaluator = TripletEvaluator(
        anchors=eval_dataset['triplet']["anchor"],
        positives=eval_dataset['triplet']["positive"],
        negatives=eval_dataset['triplet']["negative"],
        name="triplet-evaluator-dev",
    )
    dev_evaluators.append(triplet_evaluator)
    
    ## negation triplet dataset
    negation_triplet_evaluator = TripletEvaluator(
        anchors=eval_dataset['negation_triplet']["anchor"],
        positives=eval_dataset['negation_triplet']["entailment"],
        negatives=eval_dataset['negation_triplet']["negative"],
        name="negation-triplet-evaluator-dev",
    )
    dev_evaluators.append(negation_triplet_evaluator)
    
    ## pair score dataset
    pair_score_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset['pair_score']["sentence1"],
        sentences2=eval_dataset['pair_score']["sentence2"],
        scores=eval_dataset['pair_score']["score"],
        name="pair-score-evaluator-dev",
    )
    dev_evaluators.append(pair_score_evaluator)
    
    # English, Non-English dataset
    translation_evaluator = TranslationEvaluator(
        source_sentences=eval_dataset['english_non_english']["english"],
        target_sentences=eval_dataset['english_non_english']["non_english"],
        name="english_non_english-evaluator-dev",
    )
    dev_evaluators.append(translation_evaluator)
    
    ## atlasia_sent_embd_bench dataset
    atlasia_bench_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset['atlasia_sent_embd_bench']["sentence1"],
        sentences2=eval_dataset['atlasia_sent_embd_bench']["sentence2"],
        scores=eval_dataset['atlasia_sent_embd_bench']["score"],
        name="atlasia_sent_embd_bench-evaluator-dev",
    )
    dev_evaluators.append(atlasia_bench_evaluator)
    
    # Evaluate the model before training
    triplet_evaluator(model)
    negation_triplet_evaluator(model)
    pair_score_evaluator(model)
    translation_evaluator(model)
    atlasia_bench_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=losses,
        evaluator=dev_evaluators,
    )
    trainer.train()

    # 8.1 Save the trained model
    model.save_pretrained(MODEL_RUN_SAVE_PATH)

    # 8.2 Push it to the Hugging Face Hub
    model.push_to_hub(MODEL_RUN_SAVE_PATH, private=True)