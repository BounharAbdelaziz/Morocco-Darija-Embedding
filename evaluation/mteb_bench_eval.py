import gc
import torch
from sentence_transformers import SentenceTransformer
from mteb import MTEB
import mteb
import os

def info(message):
  print(25*"="+message+25*"=")

TASKS=[
    "SemRel24STS",
    "SIB200Classification",
    "SIB200ClusteringS2S",
    "AfriSentiClassification",
    # "AfriSentiLangClassification", # throws ERROR:mteb.evaluation.MTEB:Error while evaluating AfriSentiLangClassification: Dataset 'HausaNLP/afrisenti-lid-data' doesn't exist on the Hub or cannot be accessed.
    "BelebeleRetrieval",
    "FloresBitextMining",
    "SIB200Classification",
    "SIB200ClusteringS2S",
    "SemRel24STS",

]

LANGUAGES=["ary"]

MODELS_NAMES=[
  # "atlasia/Qwen2.5-72B-bs-4096-lr-0.05-ep-10-wp-0.05-gacc-1-gnm-1.0-v0.2",
  # "atlasia/XLM-RoBERTa-Morocco-bs-32-lr-2e-05-ep-2-wp-0.05-gacc-1-gnm-1.0-v0.2",
  # "atlasia/xlm-roberta-large-ft-alatlas-bs-32-lr-2e-05-ep-2-wp-0.05-gacc-1-gnm-1.0-v0.2",
  "BounharAbdelaziz/Morocco-Darija-Sentence-Embedding-v0.2",
  "BounharAbdelaziz/ModernBERT-Morocco-Sentence-Embeddings-v0.2-bs-32-lr-2e-05-ep-2-wp-0.05-gacc-1-gnm-1.0-v0.3",
  # "BounharAbdelaziz/Morocco-Darija-Sentence-Embedding-v0.1",
  # "BounharAbdelaziz/Morocco-Darija-Static-Sentence-Embedding",
  # "intfloat/multilingual-e5-large",
  # "BAAI/bge-m3"
]

batch_size = 512

if __name__=="__main__":

  # Ensure output directory exists
  SAVINGS_BASE_DIR = "./mteb_bench_eval_results"
  os.makedirs(SAVINGS_BASE_DIR, exist_ok=True)
  
  models = {}

  device = "cuda" if torch.cuda.is_available() else "cpu"
  info(f"Using device: {device}")
  
  if device == "cpu":
      info("Warning: CUDA not available. Running on CPU may be slow.")

  # prepare tasks
  TASKS=[mteb.get_task(task_name=task,languages=LANGUAGES) for task in TASKS]
  
  # Load MTEB tasks
  info("Loading MTEB tasks...")
  mteb_task = MTEB(
      tasks=TASKS,
  )
  info("DONE!")

  # Run MTEB evaluation
  for model_name in MODELS_NAMES:
      # Load models
      model = SentenceTransformer(model_name, device=device)
      name = model_name.split("/")[-1]  # Use the last part of the model name
      output_folder = os.path.join(SAVINGS_BASE_DIR, f"{name}-mteb-results")
      os.makedirs(output_folder, exist_ok=True)
      
      info(f"Running MTEB on {model_name} ...")
      mteb_task.run(
        model=model, 
        output_folder=output_folder,
        batch_size=batch_size
      )
      
      # remove model from memory
      del model
      gc.collect()
      torch.cuda.empty_cache()

  info("ALL DONE!")