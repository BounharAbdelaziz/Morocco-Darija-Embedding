import torch
from sentence_transformers import SentenceTransformer
from mteb import MTEB
import mteb

def info(message):
  print(25*"="+message+25*"=")

TASKs=[
    "SemRel24STS","SIB200Classification","SIB200ClusteringS2S",
    "AfriSentiClassification","AfriSentiLangClassification",
    "BelebeleRetrieval","FloresBitextMining","SIB200Classification",
    "SIB200ClusteringS2S","SemRel24STS",

],
LANGUAGEs=["ary"]
MODELS_NAMES=["BounharAbdelaziz/Morocco-Darija-Sentence-Embedding",
        "BounharAbdelaziz/Morocco-Darija-Static-Sentence-Embedding",
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3"]
if __name__=="__main__":
  models={}

  device="cuda" if torch.cuda.is_available() else "cpu"
  info(f"device: {device}")

  for name in MODELS_NAMES:
    info(f"load model {name} ...")
    new_name=name.split("/")[1]
    models[new_name]=SentenceTransformer(name,device=device)
    info(f"DONE!")

  info(f"MTEB Load... ")
  mteb_task=MTEB(
      tasks=TASKs,
      task_langs=LANGUAGEs
  )
  info(f"DONE!")

  for name,model in models.items():
    info(f"MTEB RUN ON {name}...")
    mteb_task.run(model=model,output_folder=f"{name}-mteb-result")
    info(f"DONE!")
  info("DONE!")
