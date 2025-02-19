from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def translate_texts(texts, model, tokenizer, device, max_length=512):
    """ Efficient batch translation without using pipeline """
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
        
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def translate_batch(batch, model, tokenizer, device):
    """ Translates a batch of triplets using efficient batched inference """
    
    batch["metadata"] = [
        {"anchor_en": a, "entailment_en": p, "negative_en": n}
        for a, p, n in zip(batch["anchor"], batch["entailment"], batch["negative"])
    ]
    
    batch["anchor"] = translate_texts(batch["anchor"], model, tokenizer, device)
    batch["entailment"] = translate_texts(batch["entailment"], model, tokenizer, device)
    batch["negative"] = translate_texts(batch["negative"], model, tokenizer, device)
    
    return batch

def translate_batch_score(batch, model, tokenizer, device):
    """ Translates a batch of triplets using efficient batched inference """
    
    batch["metadata"] = [
        {"sentence1_en": a, "sentence2_en": p, "score": n}
        for a, p, n in zip(batch["sentence1"], batch["sentence2"], batch["score"])
    ]
    
    batch["sentence1"] = translate_texts(batch["sentence1"], model, tokenizer, device)
    batch["sentence2"] = translate_texts(batch["sentence2"], model, tokenizer, device)
    batch["score"] = batch["score"]
    
    return batch

if __name__ == "__main__":
    
    # dataset to translate
    BASE_DATASET = "sentence-transformers/all-nli" #"jinaai/negation-dataset"  # All of (sentence-transformers/miracl and sentence-transformers/mldr) for en-triplet split and sentence-transformers/trivia-qa-triplet for triplet split
    # where to save translated data
    NEW_DATASET = "atlasia/jinaai-negation-dataset-moroccan-darija-ultra"
    # translation model
    TRANS_MODEL = "BounharAbdelaziz/Terjman-Ultra-v2.2" #"BounharAbdelaziz/Terjman-Ultra-v2.2" "BounharAbdelaziz/Terjman-Large-v2.2" "BounharAbdelaziz/Terjman-Nano-v2.2" 
    #
    batch_size = 64
    MAX_SAMPLES = 200_000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        TRANS_MODEL, 
        src_lang="eng_Latn", 
        tgt_lang="ary_Arab"
    )
    
    # load original dataset
    if "all-nli" in BASE_DATASET:
        dataset = load_dataset(BASE_DATASET, "pair-score", split="train")
    else:
        dataset = load_dataset(BASE_DATASET, split="train")
        
    MAX_SAMPLES = min(MAX_SAMPLES, len(dataset))
    if MAX_SAMPLES < len(dataset):
        dataset = dataset.shuffle().select(range(MAX_SAMPLES))
        
    # run translations
    if "all-nli" in BASE_DATASET:
        dataset = dataset.map(lambda x: translate_batch_score(x, model, tokenizer, device), batched=True, batch_size=batch_size)
    else:
        dataset = dataset.map(lambda x: translate_batch(x, model, tokenizer, device), batched=True, batch_size=batch_size)
    # save in hub
    dataset.push_to_hub(NEW_DATASET, commit_message=f"Translated dataset {BASE_DATASET}")