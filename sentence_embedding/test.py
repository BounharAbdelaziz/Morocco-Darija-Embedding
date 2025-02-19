import torch
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    
    MODEL_PATH = "atlasia/sentence-transformer-ft-BAAI--bge-m3"
    model = SentenceTransformer(MODEL_PATH)

    # Test the model
    anchor = "The movie was fantastic"
    sentences = ["Great film", "The weather is sunny"]
    embeddings = model.encode([anchor] + sentences)
    
    # Calculate similarities
    anchor_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]
    
    similarities = []
    for emb in sentence_embeddings:
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(anchor_embedding).unsqueeze(0),
            torch.tensor(emb).unsqueeze(0)
        )
        similarities.append(similarity.item())
    
    print("Similarities:", similarities)

