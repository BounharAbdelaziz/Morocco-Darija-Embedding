import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from typing import Dict, Tuple
import time
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_model(model: SentenceTransformer, dataset, batch_size: int = 32) -> Dict:
    """
    Evaluate a sentence transformer model and return metrics and predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_similarities = []
    benchmark_scores = []
    total_time = 0
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        
        start_time = time.time()
        embeddings1 = model.encode(batch['sentence1'], convert_to_tensor=True)
        embeddings2 = model.encode(batch['sentence2'], convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
        all_similarities.extend(similarities)
        benchmark_scores.extend(batch['score'])
    
    spearman_corr, _ = spearmanr(all_similarities, benchmark_scores)
    pearson_corr, _ = pearsonr(all_similarities, benchmark_scores)
    mse = mean_squared_error(benchmark_scores, all_similarities)
    
    return {
        'spearman_correlation': spearman_corr,
        'pearson_correlation': pearson_corr,
        'mse': mse,
        'avg_processing_time': total_time / len(dataset),
        'total_pairs': len(dataset),
        'predictions': all_similarities,
        'ground_truth': benchmark_scores
    }

if __name__ == "__main__":
    
    MODELS = [
        "BounharAbdelaziz/Morocco-Darija-Sentence-Embedding",
        "BounharAbdelaziz/Morocco-Darija-Static-Sentence-Embedding",
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3"
    ]
    
    print("Loading benchmark dataset...")
    bench = load_dataset("atlasia/Morocco-Darija-Sentence-Embedding-Benchmark", split='test')
    
    results = []
    predictions_df = pd.DataFrame()
    predictions_df['ground_truth'] = bench['score']
    
    for model_name in MODELS:
        print(f"\nEvaluating {model_name}...")
        try:
            model = SentenceTransformer(model_name)
            metrics = evaluate_model(model, bench)
            
            # Store metrics
            results.append({
                'model': model_name,
                'spearman_correlation': metrics['spearman_correlation'],
                'pearson_correlation': metrics['pearson_correlation'],
                'mse': metrics['mse'],
                'avg_processing_time': metrics['avg_processing_time'],
                'total_pairs': metrics['total_pairs']
            })
            
            # Store predictions
            short_name = model_name.split('/')[-1]
            predictions_df[short_name] = metrics['predictions']
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
    
    # Create DataFrames
    metrics_df = pd.DataFrame(results)
    metrics_df['spearman_correlation'] = metrics_df['spearman_correlation'].apply(lambda x: f"{x:.4f}")
    metrics_df['pearson_correlation'] = metrics_df['pearson_correlation'].apply(lambda x: f"{x:.4f}")
    metrics_df['mse'] = metrics_df['mse'].apply(lambda x: f"{x:.4f}")
    metrics_df['avg_processing_time'] = metrics_df['avg_processing_time'].apply(lambda x: f"{x*1000:.2f}")
    
    # Print results
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    print("\nPredictions Analysis:")
    print("\nPrediction Ranges:")
    for column in predictions_df.columns:
        min_val = predictions_df[column].min()
        max_val = predictions_df[column].max()
        std_val = predictions_df[column].std()
        print(f"{column:>35}: min={min_val:.4f}, max={max_val:.4f}, std={std_val:.4f}")

    # Create visualizations
    # create_visualizations(metrics_df, predictions_df)
    
    # Save data for the React dashboard
    dashboard_data = {
        'metrics': metrics_df.to_dict('records'),
    }
    
    with open('eval_results.json', 'w') as f:
        json.dump(dashboard_data, f)