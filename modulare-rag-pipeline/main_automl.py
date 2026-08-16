import json
import csv
import os
import sys
import torch
import time  # For tracking latency
import mlflow  # <-- NEW: Scientific Tracking Layer
from sentence_transformers import SentenceTransformer
from modules.evaluator import RAGEvaluator

sys.path.append(os.getcwd())
from config import HP_GRID, PATHS, IS_COLAB, IS_KAGGLE

# --- FINAL GRID SETUP ---
HP_GRID_FINAL = {
    "top_k_papers": [15, 20, 25],
    "top_k_chunks": [10, 15, 20],
    "use_expansion": [False]
}

def get_cloud_credentials():
    url, api_key = None, None
    if IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            url = user_secrets.get_secret("QDRANT_URL")
            api_key = user_secrets.get_secret("QDRANT_API_KEY")
        except Exception: pass
    url = url or os.getenv('QDRANT_URL')
    api_key = api_key or os.getenv('QDRANT_API_KEY')
    return url, api_key

def main():
    # --- 1. SET UP MLFLOW EXPERIMENT ---
    mlflow.set_experiment("Wissenschaftliche_RAG_Pipeline_AutoML")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Initializing Embedding Model (BGE-M3) on: {device.upper()}")
    model = SentenceTransformer('BAAI/bge-m3', device=device)

    from modules.retriever import HybridRetriever
    qdrant_url, qdrant_key = get_cloud_credentials()
    
    if qdrant_url and qdrant_key:
        print("🌐 Connecting to Qdrant Cloud Cluster...")
        retriever = HybridRetriever(url=qdrant_url, api_key=qdrant_key)
    else:
        print("🏠 Connecting to local Qdrant...")
        retriever = HybridRetriever()

    eval_path = PATHS["eval_set"]
    print(f"📂 Loading evaluation set from: {eval_path}")
    
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_set = [json.loads(line) for line in f]
    evaluator = RAGEvaluator(eval_set)

    results = []

    print(f"🚀 Starting Final AutoML Loop for {len(eval_set)} questions...")
    
    # --- 2. START THE SYSTEMatic AUTO_ML LOOP ---
    for k_p in HP_GRID_FINAL["top_k_papers"]:
        for k_c in HP_GRID_FINAL["top_k_chunks"]:
            
            # Start a nested tracking run for this specific parameter combination
            run_name = f"Papers_{k_p}_Chunks_{k_c}"
            with mlflow.start_run(run_name=run_name, nested=True):
                
                config = {
                    "top_k_papers": k_p,
                    "top_k_chunks": k_c,
                    "use_expansion": False,
                    "collection_name": "stage2_chunks_1024"
                }
                
                # Metric calculation with latency tracking
                start_time = time.time()
                score = evaluator.run_benchmark(retriever, model, config)
                latency = time.time() - start_time
                
                print(f"📊 Grid: Papers={k_p}, Chunks={k_c} -> Accuracy: {score:.2f}% | Latency: {latency:.2f}s")
                
                # --- 3. LOG METRICS TO MLFLOW RESPOSITORY ---
                mlflow.log_params(config)
                mlflow.log_metric("accuracy", score)
                mlflow.log_metric("latency_seconds", latency)
                
                results.append({**config, "accuracy": score, "latency": latency})

    # Save local CSV backup
    results_file = PATHS.get("results_csv", "final_optimization_results.csv")
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("✅ AutoML-Run with MLflow Tracking finished successfully!")

if __name__ == "__main__":
    main()
