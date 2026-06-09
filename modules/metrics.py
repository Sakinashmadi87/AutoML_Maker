import numpy as np
from sklearn.metrics import f1_score
from modules.embedder import embed_texts

def optimize_similarity_threshold(retrieved_contexts, gold_contexts, model_key):
    thresholds = np.linspace(0.30, 0.90, 25)  # 25 Werte zwischen 0.30 und 0.90
    best_threshold = 0.0
    best_f1 = 0.0

    for t in thresholds:
        f1 = compute_retrieval_f1(
            retrieved_contexts=retrieved_contexts,
            gold_contexts=gold_contexts,
            model_key=model_key,
            sim_threshold=t
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1
