import numpy as np
from sklearn.metrics import f1_score
from modules.embedder import embed_texts

# ---------------------------------------------------------
# 1. EINZELNER F1 SCORE FÜR EINEN FIXEN THRESHOLD
# ---------------------------------------------------------
def compute_retrieval_f1(retrieved_contexts, gold_contexts, model_key, sim_threshold=0.60):
    y_true = []
    y_pred = []

    for retrieved, gold in zip(retrieved_contexts, gold_contexts):
        y_true.append(1)

        if not retrieved or not gold:
            y_pred.append(0)
            continue

        # Gold-Antwort einbetten
        gold_vec = np.array(embed_texts(gold, model_key=model_key, is_query=True)[0])

        # Chunks einbetten
        chunk_vecs = np.array(embed_texts(retrieved, model_key=model_key, is_query=False))

        # Cosine Similarity
        sims = chunk_vecs @ gold_vec / (
            np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(gold_vec) + 1e-8
        )

        hit = np.max(sims) >= sim_threshold
        y_pred.append(1 if hit else 0)

    return f1_score(y_true, y_pred)


# ---------------------------------------------------------
# 2. AUTOMATISCHE THRESHOLD-OPTIMIERUNG
# ---------------------------------------------------------
def optimize_similarity_threshold(retrieved_contexts, gold_contexts, model_key):
    thresholds = np.linspace(0.30, 0.90, 25)
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
