from sklearn.metrics import f1_score
import numpy as np
from modules.embedder import embed_texts

def compute_retrieval_f1(retrieved_contexts, gold_contexts, model_key: str, sim_threshold: float = 0.60):
    """
    retrieved_contexts: Liste von Listen (Top-K Chunks pro Query)
    gold_contexts: Liste von Strings (Antworten / Gold-Standard)
    model_key: Embedding-Modell ('bge-m3' oder 'mxbai-large')
    sim_threshold: Schwelle für "Treffer" in der Cosine-Ähnlichkeit
    """
    y_true = []
    y_pred = []

    for retrieved, gold in zip(retrieved_contexts, gold_contexts):
        if not retrieved or not gold:
            y_true.append(1)
            y_pred.append(0)
            continue

        # Gold-Antwort einbetten
        gold_vec = np.array(embed_texts(gold, model_key=model_key, is_query=True)[0])

        # Alle Chunks einbetten
        chunk_vecs = np.array(embed_texts(retrieved, model_key=model_key, is_query=False))

        # Cosine-Ähnlichkeit zu jedem Chunk
        sims = chunk_vecs @ gold_vec / (
            np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(gold_vec) + 1e-8
        )

        hit = np.max(sims) >= sim_threshold

        y_true.append(1)
        y_pred.append(1 if hit else 0)

    return f1_score(y_true, y_pred)
