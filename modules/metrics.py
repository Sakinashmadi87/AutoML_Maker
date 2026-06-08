from sklearn.metrics import f1_score

def compute_retrieval_f1(retrieved_contexts, gold_contexts):
    """
    retrieved_contexts: Liste von Listen (Top-K Retrieval)
    gold_contexts: Liste von Strings (Ground Truth)
    """
    y_true = []
    y_pred = []

    for retrieved, gold in zip(retrieved_contexts, gold_contexts):
        hit = any(gold.lower() in chunk.lower() for chunk in retrieved)

        y_true.append(1)
        y_pred.append(1 if hit else 0)

    return f1_score(y_true, y_pred)
