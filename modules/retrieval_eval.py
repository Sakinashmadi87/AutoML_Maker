# modules/retrieval_eval.py
from modules.embedder import embed_texts
from modules.metrics import optimize_similarity_threshold
from modules.qdrant_storage import QdrantManager

QDRANT = QdrantManager()

def evaluate_collection_f1(collection_name: str, eval_data: list, embedding_model: str, top_k: int = 10):
    """
    Berechnet Retrieval-F1 für eine gegebene Collection.
    """
    retrieved_contexts = []
    gold_contexts = []

    for item in eval_data:
        query = item["query"]
        gold = item["ground_truth"]

        # Query embedding
        query_vec = embed_texts([query], model_key=embedding_model, is_query=True)[0]

        # Retrieval
        hits = QDRANT.search_retriever(
            collection_name=collection_name,
            query_vector=query_vec,
            top_k=top_k
        )

        retrieved_contexts.append(hits)
        gold_contexts.append(gold)

    # F1-Berechnung
    best_threshold, f1_score = optimize_similarity_threshold(
        retrieved_contexts=retrieved_contexts,
        gold_contexts=gold_contexts,
        model_key=embedding_model
    )

    return best_threshold, f1_score
