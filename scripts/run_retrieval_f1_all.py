# scripts/run_retrieval_f1_all.py
import json
from modules.embedder import embed_texts
from modules.metrics import optimize_similarity_threshold
from modules.qdrant_multi import MultiQdrantManager

manager = MultiQdrantManager()

collections = [
    "marker_hierarchical_1024",
    "stage2_docling_hybrid_bge_int8",
    "stage2_docling_hybrid_bge_f32",
    "pymupdf4llm_1024",
    "docling_hierarchical_512"
]

embedding_model = "bge-m3"
top_k = 10

with open("/kaggle/working/final_ground_truth_with_context.json", "r") as f:
    eval_data = json.load(f)

results = []

for col in collections:
    print(f"\n🔍 Evaluating Collection: {col}")

    retrieved_contexts = []
    gold_contexts = []

    for item in eval_data:
        query = item["query"]
        gold = item["ground_truth"]

        query_vec = embed_texts([query], model_key=embedding_model, is_query=True)[0]
        hits = manager.search(col, query_vec, top_k)

        retrieved_contexts.append(hits)
        gold_contexts.append(gold)

    thr, f1 = optimize_similarity_threshold(
        retrieved_contexts=retrieved_contexts,
        gold_contexts=gold_contexts,
        model_key=embedding_model
    )

    results.append((col, thr, f1))
    print(f"➡️ threshold={thr:.3f} | F1={f1:.4f}")

print("\n📊 FINAL RESULTS")
for col, thr, f1 in results:
    print(f"{col:35} | threshold={thr:.3f} | F1={f1:.4f}")
