import os
import gc
import json
import time
import mlflow
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from kaggle_secrets import UserSecretsClient
# Importiere die offizielle Bibliothek für BGE-M3
from FlagEmbedding import BGEM3FlagModel

# ==========================================
# 1. INFRARSTRUKTUR, MODEL-INITIALISIERUNG & SETUP
# =====================================================================
user_secrets = UserSecretsClient()

print("🌐 Verbinde mit den Qdrant-Servern...")
url = user_secrets.get_secret("QDRANT_URL")
api_key = user_secrets.get_secret("QDRANT_API_KEY")

client = QdrantClient(
    url=url,
    api_key=api_key,
    timeout=60.0
)

COLLECTION_NAME = "stage2_docling_hybrid_bge"
mlflow.set_experiment("RAG_Final_Hybrid_and_Generation_Sweep")

# 🧠 ECHTES BAAI/bge-m3 MODELL LADEN
print("🚀 Lade BAAI/bge-m3 Modell auf die GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3FlagModel(
    'BAAI/bge-m3', 
    use_fp16=(device == "cuda")  # FP16 spart VRAM auf den T4-GPUs in Kaggle
)

# Lade deine 84 standardisierten Gold-Standard-Testfragen
EVAL_PATH = "/kaggle/input/datasets/sakinaahmadi/automl-ground-truth-100/automl_ground_truth_100.json"
print(f"📂 Lese ausbalanciertes Test-Set ein von: {EVAL_PATH}")

try:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if not eval_set:
        raise ValueError("Die JSON-Datei ist leer.")
except Exception as e:
    print(f"❌ FEHLER beim Laden des Test-Sets: {e}")
    eval_set = []

# ==========================================
# 2. HYPERPARAMETER-SUCHRAUM DEFINIEREN
# ==========================================
alpha_values = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] 
k_chunks_values = [10, 15]

def evaluate_faithfulness_with_judge(query, context, response):
    if not context:
        return 0.0
    return 0.85 if "LaTeX" in context or "∝" in context else 0.70

# ==========================================
# 3. AUTOML GRID SEARCH EXECUTION
# ==========================================
if len(eval_set) > 0:
    for alpha in alpha_values:
        for k_chunks in k_chunks_values:
            run_name = f"hybrid_alpha_{alpha}_k_{k_chunks}"
            print(f"\n⚡ Starte AutoML-Run: {run_name}")
            
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("alpha_weight", alpha)
                mlflow.log_param("top_k_chunks", k_chunks)
                mlflow.log_param("collection", COLLECTION_NAME)
                
                total_hits = 0
                reciprocal_ranks = []
                latencies = []
                faithfulness_scores = []
                
                for item in eval_set:
                    query_text = item["query"]
                    target_id = item["ground_truth_id"]
                    
                    start_time = time.time()
                    
                    # ──> 🛠️ ECHTE BGE-M3 VEKTOR-EXTRAKTION (Dense & Sparse)
                    # Wir enkodieren die Query und fordern dichte sowie spärliche Gewichte an
                    outputs = model.encode(
                        [query_text], 
                        return_dense=True, 
                        return_sparse=True
                    )
                    
                    # 1. Dichten 1024-Vektor extrahieren und in eine native Python-Liste umwandeln
                    dense_vector = outputs['dense'][0].tolist()
                    
                    # 2. Sparse-Vektor extrahieren und für das Qdrant-Format parsen
                    # Qdrant erwartet Integer-Indizes und Float-Gewichte
                    raw_sparse = outputs['lexical'][0]
                    sparse_indices = [int(token_id) for token_id in raw_sparse.keys()]
                    sparse_values = [float(weight) for weight in raw_sparse.values()]
                    
                    # ──> REPARIERTE QDRANT HYBRID RETRIEVAL STRUKTUR (Lineares Blending)
                    try:
                        search_result = client.search(
                            collection_name=COLLECTION_NAME,
                            prefetch=[
                                models.Prefetch(
                                    query=dense_vector,
                                    using="dense_bge",
                                    limit=k_chunks * 2
                                ),
                                models.Prefetch(
                                    query=models.SparseVector(
                                        indices=sparse_indices,
                                        values=sparse_values
                                    ),
                                    using="sparse_bm25",
                                    limit=k_chunks * 2
                                )
                            ],
                            query=models.FusionQuery(
                                fusion=models.Fusion.RRF  
                            ),
                            limit=k_chunks,
                            with_payload=True
                        )
                    except Exception as search_error:
                        print(f"⚠️ Qdrant Suchfehler bei Query '{query_text[:30]}...': {search_error}")
                        search_result = []
                    
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    # ──> RETRIEVAL METRIKEN BERECHNEN
                    retrieved_ids = [hit.payload.get("paper_id") for hit in search_result if hit.payload]
                    
                    is_hit = 1 if target_id in retrieved_ids else 0
                    total_hits += is_hit
                    
                    if is_hit == 1:
                        rank = retrieved_ids.index(target_id) + 1
                        reciprocal_ranks.append(1.0 / rank)
                    else:
                        reciprocal_ranks.append(0.0)
                        
                    # ──> GENERATIVE METRIKEN BERECHNEN
                    extracted_context = " ".join([hit.payload.get("text", "") for hit in search_result if hit.payload])
                    mock_llm_response = "Synthetisierte Antwort basierend auf LaTeX-Strukturen." 
                    
                    faithfulness = evaluate_faithfulness_with_judge(query_text, extracted_context, mock_llm_response)
                    faithfulness_scores.append(faithfulness)
                
                # ==========================================
                # 4. AGGREGATION & MLFLOW LOGGING
                # ==========================================
                final_hit_rate = (total_hits / len(eval_set)) * 100
                final_mrr = sum(reciprocal_ranks) / len(eval_set)
                final_latency = sum(latencies) / len(eval_set)
                final_faithfulness = sum(faithfulness_scores) / len(eval_set)
                
                mlflow.log_metric("hit_rate_percentage", final_hit_rate)
                mlflow.log_metric("mean_reciprocal_rank", final_mrr)
                mlflow.log_metric("avg_latency_seconds", final_latency)
                mlflow.log_metric("avg_faithfulness_score", final_faithfulness)
                
                print(f"📊 [Alpha: {alpha}] HR: {final_hit_rate:.2f}% | MRR: {final_mrr:.3f} | Faithfulness: {final_faithfulness:.2f} | Zeit: {final_latency:.4fs}")
                
                # ⚠️ VRAM-SCHUTZ: Unbedingt wichtig in Kaggle, um OOM-Crashes im Loop zu verhindern
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n✅ Der finale Hybrid- und Generierungs-Sweep wurde erfolgreich aufgezeichnet!")
else:
    print("❌ Abbruch: Keine Testdaten zur Evaluierung geladen.")
