import json
import time
import os
from sentence_transformers import SentenceTransformer
from modules.retriever import HybridRetriever
from modules.evaluator import RAGEvaluator

# --- CONFIG ---
# Pfade zu den verschiedenen Datenbank-Ordnern
DB_512_PATH = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
DB_1024_PATH = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
EVAL_FILE = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_1017.jsonl"

def run_comparison():
    print("Lade Modell...")
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')

    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        eval_set = [json.loads(line) for line in f]

    evaluator = RAGEvaluator(eval_set)

    # Hier definieren wir Name, Kollektion UND den Pfad zur jeweiligen DB
    configs_to_test = [
        {"name": "512 Chunks", "coll": "stage2_chunks_modern", "path": DB_512_PATH},
        {"name": "1024 Chunks", "coll": "stage2_chunks_1024", "path": DB_1024_PATH}
    ]

    print(f"\nStarte fairen Vergleich für {len(eval_set)} Fragen...")

    for config in configs_to_test:
        print(f"\n🔍 Teste {config['name']} in Ordner: {config['path']}...")
        
        # WICHTIG: Hier erstellen wir den Retriever für DIESEN Pfad neu
        try:
            retriever = HybridRetriever(config['path'])
            
            start_time = time.time()
            score = evaluator.run_benchmark(
                retriever, 
                model, 
                {
                    "top_k_papers": 20, 
                    "top_k_chunks": 15, 
                    "use_expansion": False,
                    "collection_name": config['coll']
                }
            )
            duration = time.time() - start_time
            
            print(f"✅ Ergebnis {config['name']}: Accuracy {score:.2f}% | Zeit: {duration:.2f}s")
            
            # Verbindung explizit schließen für den nächsten Test (Windows Lock-Schutz)
            del retriever
            
        except Exception as e:
            print(f"❌ Fehler bei {config['name']}: {e}")

    print("\nVergleich abgeschlossen.")

if __name__ == "__main__":
    run_comparison()
