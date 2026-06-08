# automl_study.py

import json
import random
import optuna
from pathlib import Path

# Ragas Framework für die wissenschaftliche Evaluierung
from ragas import EvaluationDataset
from ragas.metrics import F1Score
from ragas import evaluate

# Injektion deiner modularen Pipeline-Bausteine aus dem Repository
from config.paths_config import PATHS
from modules.chunker import chunk_markdown_section, chunk_semantic, chunk_hierarchical, chunk_recursive
from modules.embedder import embed_texts
from modules.qdrant_storage import QdrantManager

# ---------------------------------------------------------------------
# 1. SETUPS & DATEN-PREPARATION
# ---------------------------------------------------------------------
QDRANT = QdrantManager()
GROUND_TRUTH_PATH = Path(PATHS.get("eval_set_100q", "/kaggle/working/eval_set_100q.jsonl"))
MARKER_DIR = Path(PATHS.get("markdown", "/kaggle/input/datasets/sakinehahmadi/marker-parsed-papers-1965/marker_parsed_papers"))

def load_ground_truth(path: Path) -> list:
    """Lädt die 79 erfolgreich generierten Paare für die Validierung."""
    dataset = []
    if not path.exists():
        raise FileNotFoundError(f"❌ eval_set_100q.jsonl nicht gefunden unter {path}!")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line.strip()))
    return dataset

# Globale Bereitstellung der Testfragen für alle Trials
EVAL_DATA = load_ground_truth(GROUND_TRUTH_PATH)

# WICHTIG FÜR DIE PERFORMANCE: Wir wählen eine feste Teilmenge (z.B. 20 Papers) 
# für das AutoML-Tuning aus, um die Schleife extrem zu beschleunigen.
ALL_MD_FILES = list(MARKER_DIR.glob("**/*.md"))
random.seed(42)  # Reproduzierbarkeit sichern
TUNING_FILES = random.sample(ALL_MD_FILES, min(20, len(ALL_MD_FILES)))

print(f"📊 Gold-Standard geladen: {len(EVAL_DATA)} Testfragen.")
print(f"🔬 AutoML-Tuning läuft über eine repräsentative Stichprobe von {len(TUNING_FILES)} Papers.")

# ---------------------------------------------------------------------
# 2. THE OPTUNA OBJECTIVE FUNCTION (Die AutoML-Schleife)
# ---------------------------------------------------------------------
def objective(trial):
    """
    Diese Funktion wird von Optuna in jedem Durchlauf (Trial) aufgerufen.
    """
    # ─── HYPERPARAMETER SUCHRAUM DEFINIEREN ───
    embedding_model_param = trial.suggest_categorical("embedding_model", [
        "bge-m3", 
        "gte-qwen2", 
        "mxbai-large"
    ])
    
    chunk_strategy_param = trial.suggest_categorical("chunk_strategy", [
        "markdown_section", 
        "semantic", 
        "hierarchical", 
        "recursive_baseline"
    ])
    
    chunk_size_param = trial.suggest_categorical("chunk_size", [512, 1024])
    overlap_param = int(chunk_size_param * 0.10)  # Festgelegte 10% Overlap laut Vorgabe
    
    print(f"\n" + "═"*60)
    print(f"🚀 STARTING TRIAL #{trial.number}")
    print(f"📦 Embedding: {embedding_model_param}")
    print(f"🧩 Strategie: {chunk_strategy_param} | Size: {chunk_size_param} (Overlap: {overlap_param})")
    print("═"*60)
    
    # Eindeutiger Name für die temporäre Remote-Collection
    collection_name = f"automl_trial_{trial.number}_{chunk_strategy_param}_{chunk_size_param}"
    
    try:
        # ─── SCHRITT 1: CHUNKING DER TEXTE ───
        trial_chunks = []
        for file_path in TUNING_FILES:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # Dynamischer Aufruf der gewählten Strategie
            if chunk_strategy_param == "markdown_section":
                chunks = chunk_markdown_section(raw_text, chunk_size_param, overlap_param)
            elif chunk_strategy_param == "semantic":
                chunks = chunk_semantic(raw_text, chunk_size_param, overlap_param)
            elif chunk_strategy_param == "hierarchical":
                chunks = chunk_hierarchical(raw_text, chunk_size_param, overlap_param)
            else:
                chunks = chunk_recursive(raw_text, chunk_size_param, overlap_param)
                
            trial_chunks.extend(chunks)
            
        # FIX 3 & 4: Filterung unbrauchbarer Chunks & Leerraum-Bereinigung vor dem Embedding
        trial_chunks = [c for c in trial_chunks if isinstance(c, str) and len(c.strip()) > 20]
        
        if len(trial_chunks) == 0:
            print("⚠️ Keine validen Chunks nach der Bereinigung generiert. Überspringe Trial.")
            return 0.0

        # ─── SCHRITT 2: EMBEDDING-GENERIERUNG (INGESTION) ───
        # Chunks einbetten (is_query=False). Liefert garantiert 1024 Dimensionen.
        print(f"🧠 Erzeuge Chunks-Embeddings für {len(trial_chunks)} valide Segmente...")
        trial_vectors = embed_texts(trial_chunks, model_key=embedding_model_param, is_query=False)

        # ─── SCHRITT 3: REMOTE QDRANT INDEXAUFBAU ───
        QDRANT.create_trial_collection(collection_name=collection_name, vector_size=1024)
        QDRANT.upload_chunks(collection_name=collection_name, chunks=trial_chunks, vectors=trial_vectors)

        # ─── SCHRITT 4: RETRIEVAL & VALIDIERUNG ───
        ragas_user_queries = []
        ragas_retrieved_contexts = []
        ragas_ground_truths = []
        
        print(f"🔍 Starte Retrieval-Testlauf gegen die Gold-Standard-Fragen...")
        for item in EVAL_DATA:
            query = item["query"]
            gold_answer = item["ground_truth"]
            
            # FIX 1 & 2: Query korrekt als Liste kapseln und ersten Vektor herauspicken
            query_vector = embed_texts([query], model_key=embedding_model_param, is_query=True)[0]
            
            # Top-3 Treffer aus dem Remote-Index abrufen
            retrieved_context = QDRANT.search_retriever(
                collection_name=collection_name, 
                query_vector=query_vector, 
                top_k=3
            )
            
            ragas_user_queries.append(query)
            ragas_retrieved_contexts.append(retrieved_context)
            ragas_ground_truths.append(gold_answer)

        # ─── SCHRITT 5: RAGAS EVALUIERUNG DER METRIK ───
        # Dataset für das Ragas-Framework strukturieren
        evaluation_dataset = EvaluationDataset.from_list([
            {"user_input": q, "retrieved_contexts": ctx, "reference": gt} 
            for q, ctx, gt in zip(ragas_user_queries, ragas_retrieved_contexts, ragas_ground_truths)
        ])
        
        print("📊 Berechne Ragas F1-Score für das Retrieval...")
        result = evaluate(dataset=evaluation_dataset, metrics=[F1Score()])
        f1_score = float(result["f1_score"])
        
        print(f"🎯 Trial #{trial.number} erfolgreich beendet mit Ragas F1-Score: {f1_score:.4f}")

    except Exception as e:
        print(f"❌ Fehler im Trial #{trial.number}: {e}")
        f1_score = 0.0  # Bestrafung des Trials bei Absturz/Laufzeitfehlern
        
    finally:
        # KRITISCH: Unbedingt die Collection auf dem Cloud-Server löschen, 
        # um den Speicherplatz sofort wieder freizugeben!
        QDRANT.delete_trial_collection(collection_name)
        
    return f1_score

# ---------------------------------------------------------------------
# 3. START DES AUTOMATIONSEXPERIMENTS
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "★"*60)
    print("      STARTE AUTOMATED HYPERPARAMETER OPTIMIZATION (AutoML)")
    print("★"*60)
    
    # Bayesian Optimization (TPE Sampler) zur intelligenten Pfadfindung im Suchraum
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    
    # 10 vollständige Durchläufe starten (für eine tiefere Optimierung auf 20-30 erhöhen)
    study.optimize(objective, n_trials=10)
    
    print("\n" + "🏆"*20)
    print("🏆 OPTIMIERUNG FÜR MASTERARBEIT ERFOLGREICH BEENDET! 🏆")
    print("🏆"*20)
    
    print("\n🏅 Die absolut beste Konfiguration:")
    for param_name, param_value in study.best_params.items():
        print(f"   👉 {param_name}: {param_value}")
        
    print(f"\n🏅 Höchster wissenschaftlich erzielter Ragas F1-Score: {study.best_value:.4f}")