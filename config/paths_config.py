# config/paths_config.py
import os

# Basis-Verzeichnisse definieren
BASE_WORKING_DIR = "/kaggle/working"

# Zentrale Pfadkonfiguration für das RAG-AutoML-Projekt
PATHS = {
    # Quellordner mit den rohen PDF-Dokumenten für die Massenverarbeitung
    "pdfs_active": "/kaggle/input/datasets/sakinaahmadi/rag-ml-data/pdfs_active",
    
    # Pfad zu deinen 99 vorvalidierten Fragen (Input für den Merge/Retrieval-Schritt)
    "automl_ground_truth_99": "/kaggle/input/datasets/sakinaahmadi/ground-truth-99/automl_ground_truth_100.json",
    
    # Wurzelverzeichnis für Ausgaben und Checkpoints (z.B. checkpoint_parsing.json)
    "output_root": BASE_WORKING_DIR,
    
    # Ordner für deine lokalen AutoML-Testergebnisse, Optuna-Logs oder Matrizen
    "output_dir": os.path.join(BASE_WORKING_DIR, "output")
}

# Automatisches Erstellen benötigter Ordner auf der beschreibbaren Kaggle-Disk
os.makedirs(PATHS["output_dir"], exist_ok=True)

print("📂 [Config] Pfade erfolgreich geladen und mit echten Datensätzen abgeglichen.")
print(f"   👉 PDF-Quelle: {PATHS['pdfs_active']}")
print(f"   👉 Basis-Ground-Truth (99Q): {PATHS['automl_ground_truth_99']}")
print(f"   👉 Output-Ziel: {PATHS['output_dir']}")