# config/paths_config.py

import os
from pathlib import Path

# Zentrale Pfadkonfiguration für das RAG-AutoML-Projekt
PATHS = {
    # Pfad zu deinen 79 generierten Gold-Standard-Fragen (Input aus Schritt A)
    "eval_set_100q": "/kaggle/input/datasets/sakineahmadi/eval-set-100q/eval_set_100q (2).jsonl",
    
    # Quellordner mit den 1965 von Marker parsten wissenschaftlichen Arbeiten
    "markdown": "/kaggle/input/datasets/sakinehahmadi/marker-parsed-papers-1965/marker_parsed_papers",
    
    # Optional: Ein Ordner für deine lokalen Testergebnisse oder Logs
    "output_dir": "/kaggle/working/output"
}

# Automatisches Erstellen des Output-Ordners, falls benötigt
os.makedirs(PATHS["output_dir"], exist_ok=True)

print("📂 [Config] Pfade erfolgreich geladen.")
print(f"   👉 Gold-Standard: {PATHS['eval_set_100q']}")
print(f"   👉 Markdown-Papers: {PATHS['markdown']}")