# AutoML_Maker (Masterarbeit RAG-Pipeline-Optimierung)

Hallo! Das ist das GitHub-Repository für meine Masterarbeit im Studiengang Informatik. 
Das Ziel dieses Projekts ist es, eine RAG-Pipeline (Retrieval-Augmented Generation) automatisch zu optimieren. Dafür wird **Optuna** für die Hyperparametersuche und **Ragas** für die automatisierte Evaluation der Pipeline-Qualität eingesetzt.

Die wissenschaftlichen Paper wurden vorab mit dem Tool `Marker` aus PDFs in Markdown-Dateien umgewandelt (insgesamt 1.965 Paper).

## 📂 Ordnerstruktur des Projekts

Der Code ist in modulare Python-Skripte aufgeteilt, um eine saubere Trennung der Zuständigkeiten zu gewährleisten:

* **config/**
    * `paths_config.py`: Enthält die Pfade zu den Datensätzen auf Kaggle (wichtig für das korrekte Handling von Sonderzeichen/Klammern in den Dateinamen).
* **modules/**
    * `chunker.py`: Enthält 4 verschiedene Strategien zur Textsegmentierung (Markdown, Semantic, Hierarchical, Recursive).
    * `embedder.py`: Übernimmt die Vektorisierung der Texte inklusive Caching-Mechanismus, um GPU-OOM-Abstürze zu verhindern.
    * `qdrant_storage.py`: Verwaltet die Verbindung zum Remote-Qdrant-Server (nutzt Kaggle Secrets für die Authentifizierung).
* **scripts/**
    * `generate_eval_set.py`: Skript aus Schritt A, das die Testfragen mittels Llama-3 generiert hat.
* **modulare-rag-pipeline/**
    * *(Ehemaliges eigenständiges Repository)*: Enthält vorherige Experimente, Vorstufen der Pipeline-Komponenten sowie Skripte zur Generierung des Gold-Standards und der Ground-Truth-Daten.
* `automl_study.py`: Das Hauptskript. Hier läuft die zentrale Optuna-Optimierungsschleife, die den gesamten Ablauf steuert.

## 🔬 Was wird hier überhaupt optimiert? (Suchraum)

Das Skript `automl_study.py` evaluiert automatisch verschiedene Kombinationen der Pipeline-Komponenten, um den **Ragas F1-Score** zu maximieren:

1. **Embedding-Modelle:**
    * `bge-m3` (Native Ausgabe: 1024 Dimensionen)
    * `mxbai-large` (Native Ausgabe: 1024 Dimensionen)

2. **Chunking-Strategien:**
    * `markdown_section`: Splittet den Text strikt anhand von Markdown-Überschriften (`#`, `##`, `###`).
    * `semantic`: Berücksichtigt semantische Trennungen und optimiert das Splitting für mathematische Brüche und Formeln im Text.
    * `hierarchical`: Nutzt eine Eltern-Kind-Struktur (große Parent-Chunks für den Kontext, kleine Child-Chunks für das Retrieval).
    * `recursive_baseline`: Der standardmäßige, rekursive Textsplitter von LangChain als Baseline.

3. **Dimensionierung:**
    * **Chunk-Größe:** Wahlweise `512` oder `1024` Tokens.
    * **Overlap:** Im Code fest auf `10%` der gewählten Chunk-Größe definiert.

## 📊 Wie funktioniert die Evaluation?

* **Testdatensatz:** Es wird die Datei `eval_set_100q.jsonl` genutzt. Diese enthält die 79 verifizierten Gold-Standard-Fragen, die fehlerfrei generiert wurden.
* **Stichproben-Validierung:** Da die Indexierung aller 1.965 Paper in jedem einzelnen Optuna-Trial zu zeitaufwändig und cloud-kostenintensiv wäre, zieht das Skript pro Durchlauf eine **Zufallsstichprobe von 20 Papern**.
* **Ressourcen-Schonung:** Nach der Evaluation jedes Einzeldurchlaufs wird die temporäre Collection in Qdrant vollständig gelöscht, um den Cloud-Speicher nicht zu überlasten.

## 🚀 Erste Schritte

### 1. Bibliotheken installieren
Die benötigten Abhängigkeiten können über das Terminal oder direkt in einer Notebook-Zelle installiert werden:

```bash
pip install optuna ragas qdrant-client sentence-transformers langchain numpy
```
