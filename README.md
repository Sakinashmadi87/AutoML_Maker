# AutoML_Maker (Masterarbeit RAG Pipeline Optimierung)

Hallo! Das ist das GitHub-Repository für meine Masterarbeit im Studiengang Informatik. 
Das Ziel von diesem Projekt ist es, eine RAG-Pipeline (Retrieval-Augmented Generation) automatisch zu optimieren. Ich benutze dafür **Optuna** für das Suchen der besten Parameter und **Ragas** für die Evaluation.

Die wissenschaftlichen Paper wurden vorher mit dem Tool `Marker` aus PDFs in Markdown-Dateien umgewandelt (insgesamt 1965 Papers).

## 📁 Ordnerstruktur von meinem Projekt

Ich habe den Code jetzt in Module aufgeteilt, damit es übersichtlicher ist als in einem großen Jupyter Notebook:

* **config/**
    * `paths_config.py`: Hier stehen die Pfade zu den Datensätzen auf Kaggle (sehr wichtig wegen den Klammern im Dateinamen!).
* **modules/**
    * `chunker.py`: Enthält die 4 verschiedenen Strategien, um den Text zu schneiden (Markdown, Semantic, Hierarchical, Recursive).
    * `embedder.py`: Hier werden die Texte in Vektoren umgewandelt (mit Cache, damit die GPU nicht abstürzt).
    * `qdrant_storage.py`: Code für die Verbindung zum Remote Qdrant Server (nutzt Kaggle Secrets für das Passwort).
* **scripts/**
    * `generate_eval_set.py`: Das Skript aus Schritt A, das die 79 Testfragen mit Llama-3 generiert hat.
* `automl_study.py`: Das Hauptskript. Hier läuft die Optuna-Schleife, die alles steuert.

## 🔬 Was wird hier überhaupt optimiert? (Suchraum)

Das Skript `automl_study.py` probiert automatisch verschiedene Kombinationen aus, um zu gucken, wo der beste Ragas F1-Score herauskommt:

1.  **Embedding-Modelle:**
    * `bge-m3` (macht nativ 1024 Dimensionen)
    * `gte-qwen2` (Achtung: Macht eigentlich 1536 Dimensionen, aber im Code schneide ich das über Matryoshka auf 1024 ab und normalisiere neu!)
    * `mxbai-large` (macht nativ 1024 Dimensionen)
2.  **Chunking-Strategien:**
    * `markdown_section` (splittet nach Überschriften #, ##, ###)
    * `semantic` (guckt nach mathematischen Brüchen im Text)
    * `hierarchical` (große Parent-Chunks und kleine Child-Chunks)
    * `recursive_baseline` (der Standard-Splitter von LangChain)
3.  **Größe:**
    * Chunk-Größe: Entweder `512` oder `1024` Tokens.
    * Overlap: Ist im Code fest auf `10%` von der Chunk-Größe eingestellt.

## 📊 Wie funktioniert die Evaluation?

* Ich benutze das `eval_set_100q.jsonl` (da sind die 79 Gold-Standard-Fragen drin, die fehlerfrei generiert wurden).
* Weil das Testen mit allen 1965 Papers in jedem Optuna-Schritt viel zu lange dauern würde (und teuer auf dem Server ist), nimmt das Skript im Moment eine **Zufallsstichprobe von 20 Papers** (`TUNING_FILES`).
* Am Ende wird die Collection in Qdrant immer wieder gelöscht, damit der Cloud-Speicher nicht voll wird.

## 🚀 Wie man das Projekt startet

### 1. Librarys installieren
Man braucht ein paar Pakete. Am besten vorher im Terminal oder in einer Notebook-Zelle installieren:
```bash
pip install optuna ragas qdrant-client sentence-transformers langchain numpy