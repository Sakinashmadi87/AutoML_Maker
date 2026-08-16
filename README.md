# AutoML_Maker – AutoML-gestützte Optimierung von RAG-Pipelines für wissenschaftliche Literatur

**Masterarbeit im Studiengang Informatik**  
**Autorin:** Sakina Ahmadi  
**Matrikelnummer:** 10016542  
**Betreuer:** Prof. Dr. rer. nat. Marius Lindauer (LUH)  
**Institut:** Institut für Künstliche Intelligenz (LUHAI), Leibniz Universität Hannover  
**Datum:** 2026

---

## 🎯 Projektübersicht

Ziel dieses Projekts ist die **automatische Optimierung einer RAG-Pipeline** (Retrieval-Augmented Generation) für wissenschaftliche Literatur. Im Fokus stehen:

- **Optuna** – Hyperparameter-Optimierung  
- **RAGAS** – Automatisierte Evaluierung der Pipeline-Qualität  
- **Drei Parser** – PyMuPDF4LLM, Docling, Marker  
- **Vier Chunking-Strategien** – Recursive, Semantic, Hierarchical, Hybrid  
- **Zwei Embedding-Modelle** – BGE‑M3, mxbai‑large  
- **AutoML-Optimierung** – 50 Trials über 3 Seeds  
- **INT8-Quantisierung** – Speicher- und Latenzoptimierung  
- **Reranking** – Cross-Encoder zur Nachfilterung  

Die zugrundeliegende Datenbasis besteht aus **3.734 ArXiv-Publikationen** (2023–2025), die mit den genannten Parsern aus PDFs in Markdown konvertiert wurden.

---

## 📂 Ordnerstruktur
AutoML_Maker/
├── config/
│ └── paths_config.py # Pfade zu Kaggle-Datasets
├── modules/
│ ├── chunker.py # 4 Chunking-Strategien
│ ├── embedder.py # Embedding mit Caching
│ └── qdrant_storage.py # Qdrant-Verbindung
├── scripts/
│ ├── generate_eval_set.py # Testfragen-Generierung (Llama-3)
│ ├── parse_and_clean_docs.py # PDF → Markdown (CPU-intensiv)
│ └── upload_to_qdrant.py # Markdown → Qdrant (GPU-intensiv)
├── automl_study.py # Hauptskript (Optuna-Optimierung)
├── requirements.txt # Abhängigkeiten
└── README.md


---

## 🔬 Optimierungs-Suchraum

Das Skript `automl_study.py` durchsucht automatisch verschiedene Kombinationen, um den **RAGAS F1‑Score** zu maximieren.

### 1. Parser (3 Optionen)

| Parser          | Ansatz        | Formel-Erkennung | Layout-Erhaltung | MRR    |
|-----------------|---------------|------------------|------------------|--------|
| **PyMuPDF4LLM** | Regelbasiert  | 0 %              | 50 %             | 0,4718 |
| **Docling**     | Vision-basiert| **95 %**         | **90 %**         | **0,6006** |
| **Marker**      | Hybrid        | 100 %            | 70 %             | 0,4638 |

### 2. Embedding-Modelle (2 Optionen)

| Modell          | Kontextfenster | MRR    | Hit Rate@1 |
|-----------------|----------------|--------|------------|
| **BGE‑M3**      | 8192 Tokens    | **0,6006** | **54,55 %** |
| **mxbai‑large** | 512 Tokens     | 0,4638 | 37,37 %    |

### 3. Chunking-Strategien (4 Optionen)

| Strategie            | Beschreibung                      | MRR    |
|----------------------|-----------------------------------|--------|
| **Recursive Baseline** | Standard-LangChain-Splitter     | 0,4718 |
| **Semantic**         | Semantische Trennungen            | 0,5907 |
| **Hierarchical**     | Eltern-Kind-Struktur              | 0,5020 |
| **Hybrid**           | Markdown + Rekursiv               | **0,6006** |

### 4. Weitere Hyperparameter

| Parameter         | Suchraum           | Beschreibung                          |
|-------------------|-------------------|----------------------------------------|
| **Chunk-Größe**   | 512 / 1024 Tokens | Textsegmentierung                     |
| **Overlap**       | 10 % der Größe    | Überlappung zwischen Chunks           |
| **Top‑K**         | 3–15              | Anzahl zurückgegebener Dokumente      |
| **Score-Threshold**| 0,35–0,75        | Mindest-Ähnlichkeitsschwelle          |
| **HNSW‑EF**       | 64–256            | Qdrant-Suchgenauigkeit                |
| **Reranking**     | True / False      | Cross-Encoder-Nachfilterung           |

---

## 📊 Ergebnisse

### Retrieval-Leistung (MRR)

| Konfiguration          | MRR    | Hit Rate@10 |
|------------------------|--------|-------------|
| Heuristische Baseline  | 0,4718 | 58,59 %     |
| Manuelle Optimierung   | 0,5907 | 70,71 %     |
| **AutoML-optimiert**   | **0,6470** | **71,72 %** |

### AutoML-Verbesserung

| Metrik         | Vorher  | Nachher  | Verbesserung |
|----------------|---------|----------|--------------|
| **MRR**        | 0,4718  | **0,6470** | **+37,1 %** |
| **Hit Rate@10**| 58,59 % | **71,72 %** | **+18,4 %** |

### Stabilität über 3 Seeds

| Seed  | Parser         | Chunking     | Größe | Embedding     | F1      |
|-------|----------------|--------------|-------|---------------|---------|
| 42    | Docling        | Semantic     | 512   | mxbai-large   | **0,9949** |
| 1337  | PyMuPDF4LLM    | Hierarchical | 512   | mxbai-large   | **0,9949** |
| 2026  | PyMuPDF4LLM    | Semantic     | 1024  | mxbai-large   | **0,9949** |

### Threshold‑Sweep

| Iteration | Threshold-Bereich | Max. F1‑Score |
|-----------|-------------------|---------------|
| 1         | 0,55 – 0,85       | 0,7258        |
| 2         | 0,30 – 0,85       | **1,0**       |
| 3         | 0,40 – 0,65       | 0,9936        |

### Quantisierung (INT8 vs. Float32)

| Metrik          | Float32 | INT8    | Differenz      |
|-----------------|---------|---------|----------------|
| **F1‑Score**    | 0,4731  | 0,4646  | −1,8 %         |
| **Rechenzeit**  | 3,9 h   | **0,04 h** | **100× schneller** |
| **RAM**         | 213 MB  | 898 MB  | +685 MB        |

### Reranking

| Metrik         | Ohne Reranking | Mit Reranking | Verbesserung   |
|----------------|----------------|---------------|----------------|
| **MRR**        | 0,6006         | **0,6470**    | +0,0474        |
| **Hit Rate@1** | 54,55 %        | **60,61 %**   | +6,06 %        |

---

## 🛠️ Installation

1. **Abhängigkeiten installieren**  
   ```bash
   pip install optuna ragas qdrant-client sentence-transformers langchain numpy
2. **Kaggle-Secrets einrichten (für Qdrant-Zugang)**
   
      from kaggle_secrets import UserSecretsClient
      user_secrets = UserSecretsClient()
      url = user_secrets.get_secret("QDRANT_URL")
      api_key = user_secrets.get_secret("QDRANT_API_KEY")
   
3. **Repository klonen**
 
      git clone https://github.com/Sakinashmadi87/AutoML_Maker.git
      cd AutoML_Maker

Verwendung
----------

1.  bashpython scripts/parse\_and\_clean\_docs.py
    
2.  bashpython scripts/upload\_to\_qdrant.py
    
3.  bashpython automl\_study.py
    

📊 Evaluation
-------------

*   **Testdatensatz:** eval\_set\_100q.jsonl (99 Gold‑Standard‑Fragen)
    
*   **Stichproben‑Validierung:** 20 zufällige Papers pro Trial
    
*   **Ressourcen‑Schonung:** Temporäre Collections werden nach jedem Trial gelöscht
    
*   **Metriken:** MRR, Hit Rate@1/3/5/10, F1‑Score, RAGAS (Faithfulness, Answer Relevancy, Context Precision)
    

🤖 KI‑Nutzung
-------------

Folgende KI‑Tools unterstützten die Entwicklung:

ToolVerwendung**ChatGPT (GPT‑4o)**Code‑Generierung, Fehleranalyse, Dokumentation**GitHub Copilot**Code‑Vervollständigung**DeepL**Übersetzung**LanguageTool**Grammatikprüfung

📄 Lizenz
---------

Dieses Projekt ist ausschließlich für akademische Zwecke bestimmt.

👩‍💻 Autorin
-------------

**Sakina Ahmadi**Leibniz Universität HannoverInstitut für Künstliche Intelligenz (LUHAI)E‑Mail: sakina.ahmadi@stud.uni-hannover.de

📚 Referenzen
-------------

1.  Lewis, P., et al. (2020). _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_. NeurIPS 2020.
    
2.  Zha, D., et al. (2023). _AutoML: From Basics to Recent Advances_. ACM Computing Surveys.
    
3.  Reimers, N., & Gurevych, I. (2019). _Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks_. EMNLP 2019.
    
4.  Xiao, S., et al. (2024). _BGE‑M3: A Multi‑Functional Embedding Model_. arXiv:2402.00085.
    
5.  Akiba, T., et al. (2019). _Optuna: A Next‑generation Hyperparameter Optimization Framework_. KDD 2019.
    
6.  Dhuliawala, S., et al. (2023). _RAGAS: Automated Evaluation Framework for Retrieval‑Augmented Generation_. arXiv:2309.15217.
    
7.  Dettmers, T., et al. (2022). _LLM.int8(): 8‑bit Matrix Multiplication for Transformers at Scale_. arXiv:2208.07339.
    
8.  Malkov, Y. A., & Yashunin, D. A. (2018). _Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs_. TPAMI 2018.
    
9.  Karpukhin, V., et al. (2020). _Dense Passage Retrieval for Open‑Domain Question Answering_. EMNLP 2020.
    
10.  Gao, L., et al. (2023). _Chunking Strategies for Retrieval‑Augmented Generation_. arXiv:2309.00071.
    

🙏 Danksagung
-------------

Ich danke **Prof. Dr. Marius Lindauer** für die Betreuung dieser Arbeit sowie dem **Institut für Künstliche Intelligenz (LUHAI)** der Leibniz Universität Hannover für die Bereitstellung der notwendigen Infrastruktur.
