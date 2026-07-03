# modules/metrics.py
import numpy as np
from modules.embedder import embed_texts

def optimize_similarity_threshold(retrieved_contexts, gold_contexts, model_key):
    """
    Optimiert den Cosine-Similarity-Threshold für ein gegebenes Modell,
    ohne die Embeddings redundant neu zu berechnen.
    
    Erweitert auf 0.30 bis 0.85, um gestauchte Vektorräume (bge-m3) aufzufangen.
    """
    # 1. Vorab-Berechnung aller maximalen Ähnlichkeiten pro Frage (Sehr schnell!)
    max_similarities = []
    
    for retrieved, gold in zip(retrieved_contexts, gold_contexts):
        if not retrieved or not gold:
            max_similarities.append(0.0)
            continue
            
        # Einbetten der Texte (nur 1x pro Trial-Aufruf)
        gold_vec = np.array(embed_texts(gold, model_key=model_key, is_query=True)[0])
        chunk_vecs = np.array(embed_texts(retrieved, model_key=model_key, is_query=False))
        
        # Matrix-Multiplikation für Cosine Similarity
        norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(gold_vec) + 1e-8
        sims = (chunk_vecs @ gold_vec) / norms
        
        max_similarities = np.array(max_similarities) if isinstance(max_similarities, np.ndarray) else np.array([])
        max_similarities = np.append(max_similarities, np.max(sims))

    # 2. Sweep über erweiterten Schwellenwert-Bereich
    if model_key == "bge-m3":
        thresholds = np.linspace(0.45, 0.70, 15)
        best_threshold = 0.45
    else:
        thresholds = np.linspace(0.55, 0.85, 15)
        best_threshold = 0.55
    
    best_f1 = 0.0
    
    for t in thresholds:
        # Berechne True Positives (getroffen) und False Negatives (verfehlt)
        tp = np.sum(max_similarities >= t)
        fn = np.sum(max_similarities < t)
        
        # Da y_true in unserem Kontext immer 1 ist (die Dokumente SOLLTEN gefunden werden),
        # entspricht Precision hier immer 1.0 für die getroffenen Instanzen.
        # Der klassische F1-Score vereinfacht sich hier zu:
        if tp == 0:
            f1 = 0.0
        else:
            precision = 1.0
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    return float(best_threshold), float(best_f1)