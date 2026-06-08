from sentence_transformers import SentenceTransformer
import numpy as np

# Cache, damit Modelle nur 1x geladen werden
_EMBED_CACHE = {}

EMBEDDING_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "mxbai-large": "mixedbread-ai/mxbai-embed-large-v1"
}

def load_embedder(model_key: str):
    """Lädt das Embedding-Modell nur einmal (Caching) mit GPU-Sicherung."""
    if model_key not in _EMBED_CACHE:
        print(f"🔧 Lade Embedding-Modell: {model_key}")
        
        # Sicherstellen, dass trust_remote_code für Qwen aktiv ist
        model = SentenceTransformer(
            EMBEDDING_MODELS[model_key],
            trust_remote_code=True,
            device="cuda"  # Explizit auf die Kaggle-GPU schieben
        )
        _EMBED_CACHE[model_key] = model
        
    return _EMBED_CACHE[model_key]


def embed_texts(texts, model_key: str, is_query: bool = False, batch_size: int = 32):
    """
    Einheitliche Embedding-Funktion für AutoML.
    - texts: Liste von Strings oder ein einzelner String
    - model_key: 'bge-m3', 'gte-qwen2', 'mxbai-large'
    - is_query: True, wenn eine Frage aus eval_set eingebettet wird; False für Chunks
    """
    # Falls versehentlich ein einzelner Text übergeben wird, in Liste wandeln
    if isinstance(texts, str):
        texts = [texts]
        
    model = load_embedder(model_key)
    
    # -----------------------------------------------------------------
    # SPEZIALFALL: Prompt-Handling für gte-Qwen2-instruct
    # -----------------------------------------------------------------
    texts_to_encode = texts
    if model_key == "gte-qwen2":
        if is_query:
            # Suchanfragen brauchen diese explizite Anweisung bei Qwen
            texts_to_encode = [
                f"Given a query, retrieve relevant passages that answer the query.\nQuery: {t}" 
                for t in texts
            ]
        else:
            # Dokumenten-Chunks werden ohne Zusatz-Prompt strukturiert
            texts_to_encode = texts

    # Batch-Embedding ausführen
    vectors = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        normalize_embeddings=True,  # Wichtig für Cosine-Ähnlichkeit
        show_progress_bar=False
    )

    # -----------------------------------------------------------------
    # SPEZIALFALL: Matryoshka-Slicing für gte-Qwen2 (1536 -> 1024 Dim)
    # -----------------------------------------------------------------
    if model_key == "gte-qwen2":
        # Wir schneiden die Vektoren auf die ersten 1024 Dimensionen ab
        vectors = vectors[:, :1024]
        
        # WICHTIG: Nach dem Abschneiden müssen wir die Vektoren erneut normalisieren!
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)

    # In Python-Listen umwandeln für Qdrant-Kompatibilität
    return vectors.tolist()