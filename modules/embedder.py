# modules/embedder.py
import torch
from sentence_transformers import SentenceTransformer

# Cache, damit Modelle nur 1x geladen werden
_EMBED_CACHE = {}

EMBEDDING_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "mxbai-large": "mixedbread-ai/mxbai-embed-large-v1"
}

def load_embedder(model_key: str):
    """Lädt das Embedding-Modell nur einmal (Caching) und nutzt GPU, falls verfügbar."""
    if model_key not in _EMBED_CACHE:
        print(f"🔧 Lade Embedding-Modell: {model_key}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SentenceTransformer(
            EMBEDDING_MODELS[model_key],
            device=device
        )

        _EMBED_CACHE[model_key] = model
        
    return _EMBED_CACHE[model_key]


def embed_texts(texts, model_key: str, is_query: bool = False, batch_size: int = 32):
    """
    Einheitliche Embedding-Funktion für AutoML.
    - texts: Liste von Strings oder ein einzelner String
    - model_key: 'bge-m3', 'mxbai-large'
    - is_query: True für Query-Embedding, False für Chunk-Embedding
    """
    if isinstance(texts, str):
        texts = [texts]
        
    model = load_embedder(model_key)

    texts_to_encode = texts

    vectors = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vectors.tolist()
