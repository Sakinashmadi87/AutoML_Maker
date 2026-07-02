from kaggle_secrets import UserSecretsClient
from qdrant_client import QdrantClient

# Mapping: Collection → (URL, API_KEY)
QDRANT_CONFIG = {
    "marker_hierarchical_1024": (
        "QDRANT_URL_M",
        "QDRANT_API_KEY_M"
    ),
    "stage2_docling_hybrid_bge_int8": (
        "QDRANT_URL",
        "QDRANT_API_KEY"
    ),
    "stage2_docling_hybrid_bge_f32": (
        "QDRANT_URL_F",
        "QDRANT_API_KEY_F"
    ),
    "pymupdf4llm_1024": (
        "QDRANT_NEW_URL",
        "QDRANT_NEW_API_KEY"
    ),
    "docling_hierarchical_512": (
        "QDRANT_URL_D",
        "QDRANT_API_KEY_D"
    )
}

class MultiQdrantManager:
    def __init__(self):
        self.user_secrets = UserSecretsClient()
        self.clients = {}

    def get_client(self, collection_name: str):
        if collection_name not in QDRANT_CONFIG:
            raise ValueError(f"❌ Keine Qdrant-Konfiguration für Collection: {collection_name}")

        url_secret, key_secret = QDRANT_CONFIG[collection_name]
        url = self.user_secrets.get_secret(url_secret)
        key = self.user_secrets.get_secret(key_secret)

        if not url or not key:
            raise ValueError(f"❌ Secrets fehlen für {collection_name}: {url_secret}, {key_secret}")

        # Cache Client
        if collection_name not in self.clients:
            self.clients[collection_name] = QdrantClient(url=url, api_key=key)

        return self.clients[collection_name]

    def search(self, collection_name: str, query_vector: list, top_k: int = 10):
        client = self.get_client(collection_name)
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points

        contexts = []
        for r in results:
            payload = r.payload
            text = payload.get("text_llm") or payload.get("text") or payload.get("content")
            if text:
                contexts.append(text)

        return contexts
