# modules/qdrant_storage.py
import os
import uuid
from kaggle_secrets import UserSecretsClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, PointStruct

class QdrantManager:
    def __init__(self):
        user_secrets = UserSecretsClient()
        qdrant_url = user_secrets.get_secret("QDRANT_URL_A")
        qdrant_api_key = user_secrets.get_secret("QDRANT_API_KEY_A")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("❌ Kritischer Fehler: QDRANT_URL oder QDRANT_API_KEY fehlt in den Kaggle Secrets!")
            
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

    def create_trial_collection(self, collection_name: str, vector_size: int = 1024):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            )
        )

    def upload_chunks(self, collection_name: str, chunks: list, vectors: list):
        if len(chunks) != len(vectors):
            raise ValueError("❌ Anzahl Chunks ≠ Anzahl Vektoren!")

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {
                "text_llm": chunk if isinstance(chunk, str) else chunk.get("text", ""),
                "metadata": {
                    "source": "unknown" if isinstance(chunk, str) else chunk.get("metadata", {}).get("source", "unknown"),
                    "chunk_id": idx
                }
            }

            point_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Batch-Upsert (500 Punkte pro Request)
        for i in range(0, len(points), 500):
            batch = points[i:i+500]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )

    def search_retriever(self, collection_name: str, query_vector: list, top_k: int = 3) -> list:
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return [
            r.payload["text_llm"]
            for r in results
            if r.payload and "text_llm" in r.payload
        ]

    def delete_trial_collection(self, collection_name: str):
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
