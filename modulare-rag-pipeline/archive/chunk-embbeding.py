import json
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Load Model (CPU-safe)
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
client = qdrant_client.QdrantClient(path="C:/Users/ahmad/Desktop/rag_ml/qdrant_db")

# 2. Create Stage 2 Collection
# We use 1024 dims for BGE-M3
collection_name = "arxiv_stage_2_body_chunks"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# 3. Index the Fused Chunks
chunk_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\chunks.jsonl"
batch_size = 16

with open(chunk_file, 'r', encoding='utf-8') as f:
    batch = []
    for i, line in enumerate(f):
        batch.append(json.loads(line))
        
        if len(batch) == batch_size:
            # We embed the FUSED text
            embeddings = model.encode([item['text'] for item in batch])
            
            points = [
                PointStruct(
                    id=abs(hash(item['paper_id'] + str(item['chunk_index']))) % (10**10), # Unique ID
                    vector=embeddings[j].tolist(),
                    payload={
                        "paper_id": item['paper_id'],
                        "title": item['metadata']['title'],
                        "content": item['text']
                    }
                ) for j, item in enumerate(batch)
            ]
            client.upsert(collection_name=collection_name, points=points)
            batch = []
            if i % 80 == 0: print(f"Indexed {i} chunks...")

print("Stage 2 Indexing Complete!")
