import json
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Setup Model & Client
model = SentenceTransformer('BAAI/bge-m3') # From your Hyperparameter Grid
client = qdrant_client.QdrantClient(path="C:/Users/ahmad/Desktop/rag_ml/qdrant_db")

# 2. Create Collection
collection_name = "arxiv_stage_1_abstracts"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE) # BGE-M3 size
)

# 3. Index Abstracts
input_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\modern_golden_set.jsonl"
batch_size = 32

with open(input_file, 'r', encoding='utf-8') as f:
    batch_papers = []
    for i, line in enumerate(f):
        batch_papers.append(json.loads(line))
        
        if len(batch_papers) == batch_size:
            texts = [p['abstract_clean'] for p in batch_papers]
            embeddings = model.encode(texts)
            
            points = [
                PointStruct(
                    id=i - batch_size + j, 
                    vector=embeddings[j].tolist(), 
                    payload=batch_papers[j]
                ) for j in range(batch_size)
            ]
            client.upsert(collection_name=collection_name, points=points)
            batch_papers = []
            if i % 320 == 0: print(f"Indexed {i} abstracts...")

print("Stage 1 Indexing Complete!")