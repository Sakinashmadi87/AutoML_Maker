import qdrant_client
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchAny
import ollama

# 1. Setup Models & DB
print("Initializing Search Engine...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
client = qdrant_client.QdrantClient(path="C:/Users/ahmad/Desktop/rag_ml/qdrant_db")

# Your Query
query = "How does Second Thoughts enable language models to re-align with human values?"

# 2. STAGE 1: Find the relevant papers
query_vector = model.encode(query).tolist()
stage1_results = client.query_points(
    collection_name="arxiv_stage_1_abstracts",
    query=query_vector,
    limit=3
).points

target_ids = [res.payload['id'] for res in stage1_results]

# 3. STAGE 2: Extract specific chunks
stage2_results = client.query_points(
    collection_name="arxiv_stage_2_body_chunks",
    query=query_vector,
    query_filter=Filter(
        must=[FieldCondition(key="paper_id", match=MatchAny(any=target_ids))]
    ),
    limit=3
).points

# 4. STAGE 4: Generation
print("Synthesizing answer using Llama-3.1...")
context_text = "\n\n".join([res.payload['content'] for res in stage2_results])

prompt = f"""
You are a Research Assistant. Use the provided paper chunks to answer the question.
If the answer isn't in the context, say you don't know. 
Be precise and cite technical details from the text.

QUESTION: {query}

CONTEXT:
{context_text}

SCIENTIFIC ANSWER:
"""

response = ollama.generate(model='llama3.1', prompt=prompt)

print("\n" + "="*50)
print("FINAL SCIENTIFIC ANSWER:")
print("="*50)
print(response['response'])
