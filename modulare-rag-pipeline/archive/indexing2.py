import os
import json
import hashlib
import re
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. CONFIG & PATHS ---
base_path = r"C:\Users\ahmad\Desktop\rag_ml\data\papers"
md_dir = os.path.join(base_path, "extracted_markdown")
db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
checkpoint_file = os.path.join(base_path, "stage2_checkpoint.txt")

# AutoML Hyperparameters (from your grid)
CHUNK_SIZE = 1024 
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = 'BAAI/bge-m3'

# --- 2. THE CLEANING FUNCTIONS ---
def get_heuristic_caption(latex_str):
    translations = {
        r'\int': ' integral ', r'\sum': ' summation ', r'\nabla': ' gradient ',
        r'\partial': ' partial derivative ', r'\alpha': ' alpha ', r'\beta': ' beta ',
        r'\approx': ' approximately equal ', r'\infty': ' infinity '
    }
    caption = latex_str.strip()
    for cmd, word in translations.items():
        caption = caption.replace(cmd, word)
    caption = re.sub(r'[\$\{\}\^_\\]', ' ', caption)
    return f"[MATH: {re.sub(r'\s+', ' ', caption).strip()}]"

def clean_for_embedding(text):
    # Remove references and URLs
    text = re.split(r'\n#+ (?:References|Bibliography)', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'http\S+', '', text)
    # Apply Math-to-Text Heuristic
    text = re.sub(r'\$\$?.*?\$\$?', lambda m: get_heuristic_caption(m.group(0)), text)
    return re.sub(r'\s+', ' ', text).strip()

# --- 3. SETUP MODELS & DB ---
print("Loading Embedding Model (CPU)...")
model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
client = qdrant_client.QdrantClient(path=db_path)
collection_name = "stage2_chunks_modern"

if not any(c.name == collection_name for c in client.get_collections().collections):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# --- 4. PROCESSING LOOP ---
processed_files = []
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        processed_files = f.read().splitlines()

all_md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]

for idx, filename in enumerate(all_md_files):
    if filename in processed_files: continue
    
    paper_id = filename.replace(".md", "").replace("_", "/")
    file_path = os.path.join(md_dir, filename)
    
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # 4.1 Structural Chunking
    sections = re.split(r'(?=#+ )', full_text)
    chunks_to_index = []
    
    for section in sections:
        if len(section) < 100: continue
        # Split large sections into chunks
        start = 0
        while start < len(section):
            end = start + CHUNK_SIZE
            chunk_raw = section[start:end]
            # Create Dual-Path: Cleaned for Embedding, Raw for LLM
            chunks_to_index.append({
                "raw": chunk_raw,
                "clean": clean_for_embedding(chunk_raw)
            })
            start += CHUNK_SIZE - CHUNK_OVERLAP

    # 4.2 Embedding & Upserting
    if chunks_to_index:
        embeddings = model.encode([c["clean"] for c in chunks_to_index])
        points = []
        for i, chunk in enumerate(chunks_to_index):
            points.append(PointStruct(
                id=hashlib.md5(f"{filename}_{i}".encode()).hexdigest(),
                vector=embeddings[i].tolist(),
                payload={
                    "paper_id": paper_id,
                    "text_llm": chunk["raw"],      # Original LaTeX
                    "text_embed": chunk["clean"],  # Processed text
                }
            ))
        client.upsert(collection_name=collection_name, points=points)

    # 4.3 Update Checkpoint
    with open(checkpoint_file, "a") as cf:
        cf.write(filename + "\n")
    
    if idx % 10 == 0:
        print(f"Processed {idx}/{len(all_md_files)} papers...")

print("🚀 Stage 2 Indexing Complete!")
