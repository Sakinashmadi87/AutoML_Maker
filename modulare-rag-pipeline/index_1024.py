#index_1024.py
import os
import json
import hashlib
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# Modulare Importe
from modules.cleaner import clean_scientific_markdown
from modules.chunker import ArxivChunker
from config import PATHS, HP_GRID, IS_COLAB, IS_KAGGLE

def main():
    print(f"🖥️ Initialisiere Indexer auf: {HP_GRID['device'].upper()}")
    model = SentenceTransformer(HP_GRID['embedding_model'], device=HP_GRID['device'])
    
    # Client-Verbindung (Cloud oder Lokal)
    if IS_COLAB or IS_KAGGLE:
        print("🌐 Verbinde mit Qdrant CLOUD Cluster...")
        client = qdrant_client.QdrantClient(
            url=os.getenv('QDRANT_URL'), 
            api_key=os.getenv('QDRANT_API_KEY'),
            check_compatibility=False,
            timeout=60.0
        )
    else:
        print("🏠 Verbinde mit lokaler Qdrant-Datenbank...")
        client = qdrant_client.QdrantClient(path=r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024")

    collection_name = "stage2_chunks_1024"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    chunker = ArxivChunker(chunk_size=1024, chunk_overlap=100)

    # Resume-Logik über die sauberen Config-Pfade
    done_files = []
    if os.path.exists(PATHS["checkpoint_1024"]):
        with open(PATHS["checkpoint_1024"], "r") as f:
            done_files = f.read().splitlines()

    if not os.path.exists(PATHS["markdown"]):
        print(f"❌ Fehler: Ordner nicht gefunden unter {PATHS['markdown']}")
        return

    all_files = [f for f in os.listdir(PATHS["markdown"]) if f.endswith(".md")]
    files_to_process = [f for f in all_files if f not in done_files]

    print(f"🚀 Verarbeite {len(files_to_process)} Dokumente...")
    UPLOAD_BATCH_SIZE = 32

    for filename in tqdm(files_to_process, desc="Total Progress"):
        file_path = os.path.join(PATHS["markdown"], filename)
        paper_id = filename.replace(".md", "").replace("_", "/")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_md = f.read()
        
        cleaned_md = clean_scientific_markdown(raw_md, replace_math=True)
        chunks = chunker.splitter.split_text(cleaned_md)
        
        if chunks:
            embeddings = model.encode(chunks, show_progress_bar=False)
            all_points = []
            for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                chunk_id = hashlib.md5(f"{paper_id}_{i}_1024".encode()).hexdigest()
                all_points.append(PointStruct(
                    id=chunk_id,
                    vector=vector.tolist(),
                    payload={"paper_id": paper_id, "text_llm": chunk_text, "chunk_size": 1024}
                ))
            
            # Krisensicherer Upload
            for k in range(0, len(all_points), UPLOAD_BATCH_SIZE):
                chunk_slice = all_points[k:k + UPLOAD_BATCH_SIZE]
                for attempt in range(3):
                    try:
                        client.upsert(collection_name=collection_name, points=chunk_slice)
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(5)
                        else:
                            raise e

        with open(PATHS["checkpoint_1024"], "a") as f:
            f.write(filename + "\n")

    print("🏆 Indexing Complete!")

if __name__ == "__main__":
    main()
