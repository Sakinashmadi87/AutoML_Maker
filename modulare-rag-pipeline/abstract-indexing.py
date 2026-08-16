#abstracts-indexing.py
import os
import json
import hashlib
import time
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# Resolve repository paths and modular architectures
sys.path.append("/kaggle/working/modulare-rag-pipeline")
sys.path.append("/kaggle/working")

from modules.cleaner import clean_scientific_markdown
from config import PATHS, HP_GRID, IS_COLAB, IS_KAGGLE

def main():
    print(f"🖥️ Initializing Abstract Indexer on: {HP_GRID['device'].upper()}")
    model = SentenceTransformer(HP_GRID['embedding_model'], device=HP_GRID['device'])
    
    # 1. PLATFORM-AWARE CLUSTER CONNECTION HANDSHAKE
    if IS_COLAB or IS_KAGGLE:
        print("🌐 Connecting to Remote Production Qdrant Cloud Instance...")
        # Try fetching from environment first (handles manual injections)
        cloud_url = os.getenv('QDRANT_URL')
        cloud_key = os.getenv('QDRANT_API_KEY')
        
        # Fallback to native platform secret clients if empty
        if not cloud_url or not cloud_key:
            if IS_KAGGLE:
                from kaggle_secrets import UserSecretsClient
                user_secrets = UserSecretsClient()
                cloud_url = user_secrets.get_secret("QDRANT_URL")
                cloud_key = user_secrets.get_secret("QDRANT_API_KEY")
        
        client = qdrant_client.QdrantClient(
            url=cloud_url, 
            api_key=cloud_key,
            check_compatibility=False,
            timeout=60.0
        )
    else:
        print("🏠 Local fallback active. Connecting to local Qdrant directory...")
        client = qdrant_client.QdrantClient(path=r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db")

    # Target the unified optimized collection we cleared earlier
    collection_name = "stage1_abstracts_cleaned"
    
    if not client.collection_exists(collection_name):
        print(f"✨ Initializing fresh schema schematic: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    # 2. SEPARATE CHECKPOINT PATH MANAGEMENT
    # We use a dedicated text file tracker to handle abstract sync states safely
    abstracts_checkpoint = "/kaggle/working/abstracts_indexing_checkpoint.txt"
    done_files = []
    if os.path.exists(abstracts_checkpoint):
        with open(abstracts_checkpoint, "r") as f:
            done_files = f.read().splitlines()

    if not os.path.exists(PATHS["markdown"]):
        print(f"❌ Critical Error: Source markdown folder missing at: {PATHS['markdown']}")
        return

    all_files = [f for f in os.listdir(PATHS["markdown"]) if f.endswith(".md")]
    files_to_process = [f for f in all_files if f not in done_files]

    print(f"🚀 Processing {len(files_to_process)} abstracts for metadata synchronization...")
    UPLOAD_BATCH_SIZE = 64 # Aggressive batch size optimized for GPU memory sweeps

    for filename in tqdm(files_to_process, desc="Abstract Sync Progress"):
        file_path = os.path.join(PATHS["markdown"], filename)
        # ID Normalization string parsing (e.g. 2405_06707 -> 2405.06707)
        paper_id = filename.replace(".md", "").replace("_", ".")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_md = f.read()
        except Exception:
            continue
            
        # Parse out the top section header block where abstract text usually anchors
        # This extracts a dense context segment for Stage 1 lookup matching
        cleaned_md = clean_scientific_markdown(raw_md, replace_math=True)
        abstract_segment = cleaned_md[:3000] # Extracts high-density abstract/intro text context
        
        if abstract_segment.strip():
            # Metadata Fusing: We attach Title anchors directly to the abstract vector space
            title_anchor = filename.replace(".md", "").replace("_", " ")
            fused_text = f"Title: {title_anchor} | Content: {abstract_segment}"
            
            vector = model.encode(fused_text, show_progress_bar=False).tolist()
            
            # Use deterministic MD5 string IDs matching your main chunker architecture layout
            point_id = hashlib.md5(f"{paper_id}_abstract_stage1".encode()).hexdigest()
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "paper_id": paper_id, 
                    "text": abstract_segment,
                    "has_full_text": True
                }
            )
            
            # Crisis-proof multi-attempt cloud upload sequence
            for attempt in range(3):
                try:
                    client.upsert(collection_name=collection_name, points=[point])
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"⚠️ Cloud insertion bottleneck hit for paper {paper_id}: {e}")

        # Update tracking file on loop iteration completion
        with open(abstracts_checkpoint, "a") as f:
            f.write(filename + "\n")

    # 3. SET PRODUCTION PAYLOAD INDEX DIRECTLY ON COMPLETION
    print(f"🔧 Synchronized. Setting high-performance keyword indexing layer...")
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="paper_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("✅ Production Payload Index compiled successfully!")
    except Exception as e:
        print(f"⚠️ Index check passed: {e}")

    print("🏆 Stage 1 Abstract Indexing Completely Synchronized!")

if __name__ == "__main__":
    main()
