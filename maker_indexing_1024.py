import os
import uuid
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from kaggle_secrets import UserSecretsClient

# Lokale Module deiner Pipeline laden
from modules.chunker import chunk_hierarchical
from modules.embedder import embed_texts

# ---------------------------------------------------------------------
# 1. KONFIGURATION FÜR DIE PRODUKTIV-DATENBANK
# ---------------------------------------------------------------------
MARKER_DIR = Path("/kaggle/input/datasets/sakinaahmadi87/marker-parsed-papers-2765/marker_parsed_papers")
PRODUCTION_COLLECTION = "maker_hierarchical_1024"
EMBED_MODEL = "mxbai-large"

# Ingestion-Sicherheitsfeatures
BATCH_SIZE_PAPERS = 20  # Verarbeitet 20 Papers am Stück, bevor Vektoren erzeugt werden
UPSERT_BATCH_SIZE = 64  # Lädt Vektoren in stabilen 64er-Häppchen zu Qdrant hoch

# ---------------------------------------------------------------------
# 2. CLIENT-INITIALISIERUNG
# ---------------------------------------------------------------------
user_secrets = UserSecretsClient()
url = user_secrets.get_secret("QDRANT_URL_A")
api_key = user_secrets.get_secret("QDRANT_API_KEY_A")
client = QdrantClient(url=url, api_key=api_key)

# ---------------------------------------------------------------------
# 3. PRODUKTIONS-INDEX INITIALISIEREN
# ---------------------------------------------------------------------
print("🚀 Erstelle finale Produktions-Collection...")
# Ein kurzes Test-Embedding, um die exakte Dimension dynamisch zu bestimmen
sample_dim = len(embed_texts(["Test"], model_key=EMBED_MODEL, is_query=False)[0])

client.create_collection(
        collection_name=PRODUCTION_COLLECTION,
        vectors_config=VectorParams(size=sample_dim, distance=Distance.COSINE)
    )
print(f"✅ Collection '{PRODUCTION_COLLECTION}' (Dimension: {sample_dim}) bereit.")

# ---------------------------------------------------------------------
# 4. STREAMING-INGESTION DER 2765 PAPERS
# ---------------------------------------------------------------------
all_md_files = list(MARKER_DIR.glob("**/*.md"))
total_files = len(all_md_files)
print(f"📂 Insgesamt {total_files} Markdown-Dateien zur Indexierung gefunden.")

current_chunks = []
processed_count = 0

print("\n📥 Starte schrittweise Ingestion (Verhindert RAM-Overflow)...")

for idx, file_path in enumerate(all_md_files, start=1):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Hierarchisches Chunking (1024 Token, 10% Overlap)
        chunks = chunk_hierarchical(text, chunk_size=1024, overlap=102)
        
        # Validierung der Chunks samt Metadaten-Payload
        for chunk in chunks:
            if isinstance(chunk, str) and len(chunk.strip()) > 20:
                current_chunks.append({
                    "text": chunk,
                    "source": file_path.name
                })
                
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen von {file_path.name}: {e}")
        continue

    # Wenn der Paper-Batch voll ist ODER wir am Ende der Liste sind -> Verarbeiten!
    if idx % BATCH_SIZE_PAPERS == 0 or idx == total_files:
        if current_chunks:
            # 1. Texte extrahieren für Embedder
            texts_to_embed = [item["text"] for item in current_chunks]
            
            # 2. Embeddings für den aktuellen Batch generieren
            embeddings = embed_texts(texts_to_embed, model_key=EMBED_MODEL, is_query=False)
            
            # 3. PointStructs für Qdrant aufbauen
            points = []
            for i, item in enumerate(current_chunks):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embeddings[i].tolist(),
                        payload={
                            "text": item["text"],
                            "source_document": item["source"]
                        }
                    )
                )
            
            # 4. In stabilen Unter-Batches hochladen
            for k in range(0, len(points), UPSERT_BATCH_SIZE):
                client.upsert(
                    collection_name=PRODUCTION_COLLECTION, 
                    points=points[k:k+UPSERT_BATCH_SIZE]
                )
            
            processed_count += (idx % BATCH_SIZE_PAPERS if idx % BATCH_SIZE_PAPERS != 0 else BATCH_SIZE_PAPERS)
            print(f"💾 Fortschritt: {idx}/{total_files} Papers indiziert. (+{len(points)} Chunks hochgeladen)")
            
            # Arbeitsspeicher für den nächsten Batch leeren
            current_chunks.clear()

print("\n" + "🏆" * 30)
print("🏆 PRODUKTIV-INDEXIERUNG ERFOLGREICH ABGESCHLOSSEN!")
print(f"🏆 Alle auffindbaren Papers wurden in '{PRODUCTION_COLLECTION}' gespeichert.")
print("🏆" * 30)