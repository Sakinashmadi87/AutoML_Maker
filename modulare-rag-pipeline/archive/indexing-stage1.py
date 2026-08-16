import json
import os
import time
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- PFADE ---
base_path = r"C:\Users\ahmad\Desktop\rag_ml\data\papers"
input_file = os.path.join(base_path, "modern_golden_set.jsonl")
checkpoint_file = os.path.join(base_path, "indexing_checkpoint_only_pdfs.txt")
pdf_active_dir = os.path.join(base_path, "pdfs_active")
db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"

# --- SETUP ---
print("Lade BGE-M3 Modell (CPU)...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')

client = qdrant_client.QdrantClient(path=db_path)
collection_name = "stage1_abstracts_only_active_pdfs"

# Scanne verfügbare PDFs
print("Scanne verfügbare PDFs in 'pdfs_active'...")
available_pdfs = {f.replace(".pdf", "") for f in os.listdir(pdf_active_dir) if f.endswith(".pdf")}
print(f"Gefunden: {len(available_pdfs)} aktive PDFs. Nur diese werden indexiert.")

# Collection erstellen falls nicht vorhanden
collections = client.get_collections().collections
if not any(c.name == collection_name for c in collections):
    print(f"Erstelle neue Collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

# Checkpoint laden
start_line = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        start_line = int(f.read().strip())
    print(f"Fortsetzen ab Zeile {start_line}...")

# --- INDEXIERUNG ---
batch_size = 16
batch_papers = []
indexed_count = 0

print(f"Starte Indexierung (Filter: NUR Paper mit vorhandenem PDF)...")

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        # Überspringe bis zum Checkpoint
        if i < start_line:
            continue
            
        paper = json.loads(line)
        safe_id = paper['id'].replace('/', '_')
        
        # FILTER: Nur wenn PDF existiert, in den Batch aufnehmen
        if safe_id in available_pdfs:
            batch_papers.append(paper)
        
        # Wenn Batch voll ist oder Dateiende erreicht
        if len(batch_papers) == batch_size:
            # Metadata Fusing
            texts_to_embed = []
            for p in batch_papers:
                kws = ", ".join(p.get('keywords', []))
                combined = f"Title: {p['title_clean']} | Keywords: {kws} | Abstract: {p['abstract_clean']}"
                texts_to_embed.append(combined)
            
            # Embeddings berechnen
            embeddings = model.encode(texts_to_embed, show_progress_bar=False)
            
            points = []
            for j in range(len(batch_papers)):
                p_data = batch_papers[j]
                s_id = p_data['id'].replace('/', '_')
                
                points.append(PointStruct(
                    id=indexed_count + j + 1, # Fortlaufende ID für Qdrant
                    vector=embeddings[j].tolist(),
                    payload={
                        "paper_id": p_data['id'],
                        "file_id": s_id,
                        "title": p_data['title_clean'],
                        "has_full_text": True  # In dieser Collection haben alle PDFs
                    }
                ))
            
            # Upload zu Qdrant
            client.upsert(collection_name=collection_name, points=points)
            indexed_count += len(batch_papers)
            
            # Checkpoint speichern (wir speichern die Zeilennummer der Quelldatei)
            with open(checkpoint_file, "w") as cf:
                cf.write(str(i + 1))
            
            print(f"Zeile {i+1} verarbeitet. Indexierte Paper bisher: {indexed_count}")
            batch_papers = []

# Letzten Batch verarbeiten (falls vorhanden)
if batch_papers:
    # ... (gleiche Logik wie oben für den Rest) ...
    texts_to_embed = [f"Title: {p['title_clean']} | Keywords: {', '.join(p.get('keywords', []))} | Abstract: {p['abstract_clean']}" for p in batch_papers]
    embeddings = model.encode(texts_to_embed, show_progress_bar=False)
    points = [PointStruct(id=indexed_count + j + 1, vector=embeddings[j].tolist(), payload={"paper_id": batch_papers[j]['id'], "file_id": batch_papers[j]['id'].replace('/', '_'), "title": batch_papers[j]['title_clean'], "has_full_text": True}) for j in range(len(batch_papers))]
    client.upsert(collection_name=collection_name, points=points)
    indexed_count += len(batch_papers)

print(f"🚀 Fertig! Insgesamt {indexed_count} Paper mit PDFs indexiert.")
