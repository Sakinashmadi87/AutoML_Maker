import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# 1. Paths
md_folder = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\markdown"
chunk_output = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\chunks.jsonl"
metadata_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\modern_golden_set.jsonl"

# 2. Load Metadata for Fusing (using ID as key)
print("Loading metadata for fusing...")
meta_lookup = {}
with open(metadata_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        meta_lookup[item['id']] = {
            "title": item.get('title_clean', 'Unknown Title'),
            "abstract": item.get('abstract_clean', 'No abstract available')
        }

# 3. Setup Chunking Strategy (Recursive-Character)
# Hyperparameters from your grid: 512 size, 10% overlap
chunk_size = 512
chunk_overlap = 51

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
)

# 4. Process and Fuse
all_chunks = []
md_files = [f for f in os.listdir(md_folder) if f.endswith('.md')]

for md_file in md_files:
    paper_id = md_file.replace('.md', '').replace('_', '/') # Restore original ID
    paper_meta = meta_lookup.get(paper_id, {"title": "Unknown", "abstract": ""})
    
    with open(os.path.join(md_folder, md_file), 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Generate chunks
    raw_chunks = splitter.split_text(content)
    
    for i, chunk in enumerate(raw_chunks):
        # Metadata Fusing (Title + Abstract Prefix)
        fused_text = f"TITLE: {paper_meta['title']}\nABSTRACT: {paper_meta['abstract'][:200]}...\n\nCHUNK CONTENT:\n{chunk}"
        
        all_chunks.append({
            "paper_id": paper_id,
            "chunk_index": i,
            "text": fused_text,
            "metadata": {
                "title": paper_meta['title'],
                "chunk_size": chunk_size,
                "overlap": chunk_overlap
            }
        })

# 5. Save chunks
with open(chunk_output, 'w', encoding='utf-8') as out:
    for c in all_chunks:
        out.write(json.dumps(c) + '\n')

print(f"Success! Created {len(all_chunks)} fused chunks from 5 papers.")
