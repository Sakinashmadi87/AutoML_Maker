from modules.chunker import ArxivChunker

# ... inside your loop over MD files ...
chunker = ArxivChunker(chunk_size=512, chunk_overlap=50)
paper_chunks = chunker.chunk_and_fuse(md_content, paper_id, meta_lookup)

# Immediately embed and upload to Qdrant (no need for intermediate jsonl)
# ... code to embed paper_chunks and upsert to Qdrant ...
