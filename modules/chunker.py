#modules/embedder.py
from langchain_experimental.text_splitter import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1. MARKDOWN SECTION SPLITTER
# -----------------------------
def chunk_markdown_section(text, chunk_size, overlap):
    # Schritt A: Nach strukturellen Markdown-Headern trennen
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    sections = header_splitter.split_text(text)
    
    # Schritt B: Jede Section auf die vom AutoML geforderte Chunk-Größe bringen
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    final_chunks = []
    for section in sections:
        # Falls du Header-Metadaten behalten willst, kannst du sie hier aus section.metadata ziehen
        chunks = sub_splitter.split_text(section.page_content)
        final_chunks.extend(chunks)
        
    return final_chunks


# -----------------------------
# 2. SEMANTIC CHUNKING (Optimiert)
# -----------------------------
# Tipp: Initialisiere das Modell EINMALIG in deiner main.py und übergib es hier,
# um extremen Overhead beim AutoML-Lauf zu vermeiden.
def chunk_semantic(text, chunk_size, overlap, semantic_model=None):
    if semantic_model is None:
        from sentence_transformers import SentenceTransformer
        # Nutze ein Modell, das flexibel ist, oder behalte das kleine nur fürs Splitten bei
        semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []
        
    embeddings = semantic_model.encode(sentences, show_progress_bar=False)

    import numpy as np
    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        # Kosinus-Ähnlichkeit berechnen
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
        )
        
        # Wenn thematischer Bruch ODER der Chunk zu groß wird
        if sim < 0.70 or len(current_chunk) > chunk_size * 4:  
            chunks.append(current_chunk)
            current_chunk = sentences[i]
        else:
            current_chunk += ". " + sentences[i]

    chunks.append(current_chunk)
    return chunks


# -----------------------------
# 3. HIERARCHICAL CHUNKING (Standardisierte Rückgabe)
# -----------------------------
def chunk_hierarchical(text, chunk_size, overlap):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 3,  # Dynamisch proportional zur AutoML-Größe
        chunk_overlap=overlap * 2,
        separators=["\n## ", "\n### ", "\n", " "]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    parents = parent_splitter.split_text(text)
    final_chunks = []

    for p_id, parent in enumerate(parents):
        children = child_splitter.split_text(parent)
        for child in children:
            # Für AutoML vereinheitlicht als String. 
            # Metadaten-Verknüpfung kannst du bei Bedarf über Payloads steuern.
            final_chunks.append(child)

    return final_chunks


# -----------------------------
# 4. RECURSIVE BASELINE
# -----------------------------
def chunk_recursive(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)