# modules/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ---------------------------------------------------------
# GLOBAL CACHE FÜR SEMANTIC MODEL
# ---------------------------------------------------------
_SEMANTIC_MODEL = None

def get_semantic_model():
    """Lädt das Semantic-Modell nur einmal."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL

# ---------------------------------------------------------
# 1. HYBRID / MARKDOWN SECTION SPLITTER
# ---------------------------------------------------------
def chunk_hybrid(text, chunk_size, overlap=None):
    """
    Hybrid Chunking: Markdown-Sections + Recursive Splitting.
    - Erhält Dokumentenstruktur durch Überschriften.
    - Fallback auf Recursive, wenn keine Überschriften vorhanden.
    """
    if overlap is None:
        overlap = int(chunk_size * 0.1)
        
    # Markdown-Überschriften erkennen (#, ##, ###, ####)
    sections = re.split(r"\n(?=#{1,4}\s)", text)

    if len(sections) <= 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    final_chunks = []
    for sec in sections:
        final_chunks.extend(splitter.split_text(sec))

    return final_chunks

# ---------------------------------------------------------
# 2. SEMANTIC CHUNKING
# ---------------------------------------------------------
def chunk_semantic(text, chunk_size, overlap=None):
    """
    Semantic Chunking basierend auf Satz-Ähnlichkeit.
    OVERLAP WIRD IGNORIERT (semantische Grenzen benötigen keine Überlappung).
    """
    model = get_semantic_model()

    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    # Begrenzung für Performance
    sentences = sentences[:2000]
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
        )

        if sim < 0.70 or len(current_chunk) > chunk_size * 4:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
        else:
            current_chunk += ". " + sentences[i]

    chunks.append(current_chunk)
    return chunks

# ---------------------------------------------------------
# 3. HIERARCHICAL CHUNKING
# ---------------------------------------------------------
def chunk_hierarchical(text, chunk_size, overlap=None):
    """
    Hierarchical Chunking: Parent (3×) + Child (1×) Struktur.
    - Parent: Große Chunks für kontextuelles Verständnis.
    - Child: Kleine Chunks für präzises Retrieval.
    """
    if overlap is None:
        overlap = int(chunk_size * 0.1)
        
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 3,
        chunk_overlap=overlap * 2,
        separators=["\n## ", "\n### ", "\n", " "]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    parents = parent_splitter.split_text(text)
    final_chunks = []

    for parent in parents:
        children = child_splitter.split_text(parent)
        final_chunks.extend(children)

    return final_chunks

# ---------------------------------------------------------
# 4. RECURSIVE BASELINE
# ---------------------------------------------------------
def chunk_recursive(text, chunk_size, overlap=None):
    """
    Recursive Baseline: Einfache, regelbasierte Segmentierung.
    - Schnell, aber keine semantische Kohärenz.
    - Dient als Baseline für Ablationsstudien.
    """
    if overlap is None:
        overlap = int(chunk_size * 0.1)
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)