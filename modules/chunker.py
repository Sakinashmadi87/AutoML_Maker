from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ---------------------------------------------------------
# 1. MARKDOWN SECTION SPLITTER
# ---------------------------------------------------------
def chunk_markdown_section(text, chunk_size, overlap):
    sections = re.split(r"\n(?=#)", text)

    if len(sections) <= 1:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
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
# 2. SEMANTIC CHUNKING (KORRIGIERT)
# ---------------------------------------------------------
def chunk_semantic(text, chunk_size, overlap):
    # Modell nur einmal laden
    if "semantic_model" not in globals():
        globals()["semantic_model"] = SentenceTransformer("all-MiniLM-L6-v2")

    model = globals()["semantic_model"]

    # Text in Sätze splitten
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    # Hard cap für AutoML
    sentences = sentences[:2000]

    # Embeddings berechnen
    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
        )

        # thematischer Bruch oder Chunk zu groß
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
def chunk_hierarchical(text, chunk_size, overlap):
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
def chunk_recursive(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)
