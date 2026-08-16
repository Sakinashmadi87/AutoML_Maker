import pymupdf4llm
import arxiv

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever

# 1. Search and Download PDF
search = arxiv.Search(query="RAG", max_results=1)
paper = next(search.results())

pdf_path = paper.download_pdf(filename="temp_paper.pdf")

# 2. Convert PDF → Markdown
md_text = pymupdf4llm.to_markdown(pdf_path)

# 3. Split by Markdown Headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(md_text)

# 4. Metadata hinzufügen
for chunk in chunks:
    chunk.metadata["title"] = paper.title
    chunk.metadata["authors"] = [a.name for a in paper.authors]
    chunk.metadata["url"] = paper.entry_id
    chunk.metadata["published"] = paper.published.strftime("%Y-%m-%d")

print("Chunks erstellt:", len(chunks))
print("Beispiel Metadata:", chunks[0].metadata)

# 5. Vector Store (FAISS)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 7. Hybrid Search (manuell, statt EnsembleRetriever)
query = "What is the core objective of RAG-Gym?"

docs_vector = vector_retriever.invoke(query)
docs_bm25 = bm25_retriever.invoke(query)

# Kombination + Duplikate entfernen
seen = set()
retrieved_docs = []

for doc in docs_vector + docs_bm25:
    text = doc.page_content
    if text not in seen:
        seen.add(text)
        retrieved_docs.append(doc)

# 8. Ergebnisse anzeigen
print(f"\nRetrieved {len(retrieved_docs)} chunks (Hybrid Search):")

for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk {i+1}")
    print("Section:", doc.metadata.get("Header 2", "N/A"))
    print(doc.page_content[:200] + "...")