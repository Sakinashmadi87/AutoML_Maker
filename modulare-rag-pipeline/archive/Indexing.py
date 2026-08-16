from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
from pathlib import Path
import fitz

# API Key laden
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Ordner
data_path = Path("data/papers")
db_path = Path("chroma_db")

# Embedding Modell
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Funktion für PDF Infos
def get_info(path):
    doc = fitz.open(path)
    info = doc.metadata
    doc.close()
    return info

# Prüfen ob DB existiert
if db_path.exists():
    print("DB existiert schon, lade sie...")
    vectorstore = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings
    )

else:
    print("Keine DB gefunden, starte neu...")

    all_docs = []

    pdfs = list(data_path.glob("*.pdf"))
    print("Anzahl PDFs:", len(pdfs))

    for pdf in pdfs:
        print("Lade:", pdf)

        info = get_info(pdf)

        loader = PyMuPDF4LLMLoader(str(pdf))
        docs = loader.load()

        print("Seiten geladen:", len(docs))

        # einfache metadata hinzufügen
        for d in docs:
            d.metadata["source"] = pdf.name
            d.metadata["title"] = info.get("title")

        all_docs.extend(docs)

    print("Gesamt Dokumente:", len(all_docs))

    # Chunking
    splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)
    print("Chunks erstellt:", len(chunks))

    # Vector DB erstellen
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(db_path)
    )

    print("DB gespeichert!")

# Test Query
query = "What is OmniDocBench?"

docs = vectorstore.similarity_search(query, k=3)

print("\nErgebnisse:\n")

for d in docs:
    print("Quelle:", d.metadata.get("source"))
    print(d.page_content[:150])
    print("-----")

# LLM
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

result = qa.invoke({"query": query})

print("\nAntwort:\n")
print(result["result"])