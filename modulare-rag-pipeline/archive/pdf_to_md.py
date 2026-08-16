import os
import json
import pymupdf4llm # Der schnelle Baseline-Parser aus deinem Grid

class ArxivParser:
    def __init__(self, method="PyMuPDF4LLM"):
        self.method = method

    def parse(self, pdf_path):
        if self.method == "PyMuPDF4LLM":
            # Extrahiert Text, Tabellen und Bilder als Markdown
            return pymupdf4llm.to_markdown(pdf_path)
        
        elif self.method == "Marker":
            # Hier käme später der Code für Marker (deep learning basiert)
            pass
            
        return ""

# --- Execution Logic ---
pdf_dir = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\pdfs_active"
output_dir = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

parser = ArxivParser(method="PyMuPDF4LLM")

print(f"Starte Parsing von {len(os.listdir(pdf_dir))} PDFs...")

for i, filename in enumerate(os.listdir(pdf_dir)):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        # Wir speichern als .md für "Structural-Markdown" Chunking
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".md"))
        
        if os.path.exists(output_path): continue
        
        try:
            content = parser.parse(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            if i % 100 == 0:
                print(f"Geparst: {i} / 3735...")
        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

print("Stage 2 Parsing abgeschlossen!")
