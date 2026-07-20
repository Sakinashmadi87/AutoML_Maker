# modules/pdf_parser.py
import os
import json

# --- Checkpoint-Manager ---
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(checkpoint_file, done_files):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(list(done_files), f, ensure_ascii=False, indent=2)


# --- Parser-Klasse (GPU-cached) ---
class ArxivParser:
    def __init__(self, method="Docling"):
        if method not in ["PyMuPDF4LLM", "Docling", "Marker"]:
            raise ValueError("Methode muss 'PyMuPDF4LLM', 'Docling' oder 'Marker' sein.")
        self.method = method
        self.converter = None
        
        if self.method == "Docling":
            print("🚀 Initialisiere Docling (Modelle werden geladen)...")
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            
        elif self.method == "Marker":
            print("🚀 Initialisiere Marker-Parser (Transformers-Modelle werden geladen)...")
            from marker.convert import Converter
            self.converter = Converter()

    def parse(self, pdf_path):
        if self.method == "PyMuPDF4LLM":
            try:
                import pymupdf4llm
                return pymupdf4llm.to_markdown(pdf_path)
            except Exception as e:
                raise RuntimeError(f"PyMuPDF4LLM fehlgeschlagen: {e}")

        elif self.method == "Docling":
            try:
                result = self.converter.convert(pdf_path)
                return result.document.export_to_markdown()
            except Exception as e:
                raise RuntimeError(f"Docling fehlgeschlagen: {e}")
                
        elif self.method == "Marker":
            try:
                rendered, _, _ = self.converter(pdf_path)
                return rendered
            except Exception as e:
                raise RuntimeError(f"Marker fehlgeschlagen: {e}")


# --- Massenverarbeitung (Erhält Pfade von außen injiziert) ---
def main(pdfs_dir, output_markdown_dir, output_root_dir, run_mode_param="V1"):
    os.makedirs(output_markdown_dir, exist_ok=True)

    checkpoint_file = os.path.join(output_root_dir, "checkpoint_parsing.json")
    parser = ArxivParser(method="Docling")
    done_files = load_checkpoint(checkpoint_file)

    all_pdfs = sorted([f for f in os.listdir(pdfs_dir) if f.endswith(".pdf")])
    total_all = len(all_pdfs)
    
    if total_all == 0:
        print(f"❌ Keine PDFs im Ordner gefunden! Pfad prüfen: {pdfs_dir}")
        return

    # Viertel-Aufteilung
    q1 = total_all // 4
    q2 = q1 * 2
    q3 = q1 * 3

    if run_mode_param == "V1":
        active_pdfs = all_pdfs[:q1]
    elif run_mode_param == "V2":
        active_pdfs = all_pdfs[q1:q2]
    elif run_mode_param == "V3":
        active_pdfs = all_pdfs[q2:q3]
    elif run_mode_param == "V4":
        active_pdfs = all_pdfs[q3:]
    else:
        active_pdfs = all_pdfs

    total_active = len(active_pdfs)
    print(f"\n📦 MODUS {run_mode_param}: Verarbeite {total_active} PDFs.")

    for i, filename in enumerate(active_pdfs):
        if filename in done_files:
            continue

        pdf_path = os.path.join(pdfs_dir, filename)
        output_path = os.path.join(output_markdown_dir, filename.replace(".pdf", ".md"))

        try:
            print(f"[{i+1}/{total_active}] Verarbeite: {filename}")
            content = parser.parse(pdf_path)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            done_files.add(filename)
            save_checkpoint(checkpoint_file, done_files)

        except Exception as e:
            print(f"❌ Fehler bei {filename}: {e}")
            continue


# --- In-Memory Schnittstellen für das Master-Skript ---
_PARSER_CACHE = {}

def _get_cached_parser(method: str):
    mapping = {"pymupdf4llm": "PyMuPDF4LLM", "docling": "Docling", "maker": "Marker"}
    method_key = mapping.get(method, "Docling")
    if method_key not in _PARSER_CACHE:
        _PARSER_CACHE[method_key] = ArxivParser(method=method_key)
    return _PARSER_CACHE[method_key]

def parse_with_pymupdf(selected_papers: list) -> list:
    parser_instance = _get_cached_parser("pymupdf4llm")
    return [parser_instance.parse(p) if p.strip() else "" for p in selected_papers]

def parse_with_docling(selected_papers: list) -> list:
    parser_instance = _get_cached_parser("docling")
    return [parser_instance.parse(p) if p.strip() else "" for p in selected_papers]

def parse_with_maker(selected_papers: list) -> list:
    parser_instance = _get_cached_parser("maker")
    return [parser_instance.parse(p) if p.strip() else "" for p in selected_papers]