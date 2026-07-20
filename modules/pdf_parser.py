# modules/parser.py
import os
import json
import sys

# --- Pfad-Korrektur für Unterordner ---
# Damit das Skript aus 'modules/' die 'config.py' im Hauptverzeichnis findet:
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)

# Jetzt fügen wir auch das aktuelle Arbeitsverzeichnis hinzu, falls nötig
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

try:
    import config
except ImportError as e:
    raise ImportError(f"config.py konnte nicht geladen werden. Suchpfade: {sys.path}. Fehler: {e}")

# --- Lade Pfade aus config ---
PDF_DIR = config.PATHS["pdfs_active"]
OUTPUT_DIR = config.PATHS["markdown"]
# Checkpoint wird sicher im Output-Root der jeweiligen Cloud abgelegt
CHECKPOINT_FILE = os.path.join(config.PATHS["output_root"], "checkpoint_parsing.json")

# --- Checkpoint-Manager ---
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(done_files):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done_files), f, ensure_ascii=False, indent=2)


# --- Parser-Klasse (Optimiert für GPU-Vermeidung von Re-Initialisierung) ---
class ArxivParser:
    def __init__(self, method="Docling"):
        if method not in ["PyMuPDF4LLM", "Docling", "Marker"]:
            raise ValueError("Methode muss 'PyMuPDF4LLM', 'Docling' oder 'Marker' sein.")
        self.method = method
        self.converter = None
        
        # Docling Modelle genau EINMAL beim Start in die GPU laden
        if self.method == "Docling":
            print("🚀 Initialisiere Docling (Modelle werden geladen)...")
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            
        # Marker Modelle genau EINMAL in die GPU laden
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


# --- Haupt-Logik mit Parameter-Steuerung für die Massenverarbeitung ---
def main(run_mode_param=None):
    # Sicherstellen, dass Ausgabeverzeichnis existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = ArxivParser(method="Docling")
    done_files = load_checkpoint()

    # Liste aller PDFs (strikt sortiert für identische Aufteilung)
    all_pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    total_all = len(all_pdfs)
    
    if total_all == 0:
        print(f"❌ Keine PDFs im Ordner gefunden! Pfad prüfen: {PDF_DIR}")
        return

    # --- Parameter-Auswertung für die 4 Viertel ---
    RUN_MODE = run_mode_param if run_mode_param else "V1"
    
    q1 = total_all // 4
    q2 = q1 * 2
    q3 = q1 * 3

    if RUN_MODE == "V1":
        active_pdfs = all_pdfs[:q1]
        print(f"\n📦 MODUS VIERTEL 1: PDFs 1 bis {q1} ({len(active_pdfs)} Dateien)")
    elif RUN_MODE == "V2":
        active_pdfs = all_pdfs[q1:q2]
        print(f"\n📦 MODUS VIERTEL 2: PDFs {q1+1} bis {q2} ({len(active_pdfs)} Dateien)")
    elif RUN_MODE == "V3":
        active_pdfs = all_pdfs[q2:q3]
        print(f"\n📦 MODUS VIERTEL 3: PDFs {q2+1} bis {q3} ({len(active_pdfs)} Dateien)")
    elif RUN_MODE == "V4":
        active_pdfs = all_pdfs[q3:]
        print(f"\n📦 MODUS VIERTEL 4: PDFs {q3+1} bis {total_all} ({len(active_pdfs)} Dateien)")
    else:
        active_pdfs = all_pdfs
        print(f"\n📦 MODUS ALL: Alle {total_all} PDFs")

    total_active = len(active_pdfs)
    print(f"Bereits verarbeitet in diesem Block: {len(done_files & set(active_pdfs))}/{total_active}\n")

    for i, filename in enumerate(active_pdfs):
        if filename in done_files:
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".md"))

        try:
            print(f"[{i+1}/{total_active}] Verarbeite: {filename}")
            content = parser.parse(pdf_path)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Fortschritt sichern
            done_files.add(filename)
            save_checkpoint(done_files)

            if (i + 1) % 10 == 0:
                print(f"✨ Zwischenstand: {i+1}/{total_active} abgeschlossen.")

        except Exception as e:
            print(f"❌ Fehler bei {filename}: {e}")
            continue

    print(f"\n✅ Durchlauf für Modus '{RUN_MODE}' erfolgreich abgeschlossen.")


# =====================================================================
# AUTOML IN-MEMORY INTERFACES FOR build_index_for_trial
# =====================================================================

# Globale Instanzen-Caches gegen RAM/VRAM-Explosionen auf Kaggle
_PARSER_CACHE = {}

def _get_cached_parser(method: str):
    """Sichert, dass jeder Parser pro Kaggle-Session nur 1x initialisiert wird."""
    mapping = {
        "pymupdf4llm": "PyMuPDF4LLM",
        "docling": "Docling",
        "maker": "Marker"
    }
    
    method_key = mapping.get(method, "Docling")

    if method_key not in _PARSER_CACHE:
        _PARSER_CACHE[method_key] = ArxivParser(method=method_key)
        
    return _PARSER_CACHE[method_key]


def parse_with_pymupdf(selected_papers: list) -> list:
    """Konvertiert ausgewählte PDFs im RAM mittels PyMuPDF4LLM in Markdown-Strings."""
    parser_instance = _get_cached_parser("pymupdf4llm")
    parsed_texts = []
    
    for paper_path in selected_papers:
        try:
            md_content = parser_instance.parse(paper_path)
            parsed_texts.append(md_content)
        except Exception as e:
            print(f"⚠️ Fehler bei parse_with_pymupdf für {paper_path}: {e}")
            parsed_texts.append("")
            
    return parsed_texts


def parse_with_docling(selected_papers: list) -> list:
    """Konvertiert ausgewählte PDFs im RAM mittels Docling in Markdown-Strings."""
    parser_instance = _get_cached_parser("docling")
    parsed_texts = []
    
    for paper_path in selected_papers:
        try:
            md_content = parser_instance.parse(paper_path)
            parsed_texts.append(md_content)
        except Exception as e:
            print(f"⚠️ Fehler bei parse_with_docling für {paper_path}: {e}")
            parsed_texts.append("")
            
    return parsed_texts


def parse_with_maker(selected_papers: list) -> list:
    """Konvertiert ausgewählte PDFs im RAM mittels Marker in Markdown-Strings."""
    parser_instance = _get_cached_parser("maker")
    parsed_texts = []
    
    for paper_path in selected_papers:
        try:
            md_content = parser_instance.parse(paper_path)
            parsed_texts.append(md_content)
        except Exception as e:
            print(f"⚠️ Fehler bei parse_with_maker für {paper_path}: {e}")
            parsed_texts.append("")
            
    return parsed_texts


if __name__ == "__main__":
    main()