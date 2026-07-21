# config/paths_config.py
import os

BASE_WORKING_DIR = "/kaggle/working"

PATHS = {
    # Gold Standard
    "automl_ground_truth_99": "/kaggle/input/datasets/sakinaahmadi/ground-truth-99/automl_ground_truth_100.json",
    "output_dir": os.path.join(BASE_WORKING_DIR, "output"),
    
    # Ordner mit den fertig geparsten Markdown-Dateien der 3 Parser:
    "markdown_pymupdf": "/kaggle/input/datasets/sakinaahmadi/rag-ml-data/extracted_markdown",
    "markdown_docling": "/kaggle/input/datasets/sakinaahmadi87/extracted-markdown-docling/markdown",
    "markdown_marker": "/kaggle/input/datasets/sakinaahmadi/marker-parsed-papers-3734/marker_parsed_papers",
}

os.makedirs(PATHS["output_dir"], exist_ok=True)