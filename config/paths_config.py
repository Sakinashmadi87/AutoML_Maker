# config/paths_config.py
import os

BASE_WORKING_DIR = "/kaggle/working"
PATHS = {
    "pdfs_active": "/kaggle/input/datasets/sakinaahmadi/rag-ml-data/pdfs_active",
    "automl_ground_truth_99": "/kaggle/input/datasets/sakinaahmadi/ground-truth-99/automl_ground_truth_100.json",
    "output_dir": os.path.join(BASE_WORKING_DIR, "output")
}

os.makedirs(PATHS["output_dir"], exist_ok=True)