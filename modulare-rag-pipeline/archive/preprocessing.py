import json
import os
import re

# Pfad zu den JSONL-Dateien mit den Papers
base_path = r"C:\Users\ahmad\Desktop\rag_ml\data\papers"
files = [
    "cs_ai_papers.jsonl", "cs_cl_papers.jsonl", "cs_cv_papers.jsonl",
    "cs_ir_papers.jsonl", "cs_lg_papers.jsonl", "cs_ne_papers.jsonl"
]
# Deduplication der Papers basierend auf der 'id' (arXiv ID) durchführen
unique_ids = set()
output_file = os.path.join(base_path, "cleaned_ai_papers.jsonl")

count = 0
with open(output_file, 'w', encoding='utf-8') as outfile:
    for file_name in files:
        file_path = os.path.join(base_path, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    paper = json.loads(line)
                    pid = paper['id']
                    
                    if pid not in unique_ids:
                        unique_ids.add(pid)
                        outfile.write(json.dumps(paper) + '\n')
                        count += 1

print(f"Done! Created {output_file}")
print(f"Total unique papers saved: {count}")

# Textbereinigung (LaTeX-Formeln entfernen, Zeilenumbrüche bereinigen)
input_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\cleaned_ai_papers.jsonl"
output_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\processed_ai_papers.jsonl"

def clean_text(text):
    if not text:
        return ""
    # 1. LaTeX-Formeln ($...$) entfernen
    text = re.sub(r'\$.*?\$', '', text)
    # 2. Zeilenumbrüche und überschüssige Leerzeichen entfernen
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        paper = json.loads(line)
        
        # Titel und Abstract bereinigen
        paper['title_clean'] = clean_text(paper['title'])
        paper['abstract_clean'] = clean_text(paper['abstract'])
        
        # In die neue Datei schreiben
        outfile.write(json.dumps(paper) + '\n')

print(f"Fertig! 288.368 Paper wurden bereinigt und in {output_file} gespeichert.")