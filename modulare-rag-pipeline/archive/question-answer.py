import json
import os
import random
import requests
from tqdm import tqdm

# --- CONFIG ---
base_path = r"C:\Users\ahmad\Desktop\rag_ml\data\papers"
active_pdf_dir = os.path.join(base_path, "pdfs_active")
golden_set_path = os.path.join(base_path, "modern_golden_set.jsonl")
output_eval = os.path.join(base_path, "eval_set_new.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"
TARGET_QUESTIONS = 50

# 1. IDs der bereits indexierten PDFs sammeln
# Wir mappen die Dateinamen zurück auf die ArXiv-IDs
active_ids = {f.replace(".pdf", "").replace("_", "/") for f in os.listdir(active_pdf_dir) if f.endswith(".pdf")}
print(f"Gefunden: {len(active_ids)} indexierbare Paper.")

# 2. Passende Metadaten aus dem Golden Set laden
available_papers = []
with open(golden_set_path, 'r', encoding='utf-8') as f:
    for line in f:
        p = json.loads(line)
        if p['id'] in active_ids:
            available_papers.append(p)

# Zufällige Auswahl von 50 Papern für das Eval-Set
sample_papers = random.sample(available_papers, min(TARGET_QUESTIONS, len(available_papers)))

# 3. LLM-Generierung (Llama-3.1-8B)
def generate_question(title, abstract):
    prompt = f"""
    Based on the following research paper abstract, generate ONE specific, highly technical question 
    and its corresponding answer. The question must be answerable ONLY by using the information in the abstract.

    Title: {title}
    Abstract: {abstract}

    Return the result in EXACT JSON format:
    {{
        "question": "your technical question",
        "ground_truth": "your detailed answer"
    }}
    """
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json"
        }, timeout=120)
        return json.loads(response.json()['message']['content'])
    except Exception as e:
        return None

# 4. Loop & Speichern
print(f"Generiere {TARGET_QUESTIONS} Testfragen mit Llama-3.1...")
eval_set = []

for paper in tqdm(sample_papers):
    qa = generate_question(paper['title_clean'], paper['abstract_clean'])
    if qa:
        qa['paper_id'] = paper['id']
        eval_set.append(qa)
        
        # Direkt zwischenspeichern
        with open(output_eval, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa) + '\n')

print(f"\n✅ Fertig! {len(eval_set)} Fragen in {output_eval} gespeichert.")
