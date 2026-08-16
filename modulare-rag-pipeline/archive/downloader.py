import json
import os
import time
import requests

# Final Paths
input_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\modern_golden_set.jsonl"
pdf_dir = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\pdfs"

if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

def download_pdf(pdf_url, paper_id):
    safe_id = paper_id.replace('/', '_')
    file_path = os.path.join(pdf_dir, f"{safe_id}.pdf")
    if os.path.exists(file_path): return "Exists"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(pdf_url, headers=headers, timeout=60)
        if response.status_code == 200:
            with open(file_path, 'wb') as f: f.write(response.content)
            return "Success"
        return f"Failed ({response.status_code})"
    except Exception as e: return f"Error: {str(e)}"

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        paper = json.loads(line)
        url = paper.get('pdf_url') + ".pdf" if not paper.get('pdf_url').endswith('.pdf') else paper.get('pdf_url')
        
        status = download_pdf(url, paper['id'])
        if i % 20 == 0: # Print progress every 20 papers
            print(f"Progress: {i}/40529 | ID: {paper['id']} | Status: {status}")
        
        if status == "Success":
            time.sleep(3) # DO NOT REMOVE: Prevents IP ban
print("Download process completed.")