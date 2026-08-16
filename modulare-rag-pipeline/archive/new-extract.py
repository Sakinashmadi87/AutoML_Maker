import json
import os

# Paths
input_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\golden_set_papers.jsonl"
output_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\modern_golden_set.jsonl"

modern_count = 0
target_years = ['2023', '2024', '2025']

if os.path.exists(input_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            paper = json.loads(line)
            # Check the published date field
            pub_date = paper.get('published', '')
            if any(year in pub_date for year in target_years):
                outfile.write(json.dumps(paper) + '\n')
                modern_count += 1
    print(f"Success! Modern Golden Set created with {modern_count} papers.")
else:
    print(f"Error: {input_file} not found. Please verify the path.")
