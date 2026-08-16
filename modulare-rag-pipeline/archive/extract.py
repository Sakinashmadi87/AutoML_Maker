import json
import yake
import os

input_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\processed_ai_papers.jsonl"
output_file = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\tagged_ai_papers.jsonl"

# Initialize YAKE
# n=2 extracts up to bigrams; top=5 gets the 5 best keywords
# Note: Lower YAKE scores indicate higher relevance
kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=5, features=None)

print("Starting keyword extraction for 288k papers...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for i, line in enumerate(infile):
        paper = json.loads(line)
        
        # Combine title and abstract for richer context
        text_to_analyze = f"{paper.get('title_clean', '')} {paper.get('abstract_clean', '')}"
        
        # Extract keywords
        keywords_with_scores = kw_extractor.extract_keywords(text_to_analyze)
        
        # Store only the keyword strings
        paper['keywords'] = [kw for kw, score in keywords_with_scores]
        
        # Write to the new file
        outfile.write(json.dumps(paper) + '\n')
        
        # Progress tracker
        if i % 10000 == 0:
            print(f"Processed {i} papers...")

print(f"Success! Tagged file saved at: {output_file}")
