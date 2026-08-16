import json
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Paths
db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
eval_input = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_new.jsonl"
eval_filtered = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_filtered.jsonl"
collection_name = "stage1_abstracts_only_active_pdfs"

def filter_eval_set():
    client = qdrant_client.QdrantClient(path=db_path)
    
    # 1. Load all evaluation questions
    with open(eval_input, 'r', encoding='utf-8') as f:
        all_eval_entries = [json.loads(line) for line in f]
    
    print(f"Total questions in eval set: {len(all_eval_entries)}")
    
    available_entries = []
    missing_ids = []

    # 2. Check each paper_id against Qdrant payload
    for entry in all_eval_entries:
        p_id = entry['paper_id']
        
        # Check if this paper_id exists in the collection
        result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=p_id))]
            ),
            limit=1
        )
        
        if result:
            available_entries.append(entry)
        else:
            missing_ids.append(p_id)

    # 3. Save the filtered eval set
    with open(eval_filtered, 'w', encoding='utf-8') as out:
        for entry in available_entries:
            out.write(json.dumps(entry) + '\n')

    print("-" * 30)
    print(f"✅ Indexed & Testable: {len(available_entries)}")
    print(f"❌ Missing from Index: {len(missing_ids)}")
    print(f"Filtered set saved to: {eval_filtered}")
    print("-" * 30)

if __name__ == "__main__":
    filter_eval_set()
