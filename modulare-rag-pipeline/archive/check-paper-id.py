import json
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Paths
db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
eval_path = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set.jsonl"
collection_name = "stage1_abstracts_only_active_pdfs"

def verify_database_coverage():
    client = qdrant_client.QdrantClient(path=db_path)
    
    # 1. Load IDs from your Eval Set
    eval_ids = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_ids.append(json.loads(line)['paper_id'])
    
    print(f"Checking coverage for {len(eval_ids)} evaluation papers...")

    found_count = 0
    missing = []

    # 2. Check each ID in Qdrant
    for p_id in eval_ids:
        # We scroll the DB for an exact match of the paper_id in the payload
        result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=p_id))]
            ),
            limit=1
        )
        
        if result:
            found_count += 1
        else:
            missing.append(p_id)

    # 3. Report
    print("-" * 30)
    print(f"✅ Found in DB: {found_count}")
    print(f"❌ Missing from DB: {len(missing)}")
    
    if missing:
        print("\nMissing Paper IDs (Check if PDFs exist for these):")
        for m in missing:
            print(f" - {m}")
    else:
        print("\nPerfect! Your database covers 100% of your evaluation set.")
    print("-" * 30)

if __name__ == "__main__":
    verify_database_coverage()
