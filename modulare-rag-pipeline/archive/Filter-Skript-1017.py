import json
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

# WICHTIG: Der Pfad zur 1024er Datenbank
db_path_1024 = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
eval_input = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_new.jsonl"
eval_output = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_1017.jsonl"

def create_1017_subset():
    # Verbinde mit der 1024er DB
    client = qdrant_client.QdrantClient(path=db_path_1024)
    collection_name = "stage2_chunks_1024"

    valid_entries = []
    
    print(f"Lade Fragen aus {eval_input}...")
    with open(eval_input, 'r', encoding='utf-8') as f:
        all_questions = [json.loads(line) for line in f]

    print(f"Prüfe Verfügbarkeit in {collection_name}...")
    for entry in all_questions:
        p_id = entry['paper_id']
        
        # Suche nach der Paper-ID in den 1024er Chunks
        res, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchValue(value=p_id))]
            ),
            limit=1
        )
        
        if res:
            valid_entries.append(entry)

    # Speichern des sauberen Subsets
    with open(eval_output, 'w', encoding='utf-8') as f:
        for e in valid_entries:
            f.write(json.dumps(e) + '\n')

    print("-" * 30)
    print(f"✅ Subset erstellt: {len(valid_entries)} von {len(all_questions)} Fragen sind testbar.")
    print(f"Datei gespeichert unter: {eval_output}")
    print("-" * 30)

if __name__ == "__main__":
    create_1017_subset()
