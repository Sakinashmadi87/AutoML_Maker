import os
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from config import EXPANSION_RULES, IS_COLAB, IS_KAGGLE

class HybridRetriever:
    def __init__(self, db_path=None, url=None, api_key=None):
        self.expansion_rules = EXPANSION_RULES
        
        # 🌐 CLOUD-MODUS (Sowohl für Google Colab als auch für Kaggle)
        if IS_COLAB or IS_KAGGLE or (url is not None):
            print("🌐 Retriever: Initialisiere Qdrant Cloud Client...")
            # Parameter haben Priorität, danach kommen die Umgebungsvariablen
            cloud_url = url or os.getenv('QDRANT_URL')
            cloud_key = api_key or os.getenv('QDRANT_API_KEY')
            
            self.main_client = qdrant_client.QdrantClient(
                url=cloud_url, 
                api_key=cloud_key,
                check_compatibility=False,
                timeout=60.0
            )
            self.chunk_client = self.main_client

            
        # 🏠 LOKALER MODUS (Surface Laptop)
        else:
            print("🏠 Retriever: Initialisiere lokale Qdrant Clients...")
            main_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
            self.main_client = qdrant_client.QdrantClient(path=main_path)
            
            if db_path is None:
                db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
                
            if db_path.lower() == main_path.lower():
                self.chunk_client = self.main_client
            else:
                self.chunk_client = qdrant_client.QdrantClient(path=db_path)

    def search(self, query_text, model, top_k_papers=3, top_k_chunks=5, use_expansion=False, collection_name="stage2_chunks_1024"):
        processed_query = self.expand_query(query_text) if use_expansion else query_text
        query_vector = model.encode(processed_query).tolist()

        # --- STAGE 1: Suche in den Abstracts ---
        stage1_results = self.main_client.query_points(
            collection_name="stage1_abstracts_cleaned",
            query=query_vector,
            limit=top_k_papers
        ).points
        
        raw_ids = [res.payload['paper_id'] for res in stage1_results]
        if not raw_ids: 
            return []

        relevant_paper_ids = []
        for p_id in raw_ids:
            relevant_paper_ids.append(p_id)
            relevant_paper_ids.append(p_id.replace('.', '_'))

        # --- STAGE 2: Gezielte Suche in den Full-Text Chunks ---
        stage2_results = self.chunk_client.query_points(
            collection_name=collection_name, 
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="paper_id", match=MatchAny(any=relevant_paper_ids))]
            ),
            limit=top_k_chunks
        ).points

        return stage2_results

    def expand_query(self, query):
        expanded = query.lower()
        for key, expansion in self.expansion_rules.items():
            if key in expanded:
                if expansion not in expanded:
                    expanded += f" {expansion}"
        return expanded
