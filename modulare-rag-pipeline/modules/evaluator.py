class RAGEvaluator:
    def __init__(self, eval_data):
        self.eval_data = eval_data

    def run_benchmark(self, retriever, model, config):
        hits = 0
        rr_sum = 0  # Für die Berechnung des Mean Reciprocal Rank (MRR)
        
        for entry in self.eval_data:
            # Flexibles Auslesen der Keys (unterstützt alte und neue Datensets)
            query = entry.get('question') or entry.get('query') or entry.get('text', '')
            
            raw_expected_id = entry.get('paper_id') or entry.get('expected_paper', '')
            expected_id = str(raw_expected_id).replace('.', '_').replace('/', '_')
            
            # Suche ausführen
            results = retriever.search(
                query, model, 
                top_k_papers=config.get('top_k_papers', 10), 
                top_k_chunks=config.get('top_k_chunks', 10),
                use_expansion=config.get('use_expansion', False),
                collection_name=config.get('collection_name', "stage2_chunks_1024")
            )
            
            # Gefundene IDs sammeln und normalisieren
            found_ids = [
                str(hit.payload.get('paper_id', '')).replace('.', '_').replace('/', '_') 
                for hit in results
            ]
            
            # 1. Hit-Rate berechnen
            if expected_id in found_ids:
                hits += 1
                
                # 2. Reciprocal Rank berechnen (Wo steht der Treffer?)
                rank = found_ids.index(expected_id) + 1  # 1-basierter Index
                rr_sum += 1.0 / rank
            else:
                rr_sum += 0.0

        # Berechne finale Scores
        total = len(self.eval_data)
        accuracy = (hits / total) * 100 if total > 0 else 0
        mrr = (rr_sum / total) if total > 0 else 0
        
        # Wir geben die Genauigkeit zurück (für Ihren AutoML Loop), 
        # könnten hier aber auch beide Metriken tracken
        print(f"📊 Zwischenstand Benchmark -> Hit-Rate: {accuracy:.2f}% | MRR: {mrr:.3f}")
        return accuracy
