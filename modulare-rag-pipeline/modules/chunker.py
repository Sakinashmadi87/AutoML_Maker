import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ArxivChunker:
    def __init__(self, method="recursive", chunk_size=1024, chunk_overlap=100, embedding_model=None):
        self.method = method.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if self.method == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
            )
        elif self.method == "semantic":
            # 🚨 NEU: AI-basiertes Trennen nach Sinnabschnitten
            if embedding_model is None:
                raise ValueError("❌ Für semantisches Chunking muss das Embedding-Modell übergeben werden!")
            
            # Nutzt standardmäßig Perzentil-Abweichungen für die Trennung
            self.splitter = SemanticChunker(
                embedding_model,
                breakpoint_threshold_type="percentile" 
            )
        else:
            raise ValueError(f"❌ Unbekannte Methode: {self.method}")

    def chunk_and_fuse(self, md_content, paper_id, meta_lookup):
        """
        Nimmt rohes Markdown, zerschneidet es modular nach der gewählten Methode
        und fusioniert jeden Chunk mit den Metadaten.
        """
        # 1. Metadaten für dieses spezifische Paper holen
        paper_meta = meta_lookup.get(paper_id, {"title": "Unknown", "abstract": ""})
        
        # 2. Text splitten basierend auf gewählter Methode
        if self.splitter is not None:
            raw_chunks = self.splitter.split_text(md_content)
        else:
            # Fallback: Wenn keine Methode gewählt, nimm den gesamten Text als einen Block
            raw_chunks = [md_content]
        
        fused_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Metadata Fusing (Titel + Abstract-Präfix) - Der Schlüssel für Stage-2-Genauigkeit
            fused_text = (
                f"TITLE: {paper_meta.get('title', 'Unknown')}\n"
                f"ABSTRACT: {paper_meta.get('abstract', '')[:200]}...\n\n"
                f"CHUNK CONTENT:\n{chunk}"
            )
            
            fused_chunks.append({
                "paper_id": paper_id,
                "chunk_index": i,
                "text": fused_text,
                "metadata": {
                    "title": paper_meta.get('title', 'Unknown'),
                    "chunk_method": self.method, # Dokumentiert die Methode im Vektor-Payload
                    "chunk_size": self.chunk_size,
                    "overlap": self.chunk_overlap
                }
            })
        return fused_chunks
