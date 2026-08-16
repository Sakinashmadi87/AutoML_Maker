import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Ensure modules directory path resolution
sys.path.append(os.getcwd())
from config import PATHS, HP_GRID, IS_KAGGLE, IS_COLAB

# 1. PLATFORM AWARE CREDENTIAL HANDSHAKE
def load_cloud_keys():
    url, api_key = None, None
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        url = user_secrets.get_secret("QDRANT_URL")
        api_key = user_secrets.get_secret("QDRANT_API_KEY")
    else:
        url = os.getenv('QDRANT_URL')
        api_key = os.getenv('QDRANT_API_KEY')
    return url, api_key

def main():
    # 2. HARDWARE PIPELINE INITIALIZATION
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Initializing Embedding Layer (BGE-M3) on: {device.upper()}")
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    
    # Connect to Remote Vector Hub
    from modules.retriever import HybridRetriever
    q_url, q_key = load_cloud_keys()
    
    if q_url and q_key:
        print("🌐 Connecting to Production Qdrant Cloud Instance...")
        retriever = HybridRetriever(url=q_url, api_key=q_key)
    else:
        print("🏠 Local fallback active. Mounting path...")
        retriever = HybridRetriever(db_path=PATHS.get("db"))

    # Load GPU Generation Layer via Unsloth weights
    llm_model_id = "unsloth/llama-3-8b-Instruct"
    print(f"🧠 Loading Generation Model [{llm_model_id}] on GPU...")
    generator_pipeline = pipeline(
        "text-generation",
        model=llm_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 3. INTERACTIVE CHAT EXECUTION LAYER
    print("\n" + "="*50)
    print("🚀 PRODUCTION SCIENTIFIC RAG INTERFACE ACTIVE")
    print("="*50)
    
    user_query = "What are the structural scaling rules for compute-optimal transformer architectures?"
    print(f"\nUser Query: '{user_query}'")
    
    # 4. EXECUTING SWEET SPOT RETRIEVAL SWEEP
    print("🔍 Fetching contexts from Cloud Matrix...")
    # Using your optimized settings: Papers=20, Chunks=15, Expansion=False
    chunks = retriever.search(
        query_text=user_query, 
        model=model, 
        top_k_papers=20, 
        top_k_chunks=15, 
        use_expansion=False,
        collection_name="stage2_chunks_1024"
    )
    
    if not chunks:
        print("❌ Search returned 0 relevant document hits.")
        return

    print(f"✅ Retrieved {len(chunks)} high-density context fragments.")

    # 5. GENERATIVE SYNTHESIS LAYER
    # Compile text payload array from payload metadata structures
    context_payload = "\n\n".join([f"[Source Paper: {c.payload.get('paper_id')}]: {c.payload.get('text_llm', c.payload.get('text', ''))}" for c in chunks])
    
    messages = [
        {
            "role": "system", 
            "content": "You are a professional research scientist. Synthesize a comprehensive answer to the user question based ONLY on the technical context paragraphs provided. Cite the source papers explicitly in your text."
        },
        {
            "role": "user", 
            "content": f"Context parameters:\n{context_payload}\n\nQuestion: {user_query}"
        }
    ]

    print("✍️ Synthesizing deep technical analysis answer...")
    outputs = generator_pipeline(messages, max_new_tokens=512, temperature=0.2, do_sample=False)
    final_answer = outputs[0]["generated_text"][-1]["content"]

    print("\n" + "="*25 + " FINAL SYSTEM ANSWER " + "="*25 + "\n")
    print(final_answer)
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
