import requests

class ArxivGenerator:
    def __init__(self, model_name="phi3:mini", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate_answer(self, query, context_chunks):
        # Only take the top 3 chunks to save RAM
        limited_chunks = context_chunks[:3]
        context_text = "\n\n---\n\n".join([c.payload['text_llm'] for c in limited_chunks])
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Brief scientific assistant. Use context provided."},
                {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048, # Smaller context window for 8GB RAM
                "num_thread": 4  # Limit CPU threads to prevent freezing
            }
        }

        try:
            # Increase timeout to 300 seconds (5 minutes)
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            return f"Ollama Error: {str(e)}"
