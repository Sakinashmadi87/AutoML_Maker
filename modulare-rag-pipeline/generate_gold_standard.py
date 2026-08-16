import os
import json
import random
from tqdm import tqdm
import torch
from transformers import pipeline

# Modular Imports
from config import PATHS, HP_GRID, IS_COLAB, IS_KAGGLE
from modules.cleaner import clean_scientific_markdown

def main():
    NUM_QUESTIONS = 100
    model_id = HP_GRID["llm_model"]  # "llm_model": "unsloth/llama-3-8b-Instruct"
    device = HP_GRID["device"]       # "cuda" on Kaggle/Colab
    
    # 1. Output location check (Using safe fallback mapping)
    output_file = PATHS.get("eval_set_100q", PATHS.get("eval_set", "/kaggle/working/eval_set_100q.jsonl"))
    
    # Ensure parent folder structure exists
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    
    # 2. Collect papers
    if not os.path.exists(PATHS["markdown"]):
        print(f"❌ Error: Markdown source folder not found at {PATHS['markdown']}")
        return
        
    all_files = [f for f in os.listdir(PATHS["markdown"]) if f.endswith(".md")]
    sampled_files = random.sample(all_files, min(NUM_QUESTIONS, len(all_files)))
    
    print(f"🖥️ Initializing {model_id} on {device.upper()} for synthetic gold-standard generation...")
    
    # 3. Setup Hugging Face Pipeline optimized for Kaggle T4 GPU
    # Llama-3.1-8B fits perfectly in 16GB VRAM using bfloat16 quantization
    generator = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto" if device == "cuda" else None
    )
    
    new_gold_standard = []
    
    for filename in tqdm(sampled_files, desc="Generating QA Grid"):
        paper_id = filename.replace(".md", "").replace("_", "/")
        file_path = os.path.join(PATHS["markdown"], filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        # Run through modular cleaning workflow (clean math/latex filters)
        context_chunk = clean_scientific_markdown(raw_content, replace_math=True)[:3500]
        
        # System Prompt targeted at high-technicality Llama-3.1 synthesis
        prompt = f"""
        You are an expert AI professor reviewing arXiv papers.
        Based ONLY on the following context extracted from a scientific paper, generate exactly ONE highly technical, specific, and non-trivial question.
        The question must be answerable using the text provided. Do not ask generic questions.
        
        CONTEXT:
        {context_chunk}
        
        Provide your output strictly in the following JSON format without any introduction, explanations, or markdown syntax blocks:
        {{"question": "Your technical question here", "ground_truth": "The exact sentence or technical answer from the text"}}
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            outputs = generator(messages, max_new_tokens=256, temperature=0.2, do_sample=False)
            response_text = outputs[0]["generated_text"][-1]["content"]
            
            qa_pair = json.loads(response_text.strip())
            
            new_gold_standard.append({
                "query": qa_pair["question"],
                "expected_paper": paper_id,
                "ground_truth": qa_pair["ground_truth"]
            })
        except Exception:
            continue # Skip edge-cases where the structure formats wrong

    # 4. Save your fresh benchmarking evaluation set
    with open(output_file, "w", encoding="utf-8") as f:
        for item in new_gold_standard:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Generated {len(new_gold_standard)} new evaluation benchmarks saved to: {output_file}")

if __name__ == "__main__":
    main()
