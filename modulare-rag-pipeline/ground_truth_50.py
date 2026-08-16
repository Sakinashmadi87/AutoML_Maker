import json
import re
import random
from tqdm import tqdm
import pandas as pd

def extract_json(text: str):
    # 1. JSON in ```json ... ``` suchen
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        # 2. Fallback: erstes {...} im Text (jetzt non-greedy mit *?)
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)

    # 3. Reparatur: fehlende schließende Klammer
    if candidate.count("{") > candidate.count("}"):
        candidate += "}"

    # 4. JSON parsen
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def is_valid_chunk(text: str) -> bool:
    if not text:
        return False
    if len(text) < 200:
        return False
    if len(text.split()) < 30:
        return False
    if text.count("$") > len(text) * 0.2:
        return False
    return True

def robust_generate_qa(item, pipe, tokenizer, system_prompt):
    context = item["context"]
    if not is_valid_chunk(context):
        return None

    user_content = f"""Generiere basierend auf dem Textabschnitt eine hochspezifische Frage 
und eine exakte Antwort als valides JSON-Objekt.

TEXTABSCHNITT:
{context}

FORMAT:
{{
  "question": "Deine Frage hier?",
  "answer": "Die exakte Antwort."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Erster Versuch (sehr deterministisch)
    outputs = pipe(
        prompt,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        max_new_tokens=256,
        temperature=0.1,
        return_full_text=False,
    )
    response_text = outputs[0]["generated_text"].strip()
    qa = extract_json(response_text)

    # Zweiter Versuch (Fallback mit etwas mehr Kreativität), falls JSON kaputt
    if qa is None:
        outputs = pipe(
            prompt,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            max_new_tokens=256,
            temperature=0.3,
            return_full_text=False,
        )
        response_text = outputs[0]["generated_text"].strip()
        qa = extract_json(response_text)

    if qa is None:
        return None

    # Keys normalisieren
    normalized = {k.lower().strip(): v for k, v in qa.items()}
    question = normalized.get("question")
    answer = normalized.get("answer")

    if not question or not answer:
        return None

    return {
        "id": None,  
        "arxiv_id": item["arxiv_id"],
        "query": str(question).strip(),
        "gold_standard_answer": str(answer).strip(),
        "expected_context": context,
    }

# Diese Funktion kapselt den Ablauf sauber, sodass du sie im Notebook aufrufen kannst
def run_ground_truth_generation(all_valid_chunks, pipe, tokenizer, system_prompt, output_path):
    if len(all_valid_chunks) > 100:
        selected_chunks = random.sample(all_valid_chunks, 100)
    else:
        selected_chunks = all_valid_chunks

    ground_truth_dataset = []

    for item in tqdm(selected_chunks, desc="Generiere Ground Truth"):
        # HIER WAR DER FEHLER BEHOBEN: Jetzt werden alle Parameter übergeben!
        qa = robust_generate_qa(item, pipe, tokenizer, system_prompt)
        if qa:
            qa["id"] = len(ground_truth_dataset) + 1
            ground_truth_dataset.append(qa)
        if len(ground_truth_dataset) >= 50:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_dataset, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 FERTIG! {len(ground_truth_dataset)} Frage-Antwort-Paare erfolgreich generiert.")
    print(f"💾 Gespeichert unter: {output_path}")

    if ground_truth_dataset:
        df_preview = pd.DataFrame(ground_truth_dataset).head(3)
        print("\n🔍 Vorschau:")
        print(df_preview[["arxiv_id", "query", "gold_standard_answer"]])
        
    return ground_truth_dataset