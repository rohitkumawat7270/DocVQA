import json
import torch
import os
import re
import string
import Levenshtein
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


#  ALL 9 TARGET CATEGORIES
# ==============================
TARGET_TYPES = [
    "Yes/No", "table/list", "form", 
    "figure/diagram", "handwritten", "photograph", 
    "others", "running text" ,"layout"

]

# ---------------------------------------------------------
# OFFICIAL ACADEMIC NORMALIZATION (SQuAD / VQA Standard)
# ---------------------------------------------------------
def normalize_text(s):
    s = str(s).lower().strip()
    
    # Remove LLM Conversational Fillers
    fillers = ["the answer is ", "it is ", "the value is ", "shows ", "indicates "]
    for f in fillers:
        if s.startswith(f):
            s = s.replace(f, "", 1).strip()
            
    # Remove Articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    # Remove Punctuation 
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    
    # Fix Whitespace
    s = ' '.join(s.split())
    return s

def exact_match(pred, answers):
    pred = normalize_text(pred)
    answers = [normalize_text(a) for a in answers]
    return int(any(pred == a for a in answers))

def anls(pred, answers, tau=0.5):
    pred = normalize_text(pred)
    scores = []
    for gt in answers:
        gt = normalize_text(gt)
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            score = 1.0 if pred == gt else 0.0
        else:
            dist = Levenshtein.distance(pred, gt)
            score = 1 - dist / max_len
        scores.append(score if score >= tau else 0)
    return max(scores) if scores else 0


# -----------------------
# MAIN EXECUTION
# -----------------------
if __name__ == "__main__":
    
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # Ultra-HD Vision
    processor = AutoProcessor.from_pretrained(
        model_id,
        max_pixels=1536 * 28 * 28 
    )

    print("Loading Validation Dataset...")
    with open("val_v1.0_withQT.json", "r") as f:
        dataset = json.load(f)["data"]

    # ==========================================
    # AUTOMATIC LOOP FOR ALL CATEGORIES
    # ==========================================
    for target in TARGET_TYPES:
        filtered_data = [
            s for s in dataset
            if target in s.get("question_types", [])
        ]

        if not filtered_data:
            print(f"\nSkipping {target}: No samples found.")
            continue

        print(f"\n" + "="*45)
        print(f"Running Inference for Type: {target}")
        print(f"Total Samples to Evaluate: {len(filtered_data)}")
        print(f"="*45 + "\n")
            
        results = []
        total_em = 0
        total_anls = 0
        count = 0

        for sample in tqdm(filtered_data):
            image_name = os.path.basename(sample["image"])
            image_path = os.path.join("sp_images_folder", image_name)

            if not os.path.exists(image_path):
                continue

            question = sample["question"]
            answers = sample["answers"]

            # Few-Shot SOTA Prompting
            system_prompt = "You are an expert Document Data Extractor. Your job is to extract exact, short answers directly from the image."
            user_prompt = (
                "Read the document carefully and extract the exact text to answer the question.\n"
                "CRITICAL RULES: Do not write full sentences. Do not explain. Return ONLY the final value.\n\n"
                "--- EXAMPLES ---\n"
                "Question: What is the total amount due?\n"
                "Answer: 450\n\n"
                "Question: Who is the letter addressed to?\n"
                "Answer: John Smith\n\n"
                "Question: Is the box checked?\n"
                "Answer: Yes\n"
                "----------------\n\n"
                "NOW PROCESS THIS:\n"
                f"Question: {question}\n"
                "Answer:"
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": user_prompt}]}
            ]
            
            torch.cuda.empty_cache()

            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs, 
                        max_new_tokens=100, 
                        num_beams=3, 
                        do_sample=False
                    )
                output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                raw_prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                continue

            em = exact_match(raw_prediction, answers)
            anls_score = anls(raw_prediction, answers)

            total_em += em
            total_anls += anls_score
            count += 1

            results.append({
                "questionId": sample["questionId"],
                "question": question,
                "raw_model_output": raw_prediction,
                "normalized_prediction": normalize_text(raw_prediction),
                "answers": answers,
                "exact_match": em,
                "anls": anls_score
            })

        accuracy = total_em / count if count else 0
        mean_anls = total_anls / count if count else 0

        print("\n" + "="*30)
        print(f" FINAL RESULTS: {target} ")
        print("="*30)
        print(f"Accuracy (EM): {accuracy:.4f}")
        print(f"ANLS Score: {mean_anls:.4f}")
        print("="*30)
            
        # Save results individually for each category
        output_file = f"results_{target.replace('/','_')}.json"
        with open(output_file, "w") as f:
            json.dump({
                "type": target,
                "accuracy": accuracy,
                "anls": mean_anls,
                "results": results
            }, f, indent=2)

        print(f" Saved to {output_file}\n")