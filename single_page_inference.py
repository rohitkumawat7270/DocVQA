import os
import json
import re
import torch
import Levenshtein
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# -----------------------
# 1. Configuration
# -----------------------
TARGET_TYPE = "layout"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# -----------------------
# 2. Data Preparation (Downloads data directly)
# -----------------------
if not os.path.exists("val.json"):
    print("Downloading full single-page DocVQA dataset...")
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    os.makedirs("documents", exist_ok=True)
    formatted_data = []
    
    print("Extracting images...")
    for i, row in enumerate(tqdm(dataset)):
        image_path = f"documents/val_image_{i}.png"
        if not os.path.exists(image_path):
            row["image"].save(image_path)
        formatted_data.append({
            "questionId": row.get("questionId", i),
            "question": row["question"],
            "question_types": row.get("question_types", ["layout"]),
            "image": image_path,
            "answers": row["answers"]
        })
        
    with open("val.json", "w") as f:
        json.dump({"data": formatted_data}, f, indent=4)
    print("Full Single-page data ready!\n")

# -----------------------
# 3. Improvement & Metric Functions
# -----------------------
def clean_prediction(pred):
    pred = pred.strip().rstrip(".:,;-")
    num = re.search(r"\b\d+(\.\d+)?\b", pred)
    if num: return num.group()
    if "yes" in pred.lower(): return "yes"
    if "no" in pred.lower(): return "no"
    return pred

def exact_match(pred, answers):
    pred = pred.lower().strip()
    answers = [a.lower().strip() for a in answers]
    return int(any(pred == a for a in answers))

def anls(pred, answers, tau=0.5):
    pred = pred.lower().strip()
    scores = []
    for gt in answers:
        gt = gt.lower().strip()
        dist = Levenshtein.distance(pred, gt)
        score = 1 - dist / max(len(pred), len(gt))
        scores.append(score if score >= tau else 0)
    return max(scores) if scores else 0

# -----------------------
# 4. Load Model
# -----------------------
print(f"Loading {MODEL_ID} onto GPU...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# -----------------------
# 5. Load Dataset
# -----------------------
with open("val.json", "r") as f:
    dataset = json.load(f)["data"]

filtered_data = [s for s in dataset if TARGET_TYPE in s.get("question_types", [])]
print(f"\nRunning High-Resolution inference for type: {TARGET_TYPE}")
print(f"Total samples to process: {len(filtered_data)}\n")

results = []
total_em, total_anls, count = 0, 0, 0

# -----------------------
# 6. Main Inference Loop
# -----------------------
for sample in tqdm(filtered_data):
    image_path = sample["image"]
    question = sample["question"]
    answers = sample["answers"]

    if not os.path.exists(image_path):
        continue

    # Loading image in original resolution (Best for IIT Server)
    image = Image.open(image_path).convert("RGB")

    prompt = (
        "Read the document carefully and answer exactly using text from the document. "
        "Give a short answer only. Do not explain.\n"
        "Do not add extra 0 after decimal place.\n"
        "Do not add full stop if not required.\n"
        f"Question: {question}"
    )

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    prediction = clean_prediction(prediction)  

    em = exact_match(prediction, answers)
    anls_score = anls(prediction, answers)

    total_em += em
    total_anls += anls_score
    count += 1

    results.append({
        "questionId": sample["questionId"],
        "question": question,
        "prediction": prediction,
        "answers": answers,
        "exact_match": em,
        "anls": anls_score
    })

# -----------------------
# 7. Final Output
# -----------------------
accuracy = total_em / count if count else 0
mean_anls = total_anls / count if count else 0

print("\n===== FINAL RESULTS =====")
print(f"Accuracy (Exact Match): {accuracy:.4f}")
print(f"ANLS Score: {mean_anls:.4f}")

output_file = f"single_page_results_{TARGET_TYPE}_high_res.json"
with open(output_file, "w") as f:
    json.dump({"type": TARGET_TYPE, "accuracy": accuracy, "anls": mean_anls, "results": results}, f, indent=2)