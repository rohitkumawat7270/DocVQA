import os
import json
import torch
import Levenshtein
import re
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# -----------------------
# 1. Configuration
# -----------------------
IMAGE_DIR = "multidoc/images"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# -----------------------
# 2. Evaluation Metrics
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
# 3. Load Model
# -----------------------
print(f"Loading {MODEL_ID} onto GPU...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# -----------------------
# 4. Core Routing Functions
# -----------------------
def get_page_summary(page_id):
    """Generates an internal spatial context summary of a single page."""
    image_path = os.path.join(IMAGE_DIR, page_id + ".jpg")
    if not os.path.exists(image_path):
        return ""
    
    # Keeping thumbnail here to make routing ultra-fast
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((512, 512)) 

    prompt = "Give a short summary of this page in 5 words."
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    summary = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip().lower()
    return summary

def select_best_page(page_ids, question):
    best_page = None
    best_score = -1
    for page_id in page_ids:
        summary = get_page_summary(page_id)
        if not summary: continue
        score = sum(1 for w in question.lower().split() if w in summary)
        if score > best_score:
            best_score = score
            best_page = page_id
    return best_page

# -----------------------
# 5. Main Inference Loop
# -----------------------
if not os.path.exists("multi_val.json"):
    raise FileNotFoundError("multi_val.json not found! Run data_preparation.py first.")

with open("multi_val.json", "r") as f:
    dataset = json.load(f)["data"]

print(f"\nProcessing FULL dataset. Total Documents: {len(dataset)}")

results = []
total_em, total_anls, count = 0, 0, 0

for sample in tqdm(dataset):
    question = sample["question"]
    answers = sample["answers"]
    page_ids = sample["page_ids"]

    # Step 1: Routing
    best_page = select_best_page(page_ids, question)
    if best_page is None:
        continue

    image_path = os.path.join(IMAGE_DIR, best_page + ".jpg")

    # Step 2: Target Extraction (Original High-Resolution)
    answer_image = Image.open(image_path).convert("RGB")

    prompt = (
        "Read the document carefully and answer exactly using text from the document. "
        "Give a short answer only. Do not explain.\n"
        f"Question: {question}"
    )

    messages = [{"role": "user", "content": [{"type": "image", "image": answer_image}, {"type": "text", "text": prompt}]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    prediction = clean_prediction(prediction)

    # Scorig
    em = exact_match(prediction, answers)
    anls_score = anls(prediction, answers)

    total_em += em
    total_anls += anls_score
    count += 1

    results.append({
        "questionId": sample["questionId"],
        "prediction": prediction,
        "answers": answers,
        "exact_match": em,
        "anls": anls_score
    })

# -----------------------
# 6. Final Report
# -----------------------
accuracy = total_em / count if count else 0
mean_anls = total_anls / count if count else 0

print("\n===== FINAL RESULTS =====")
print(f"Accuracy (Exact Match): {accuracy:.4f}")
print(f"ANLS Score: {mean_anls:.4f}")

with open("mp_docvqa_results_high_res.json", "w") as f:
    json.dump({"accuracy": accuracy, "anls": mean_anls, "results": results}, f, indent=2)