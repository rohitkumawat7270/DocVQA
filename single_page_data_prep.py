import os
import json
from datasets import load_dataset
from tqdm import tqdm

# -----------------------
# 1. Configuration
# -----------------------
IMAGE_DIR = "documents"
JSON_OUT = "val.json"

# -----------------------
# 2. Downloading Dataset
# -----------------------
print("Downloading FULL Single-Page DocVQA dataset from Hugging Face...")
# 'lmms-lab/DocVQA' is the official dataset repo
dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

os.makedirs(IMAGE_DIR, exist_ok=True)
formatted_data = []

# -----------------------
# 3. Extracting and Saving Data
# -----------------------
print(f"Extracting {len(dataset)} images and saving locally...")

for i, row in enumerate(tqdm(dataset)):
    # Define exact path for the image
    image_path = os.path.join(IMAGE_DIR, f"val_image_{i}.png")
    
    # Extract and save the PIL image from dataset to hard drive
    if not os.path.exists(image_path):
        row["image"].save(image_path)
        
    # Keep only the data we need for VQA
    formatted_data.append({
        "questionId": row.get("questionId", i),
        "question": row["question"],
        "question_types": row.get("question_types", ["layout"]),
        "image": image_path,
        "answers": row["answers"]
    })

# -----------------------
# 4. Saving JSON
# -----------------------
with open(JSON_OUT, "w") as f:
    json.dump({"data": formatted_data}, f, indent=4)

print(f"\n✅ Single-Page Data preparation complete! {len(formatted_data)} records saved in '{JSON_OUT}'.")
print(f"Images are saved in the '{IMAGE_DIR}' folder.")