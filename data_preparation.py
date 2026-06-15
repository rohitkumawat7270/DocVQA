import os
import json
import ast
from datasets import load_dataset
from tqdm import tqdm

# Configuration
IMAGE_DIR = "multidoc/images"
JSON_OUT = "multi_val.json"

print("Downloading FULL MP-DocVQA dataset from Hugging Face...")
dataset = load_dataset("lmms-lab/MP-DocVQA", split="val")

os.makedirs(IMAGE_DIR, exist_ok=True)
formatted_data = []

print("Extracting pages and saving images locally (Full Dataset)...")
# Removed limit: processing the entire validation dataset for the server
for i, row in enumerate(tqdm(dataset)):
    
    # Safely parse answers string array into actual python lists
    answers = ast.literal_eval(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
    page_ids = []
    
    # Loop through the 20 possible image columns
    for page_num in range(1, 21):
        image_key = f"image_{page_num}"
        
        if image_key in row and row[image_key] is not None:
            page_id = f"doc_{i}_page_{page_num}"
            page_ids.append(page_id)
            
            # Save image if it doesn't already exist
            img_path = os.path.join(IMAGE_DIR, f"{page_id}.jpg")
            if not os.path.exists(img_path):
                row[image_key].save(img_path)
    
    formatted_data.append({
        "questionId": row.get("questionId", str(i)),
        "question": row["question"],
        "answers": answers,
        "page_ids": page_ids
    })

with open(JSON_OUT, "w") as f:
    json.dump({"data": formatted_data}, f, indent=4)

print("\nData preparation complete! Run multi_page_inference.py next.")