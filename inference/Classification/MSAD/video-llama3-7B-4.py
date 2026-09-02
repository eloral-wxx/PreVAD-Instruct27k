import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from sklearn.metrics import accuracy_score, f1_score
import random
import warnings
import re
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_ID = 'DAMO-NLP-SG/VideoLLaMA3-7B'
cache_path = HF_CACHE_DIR 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir=cache_path,
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_path)

test_data_path = build_data_path('MSAD', 'msad_test_anno.json')  
video_path_prefix = build_data_path('MSAD', 'other_datasets', 'msad_videos')    
output_file = build_output_path('inference', 'Classification', 'MSAD', 'videollama3-4.txt')

with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

class_keywords = ['Assault', 'Explosion', 'Fighting', 'Fire', 'Object_falling', 'People_falling', 'Robbery', 'Shooting', 'Traffic_accident', 'Vandalism', 'Water_incident', 'Normal']

true_labels = []
predicted_labels = []
jindu = 0
for item in data:
    jindu += 1
    file_name = re.search(r"[^/]+$", item['video_path']).group() 
    video_path = os.path.join(video_path_prefix, file_name)
    random.shuffle(class_keywords)
    question = "Based on the video, classify the event into one of the following categories: " + ", ".join(class_keywords) + ". Please provide the most likely category."
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 8}},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"response:{response}, true:{ item['class_name']}")

    predicted_superclass = None
    for keyword in class_keywords:
        if keyword.lower() in response.lower():
            predicted_superclass = keyword
            break

    if predicted_superclass is None:
        predicted_superclass = "Unknown"  

    true_superclass = item['class_name']

    true_labels.append(true_superclass)
    predicted_labels.append(predicted_superclass)
    del inputs
    del output_ids
    torch.cuda.empty_cache()
    print(f"进度{jindu}，pred:{predicted_superclass}, true:{ item['class_name']}")

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

with open(output_file, 'w') as f:
    f.write("真实标签,预测标签\n")
    for true, pred in zip(true_labels, predicted_labels):
        f.write(f"{true},{pred}\n")
    f.write("\n评估指标:\n")
    f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
    f.write(f"F1 分数 (Weighted): {f1:.4f}\n")

print(f"\n结果已保存至: {output_file}")
