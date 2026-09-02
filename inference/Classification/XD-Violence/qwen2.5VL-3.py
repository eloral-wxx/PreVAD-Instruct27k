import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from sklearn.metrics import accuracy_score, f1_score
import random

class_keywords = ["Normal","Fighting", "Shooting", "Riot", "Explosion", "Car accident","Abuse"]

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
cache_path = HF_CACHE_DIR 
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_path
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_path)

test_data_path = build_data_path('xd-violence', 'xd_test_anno.json')  
video_path_prefix = build_data_path('xd-violence', 'other_datasets', 'xd_videos') 
output_file = build_output_path('inference', 'Classification', 'XD-Violence', 'qwen2.5VL-3.txt') 

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

predicted_labels =   []
true_labels = []
jindu = 0

for item in test_data:
    jindu += 1
    file_name = re.search(r"[^/]+$", item['video_path']).group() 
    video_path = os.path.join(video_path_prefix, file_name)

    # if file_name =='Mission.Impossible.Fallout.2018__#00-31-21_00-32-50_label_B1-0-0.mp4':
    #     continue
    # if file_name =='Mission.Impossible.Fallout.2018__#00-39-18_00-40-36_label_B1-0-0.mp4':
    #     continue
    # if file_name =='Mission.Impossible.Fallout.2018__#02-03-50_02-04-35_label_B1-0-0.mp4':
    #     continue
    # if jindu == 639:
    #     video_path = "./videos/Lord.of.War__#00-50-10_00-50-50_label_G-0-0.mp4"

    random.shuffle(class_keywords)
    question = f"Based on the video, classify the event into one of the following categories: {', '.join(class_keywords)}. Please provide the most likely category."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 224 * 224,
                    "fps": 1.0,
                    "max_frames": 8
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(    
        text=[text],
        images=image_inputs, 
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    output_ids = model.generate(**inputs, max_new_tokens=128)
    model_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    parts = model_response.split('assistant\n')
    if len(parts) > 1:
        response = parts[1].strip()
    else:
        response = response.strip()

    predicted_superclass = None
    for keyword in class_keywords:
        if keyword.lower() in response.lower():
            predicted_superclass = keyword
            break

    if predicted_superclass is None:
        predicted_superclass = "Unknown"  
    print(f"进度：{jindu}, predict:{predicted_superclass}, real:{item['class_name']}")

    true_superclass = item['class_name']

    true_labels.append(true_superclass)
    predicted_labels.append(predicted_superclass)

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
