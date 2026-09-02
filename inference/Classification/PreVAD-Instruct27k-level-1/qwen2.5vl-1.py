import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from sklearn.metrics import accuracy_score, f1_score
import random

superclass_keywords = [
    "Normal",
    "Daily Accident",
    "Violence",
    "Production Accident",
    "Vehicle Accident",
    "Fire-related Accident",
    "Robbery",
    "Animal-related Violence"
]

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
cache_path = HF_CACHE_DIR 
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_path
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_path)



test_data_path = build_data_path('PreVAD-Instruct27k', 'filter_test.json')  
video_path_prefix = build_data_path('PreVAD-Instruct27k')    
output_file = build_output_path('inference', 'Classification', 'PreVAD-Instruct27k-level-1', 'qwen2.5vl-1.txt') 
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)



predicted_labels =   []
true_labels = []
jindu = 0

for item in test_data:
    jindu += 1
    video_path = os.path.join(video_path_prefix, item['video_path'])
    random.shuffle(superclass_keywords)
    question = f"Based on the video, classify the event into one of the following categories: {', '.join(superclass_keywords)}. Please provide the most likely category."

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
    for keyword in superclass_keywords:
        if keyword.lower() in response.lower():
            predicted_superclass = keyword
            break

    if predicted_superclass is None:
        predicted_superclass = "Unknown"  

    print(f"进度：{jindu}, predict:{predicted_superclass}, real:{item['superclass_name']}")
    # 获取真实标签
    true_superclass = item['superclass_name']

    # 存储真实标签和预测标签
    true_labels.append(true_superclass)
    predicted_labels.append(predicted_superclass)

# 计算准确率和F1分数
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

with open(output_file, 'w') as file:
    file.write("True Labels:\n")
    file.write('\n'.join(true_labels) + '\n\n')
    file.write("Predicted Labels:\n")
    file.write('\n'.join(predicted_labels) + '\n\n')
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")

print(f"\n结果已保存至: {output_file}")