import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import av
import torch
import numpy as np
import json
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score
import re
import random

class_keywords = ['Vandalism', 'Arrest', 'Stealing', 'Abuse',
                 'Shooting', 'Fighting', 'Normal', 'Burglary', 'RoadAccidents', 'Shoplifting', 
                 'Assault', 'Robbery', 'Explosion', 'Arson']

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

MODEL_ID = 'LanguageBind/Video-LLaVA-7B-hf'
cache_path = HF_CACHE_DIR 

model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_path
)
processor = VideoLlavaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_path)

test_data_path = build_data_path('UCF-Crime', 'ucf-crime_test_anno.json') 
video_path_prefix = build_data_path('UCF-Crime', 'videos', 'test')   
output_file = build_output_path('inference', 'Classification', 'UCF-Crime', 'videollava-5.txt')  
  
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)


true_labels = []
predicted_labels = []
jindu = 0
for item in test_data:
    jindu += 1
    video_path  = item['video_path']
    random.shuffle(class_keywords)
    prompt = f"""USER: <video>Identify the event type from {', '.join(class_keywords)}  \nASSISTANT:"""

    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames // 8).astype(int)
    clip = read_video_pyav(container, indices)
    inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_length=4096)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    assistant_response = response.split("ASSISTANT:")[1].strip() 
    print(assistant_response)
    predicted_class = None
    for keyword in class_keywords:
        if keyword.lower() in assistant_response.lower():
            predicted_class = keyword
            break

    if predicted_class is None:
        predicted_class = "Unknown"
    print(f"进度：{jindu}, predict:{predicted_class}, real:{item['class_name']}")

    true_labels.append(item['class_name'])
    predicted_labels.append(predicted_class)

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