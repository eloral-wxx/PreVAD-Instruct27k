import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import torch
import numpy as np
import json
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score
import re
import random
from decord import VideoReader

class_keywords = ['Assault', 'Explosion', 'Fighting', 'Fire', 'Object_falling', 'People_falling', 'Robbery', 'Shooting', 'Traffic_accident', 'Vandalism', 'Water_incident', 'Normal']
def read_video_videoreader(video_path, indices):
    vreader = VideoReader(video_path)
    frames = vreader.get_batch(indices).asnumpy()  
    return frames

MODEL_ID = 'LanguageBind/Video-LLaVA-7B-hf'
cache_path = HF_CACHE_DIR 

model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_path
)
processor = VideoLlavaProcessor.from_pretrained(MODEL_ID, cache_dir=cache_path)

test_data_path = build_data_path('MSAD', 'msad_test_anno.json')  
video_path_prefix = build_data_path('MSAD', 'other_datasets', 'msad_videos')    
output_file = build_output_path('inference', 'Classification', 'MSAD', 'videollava-4.txt')
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)


true_labels = []
predicted_labels = []
jindu = 0
for item in test_data:
    jindu += 1
    file_name = re.search(r"[^/]+$", item['video_path']).group() 
    video_path = os.path.join(video_path_prefix, file_name)
    random.shuffle(class_keywords)
    prompt = f"""USER: <video>Identify the event type from {', '.join(class_keywords)}  \nASSISTANT:"""

    # 读取视频帧
    vreader = VideoReader(video_path)
    total_frames = len(vreader)
    indices = np.arange(0, total_frames, total_frames // 8).astype(int)
    clip = read_video_videoreader(video_path, indices)
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
