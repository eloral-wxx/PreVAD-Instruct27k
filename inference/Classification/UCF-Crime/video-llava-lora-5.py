import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import av
import torch
import numpy as np
import json
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
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
LORA_WEIGHTS_PATH = ''

cache_path = HF_CACHE_DIR 
base_model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_path
)

model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH,cache_dir=cache_path)
model = model.merge_and_unload()
processor = VideoLlavaProcessor.from_pretrained(MODEL_ID,cache_dir=cache_path)

test_data_path = build_data_path('UCF-Crime', 'ucf-crime_test_anno.json') 
video_path_prefix = build_data_path('UCF-Crime', 'videos', 'test')   
output_file = build_output_path('inference', 'Classification', 'UCF-Crime', 'videollava-lora-5.txt')    

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)


true_labels = []
predicted_labels = []
jindu = 0
for item in test_data:
    jindu += 1
    random.shuffle(class_keywords)
    video_path  = item['video_path']
    prompt = f"""USER:<video>Based on the video, classify the video into one of the following categories: {', '.join(class_keywords)}. Please provide the most likely category.\nASSISTANT:"""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames // 8).astype(int)
    clip = read_video_pyav(container, indices)
    inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_length=2800)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    assistant_response = response.split("ASSISTANT:")[1].strip()

    predicted_superclass = None
    for keyword in class_keywords:
        if keyword.lower() in assistant_response.lower():
            predicted_superclass = keyword
            break
    if predicted_superclass is None:
        predicted_superclass = "Unknown"  

    print(f"进度：{jindu}, predict:{predicted_superclass}, real:{item['class_name']}, video_path:{video_path}")
    true_labels.append(item['class_name'])
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