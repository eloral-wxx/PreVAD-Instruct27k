import os
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import torch
import json
from holmesvau.holmesvau_utils import load_model, generate
from sklearn.metrics import accuracy_score, f1_score
import warnings
import re
import random
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

class_keywords = [ "Normal","Fighting", "Shooting", "Riot", "Explosion", "Car accident","Abuse"]

MODEL_ID = 'ppxin321/HolmesVAU-2B'
cache_path = HF_CACHE_DIR 
mllm_path = os.path.join(HF_CACHE_DIR, 'HolmesVAU-2B')
sampler_path = build_external_path('HolmesVAU', 'holmesvau', 'ATS', 'anomaly_scorer.pth')
model, tokenizer, generation_config, sampler = load_model(mllm_path, sampler_path, device)

test_data_path = build_data_path('xd-violence', 'xd_test_anno.json')  
video_path_prefix = build_data_path('xd-violence', 'other_datasets', 'xd_videos') 
output_file = build_output_path('inference', 'Classification', 'XD-Violence', 'holmes-vau-3.txt')                         


with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

true_labels = []
predicted_labels = []

progress = 0
total_videos = len(test_data)
for item in test_data:
    progress += 1
    try:
        video_file = re.search(r"[^/]+$", item['video_path']).group()
        video_path = os.path.join(video_path_prefix, video_file)
        random.shuffle(class_keywords)
        prompt = ("Based on the video, classify the event into one of the following categories: " 
                 + ", ".join(class_keywords) + ". Please provide the most likely category.")
        
        pred, _, _, _ = generate(
            video_path=video_path,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            sampler=sampler,
            select_frames=8,
            use_ATS=True
        )
        
        predicted_class = None
        for keyword in class_keywords:
            if keyword.lower() in pred.lower():
                predicted_class = keyword
                break
        
        if predicted_class is None:
            predicted_class = "Unknown"
        
        true_class = item['class_name']
        
        true_labels.append(true_class)
        predicted_labels.append(predicted_class)
        
        print(f"进度: {progress}/{total_videos} | 视频: {video_file}")
        print(f"预测: {predicted_class} | 真实: {true_class}")
        print(f"模型输出: {pred}\n")
        
        torch.cuda.empty_cache()
        
    except FileNotFoundError:
        print(f"视频文件不存在: {video_path}")
    except (RuntimeError, ValueError) as e:
        print(f"视频处理错误: {str(e)}")
    except torch.cuda.OutOfMemoryError:
        print(f"显存不足，跳过视频: {video_path}")

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print("\n评估结果:")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"F1 分数 (Weighted): {f1:.4f}")

with open(output_file, 'w') as f:
    f.write("真实标签,预测标签\n")
    for true, pred in zip(true_labels, predicted_labels):
        f.write(f"{true},{pred}\n")
    f.write("\n评估指标:\n")
    f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
    f.write(f"F1 分数 (Weighted): {f1:.4f}\n")

print(f"\n结果已保存至: {output_file}")