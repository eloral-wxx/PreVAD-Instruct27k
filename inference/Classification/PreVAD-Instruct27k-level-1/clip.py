import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import json
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from decord import VideoReader, cpu
from sklearn.metrics import accuracy_score, f1_score

class_names = [
    "Normal",
    "Daily Accident",
    "Violence",
    "Production Accident",
    "Vehicle Accident",
    "Fire-related Accident",
    "Robbery",
    "Animal-related Violence"
]

test_data_path = build_data_path('PreVAD-Instruct27k', 'filter_test.json')  
video_path_prefix = build_data_path('PreVAD-Instruct27k')    
output_file = build_output_path('inference', 'Classification', 'PreVAD-Instruct27k-level-1', 'clip-1.txt')     
num_frames = 8              
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(test_data_path, "r") as f:
    data = json.load(f)

def sample_frames(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = [Image.fromarray(vr[i].asnumpy()) for i in indices]
    return frames

MODEL_ID = 'openai/clip-vit-base-patch32'
cache_path = HF_CACHE_DIR 
model = CLIPModel.from_pretrained(MODEL_ID,cache_dir=cache_path).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=cache_path)


all_preds = []
all_labels = []
for item in data:
    video_path = os.path.join(video_path_prefix, item['video_path'])
    label = item["superclass_name"]
    
    frames = sample_frames(video_path, num_frames=num_frames)
    
    # 图像编码
    inputs = processor(images=frames, return_tensors="pt").to(device)
    image_embeds = model.get_image_features(**inputs)
    image_embeds = F.normalize(image_embeds, dim=-1)
    
    # 文本编码
    if not class_names:
        class_names.append(label)  
    text_inputs = processor(text=class_names, return_tensors="pt", padding=True).to(device)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # 相似度计算
    sim = image_embeds @ text_embeds.T       
    avg_sim = sim.mean(dim=0)                
    pred_idx = avg_sim.argmax().item()
    pred_label = class_names[pred_idx]
    
    all_preds.append(pred_label)
    all_labels.append(label)
    
    print(f"Video: {video_path} | True: {label} | Pred: {pred_label}")

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"\nOverall Accuracy: {acc:.4f}")
print(f"Overall F1-score: {f1:.4f}")

with open(output_file, 'w') as f:
    f.write("真实标签,预测标签\n")
    for true, pred in zip(all_labels, all_preds):
        f.write(f"{true},{pred}\n")
    f.write("\n评估指标:\n")
    f.write(f"准确率 (Accuracy): {acc:.4f}\n")
    f.write(f"F1 分数 (Weighted): {f1:.4f}\n")
print(f"\n结果已保存至: {output_file}")