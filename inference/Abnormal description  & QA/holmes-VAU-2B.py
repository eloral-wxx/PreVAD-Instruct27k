import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import json
import torch
import warnings
from holmesvau.holmesvau_utils import load_model, generate

warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

MODEL_ID = 'ppxin321/HolmesVAU-2B'
cache_path = HF_CACHE_DIR 
mllm_path = os.path.join(HF_CACHE_DIR, 'HolmesVAU-2B')
sampler_path = build_external_path('HolmesVAU', 'holmesvau', 'ATS', 'anomaly_scorer.pth')

video_root = build_data_path('PreVAD-Instruct27k')
json_path = build_data_path('PreVAD-Instruct27k', 'filter_test.json')

# Abnormal description TASK
output_file = build_output_path('metrics', 'model-answers', 'holmesVAU-des.json')
# QA TASK
output_file = build_output_path('metrics', 'model-answers', 'holmesVAU-qa.json')


# 加载模型
model, tokenizer, generation_config, sampler = load_model(mllm_path, sampler_path, device)

# 读取 JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 循环处理每个条目
for idx, item in enumerate(data):
    rel_path = item.get("video_path")

    # QA TASK
    prompt = item['question'][0]
    # Abnormal description TASK
    prompt = "Briefly describe the anomalies occurring in the video."

    if not rel_path:
        item["holmesvau_answer"] = None
        continue

    video_path = os.path.join(video_root, rel_path)
    if not os.path.exists(video_path):
        item["holmesvau_answer"] = None
        print(f"[{idx}] 视频不存在: {video_path}")
        continue

    try:
        answer, _, _, _ = generate(
            video_path=video_path,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            sampler=sampler,
            select_frames=8,
            use_ATS=True
        )
        item["holmesvau_answer"] = answer
        print(f"进度:{idx}，回复：{answer}")

    except FileNotFoundError:
        item["holmesvau_answer"] = None
        print(f"[{idx}] 文件未找到: {video_path}")
    except (RuntimeError, ValueError) as e:
        item["holmesvau_answer"] = f"视频处理错误: {str(e)}"
        print(f"[{idx}] 处理错误: {video_path} - {e}")
    except torch.cuda.OutOfMemoryError:
        item["holmesvau_answer"] = "显存不足"
        torch.cuda.empty_cache()
        print(f"[{idx}] 显存不足: {video_path}")

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"处理完成，结果已保存到 {output_file}")
