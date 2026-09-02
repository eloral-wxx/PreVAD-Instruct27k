import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
cache_path = HF_CACHE_DIR 
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_path
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_path)

video_path_prefix = build_data_path('PreVAD-Instruct27k')
input_json_file = build_data_path('PreVAD-Instruct27k', 'filter_test.json')

with open(input_json_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
jindu = 0
# 遍历每条数据，提取question并生成回答
for item in data_list:
    jindu += 1
    
    # QA TASK
    question = item['question'][0]
    # Abnormal description TASK
    question = "Briefly describe the anomalies occurring in the video."
    
    video_path = os.path.join(video_path_prefix, item['video_path'])
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

    # 模型推理
    output_ids = model.generate(**inputs, max_new_tokens=128)
    model_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # 提取assistant之后的内容
    parts = model_response.split('assistant\n')
    if len(parts) > 1:
        qwen_answer = parts[1].strip()
    else:
        qwen_answer = model_response.strip()
    print(f"进度:{jindu}，回复：{qwen_answer}")
    # 将回答保存到原数据中
    item['qwen_answer'] = qwen_answer

# Abnormal description TASK
output_json_file = 'metrics/model-answers/qwen_des.json' 
# QA TASK
output_json_file = 'metrics/model-answers/qwen_qa.json' 

with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"Updated data has been written to {output_json_file}")