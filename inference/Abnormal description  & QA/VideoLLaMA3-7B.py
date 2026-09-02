import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings
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

video_path_prefix = build_data_path('PreVAD-Instruct27k')
input_json_file = build_data_path('PreVAD-Instruct27k', 'filter_test.json')

with open(input_json_file, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

jindu = 0

for item in data_list:
    jindu += 1
    video_path = os.path.join(video_path_prefix, item['video_path'])

    # QA TASK
    question = item['question'][0]
    # Abnormal description TASK
    question = "Briefly describe the anomalies occurring in the video."

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
    output_ids = model.generate(**inputs,  max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    videollama3_answer = response
    print(f"进度:{jindu}，回复：{videollama3_answer}")
    item['videollama3_answer'] = videollama3_answer
    del inputs
    del output_ids
    torch.cuda.empty_cache()

# Abnormal description TASK
output_json_file = build_output_path('metrics', 'model-answers', 'videollama3_des.json') 
# QA TASK
output_json_file = 'metrics/model-answers/videollama3_qa.json'

with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)

print(f"Updated data has been written to {output_json_file}")
