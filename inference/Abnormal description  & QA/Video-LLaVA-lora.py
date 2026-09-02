import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import av
import torch
import numpy as np
import json
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from peft import PeftModel

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

video_path_prefix = build_data_path('PreVAD-Instruct27k')
test_data_path = build_data_path('PreVAD-Instruct27k', 'filter_test.json')

with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

jindu = 0
for item in test_data:
    jindu += 1
    video_path = os.path.join(video_path_prefix, item['video_path']) 
    # QA TASK
    question = item['question'][0]
    prompt = f"""USER: <video>{question}\nASSISTANT:"""
    # Abnormal description TASK
    prompt = """USER: <video>Briefly describe the anomalies occurring in the video. 
                      ASSISTANT:"""

    # 读取视频帧
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames // 8).astype(int)
    clip = read_video_pyav(container, indices)
    inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
    generate_ids = model.generate(
    **inputs,
    max_new_tokens=128,        
    temperature=0.7,            
    top_p=0.9,                 
    repetition_penalty=1.2,      
    no_repeat_ngram_size=3,      
    length_penalty=1.0           
)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    assistant_response = response.split("ASSISTANT:")[1].strip()
    item['our_answer'] = assistant_response
    print(f"进度:{jindu}") 
    print(assistant_response)

# Abnormal description TASK
output_json_file = build_output_path('metrics', 'model-answers', 'videollava_des.json') 
# QA TASK
output_json_file = build_output_path('metrics', 'model-answers', 'videollava_qa.json') 

with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"Updated data has been written to {output_json_file}")