import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import warnings
warnings.filterwarnings("ignore")
import json
from sklearn.metrics import accuracy_score, f1_score
import sys
import random
import re
from types import SimpleNamespace

sys.path.append(build_external_path('hawk'))

from hawk.common.config import Config
from hawk.common.registry import registry
from hawk.conversation.conversation_video import Chat, conv_llava_llama_2

args = SimpleNamespace()
args.cfg_path = build_external_path('hawk', 'configs', 'eval_configs', 'eval.yaml')
args.options = []

cfg = Config(args)
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda:0")
model.eval()

vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

chat = Chat(model, vis_processor, device="cuda:0")

class_keywords = [ "Normal","Fighting", "Shooting", "Riot", "Explosion", "Car accident","Abuse"]

test_data_path = build_data_path('xd-violence', 'xd_test_anno.json')  
video_path_prefix = build_data_path('xd-violence', 'other_datasets', 'xd_videos') 
output_file = build_output_path('inference', 'Classification', 'XD-Violence', 'hawk-3.txt') 

with open(test_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

true_labels = []
predicted_labels = []


for i, item in enumerate(data):
    file_name = re.search(r"[^/]+$", item['video_path']).group() 
    video_path = video_prefix + file_name
    if file_name == 'Lord.of.War__#00-50-10_00-50-50_label_G-0-0.mp4':
        continue

    random.shuffle(class_keywords)
    question = (
        f"Classify this video into ONE category from the following list: {', '.join(class_keywords)}.\n"
        f"Output ONLY the category name, no other words."
    )

    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "You are able to understand the visual content provided."
    img_list = []
    chat.upload_video_without_audio(video_path, chat_state, img_list)
    chat.ask(question, chat_state)

    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=0.7,
        max_new_tokens=128, 
    )[0]
    print(f"Raw output: {llm_message}")

    predicted = "Unknown"
    for cat in class_keywords:
        if re.search(rf"\b{re.escape(cat.lower())}\b", llm_message.lower()):
            predicted = cat
            break

    print(f"[{i+1}/{len(data)}] Pred: {predicted}, GT: {item['class_name']}")
    true_labels.append(item["class_name"])
    predicted_labels.append(predicted)


accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average="weighted")
print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

with open(output_file, 'w') as f:
    f.write("真实标签,预测标签\n")
    for true, pred in zip(true_labels, predicted_labels):
        f.write(f"{true},{pred}\n")
    f.write("\n评估指标:\n")
    f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
    f.write(f"F1 分数 (Weighted): {f1:.4f}\n")

print(f"\n结果已保存至: {output_file}")
