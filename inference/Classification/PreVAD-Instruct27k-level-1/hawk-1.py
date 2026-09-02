import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import warnings
warnings.filterwarnings("ignore")
import json
from sklearn.metrics import accuracy_score, f1_score
import sys
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

categories = [
    "Normal", "Daily Accident", "Violence", "Production Accident",
    "Vehicle Accident", "Fire-related Accident", "Robbery", "Animal-related Violence"
]

test_data_path = build_data_path('PreVAD-Instruct27k', 'filter_test.json')  
video_path_prefix = build_data_path('PreVAD-Instruct27k')    
output_file = build_output_path('inference', 'Classification', 'PreVAD-Instruct27k-level-1', 'hawk-1.txt')   
with open(test_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

true_labels = []
predicted_labels = []

# ---------------- 视频推理循环 ----------------
for i, item in enumerate(data):
    video_path = os.path.join(video_path_prefix, item["video_path"])
    question = item['question'][0]

    # 初始化聊天状态
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

    # 关键词匹配提取类别
    predicted = "Unknown"
    for cat in categories:
        if cat.lower() in llm_message.lower():
            predicted = cat
            break

    print(f"[{i+1}/{len(data)}] Pred: {predicted}, GT: {item['superclass_name']}")
    true_labels.append(item["superclass_name"])
    predicted_labels.append(predicted)

# ---------------- 计算指标 ----------------
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average="weighted")
print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# ---------------- 保存结果 ----------------
result_file = "hawk_classification_results.txt"
with open(result_file, "w", encoding="utf-8") as f:
    for vid, gt, pred in zip(data, true_labels, predicted_labels):
        f.write(f"{vid['video_path']}\t{gt}\t{pred}\n")
    f.write(f"\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\n")

print(f"Results saved to {result_file}")
