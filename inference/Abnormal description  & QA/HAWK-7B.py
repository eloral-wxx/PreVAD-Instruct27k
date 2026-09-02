import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from path_config import data_path as build_data_path, external_path as build_external_path, output_path as build_output_path, HF_CACHE_DIR
import warnings
warnings.filterwarnings("ignore")
import json
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

video_prefix = build_data_path('PreVAD-Instruct27k')
json_file = build_data_path('PreVAD-Instruct27k', 'filter_test.json')
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, item in enumerate(data):
    video_path = os.path.join(video_prefix, item["video_path"])

    # QA TASK
    question = item['question'][0]
    # Abnormal description TASK
    question = "Briefly describe the anomalies occurring in the video."

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
    item["hawk_answer"] = llm_message
    print(f"进度{i}, 回答:{llm_message}")

# Abnormal description TASK
output_file = build_output_path('metrics', 'model-answers', 'hawk_des.json') 
# QA TASK
output_file = build_output_path('metrics', 'model-answers', 'hawk_qa.json') 

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Results with model answers saved to {output_file}")

