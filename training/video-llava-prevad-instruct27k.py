import os
# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys  # 提供对 Python 解释器的访问
import json  # 用于处理 JSON 数据
import av  # 用于视频处理
import re  # 正则表达式模块
import bisect  # 用于有序列表的二分查找和插入
import numpy as np  # 数学计算库
import wandb  # 用于实验跟踪和可视化
import datetime  # 日期和时间处理
import cv2  # OpenCV 库，用于图像和视频处理
import pandas as pd
import random 

from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor
# BitsAndBytesConfig: 用于量化配置
# VideoLlavaForConditionalGeneration: Video-LLaVA 模型的model导入
# VideoLlavaProcessor: 数据处理工具

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# LoraConfig: LoRA（Low-Rank Adaptation）配置
# prepare_model_for_kbit_training: 准备模型进行量化训练
# get_peft_model: 获取 PEFT（Parameter-Efficient Fine-Tuning）模型

import torch  # PyTorch 深度学习框架
from torch.utils.data import Dataset  # 数据集类
from torch.utils.data import DataLoader  # 数据加载器
from datasets import load_dataset, concatenate_datasets, load_from_disk
from datasets import Dataset as HF_Dataset # 导入huggingface的dataset，起别名用于与torch的Dataset作区分
# load_dataset: 加载 Hugging Face 数据集
# concatenate_datasets: 合并多个数据集
# load_from_disk: 从磁盘加载数据集

import lightning as L  # PyTorch Lightning，用于简化训练流程
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # 提前停止回调
from lightning.pytorch.callbacks import Callback  # 回调基类
from lightning.pytorch.profilers import SimpleProfiler  # 性能分析器
from lightning.pytorch.loggers import WandbLogger
import flash_attn
from rouge_score import rouge_scorer
from path_config import HF_CACHE_DIR, data_path, output_path

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

# 定义全局变量
NUM_FRAMES_VIDEO = 8  # 每个视频采样的帧数
MAX_LENGTH_PROCESSOR = 4096  # 数据处理器的最大长度限制

MODEL_ID = 'LanguageBind/Video-LLaVA-7B-hf' # 用于lora微调的模型的huggingface名称
CACHE_PATH = HF_CACHE_DIR # 存储huggingface上下载的模型权重缓存的地方
VIDEO_ROOT = os.getenv("PREVAD_VIDEO_ROOT", data_path("PreVAD-Instruct27k"))

# 临时文件和模型快照的基础路径
LOCAL_PATH = output_path("training")


def resize_and_crop(img, target_length=224):
    """
    将输入图像按宽高比缩放至最长边为`target_size`，并居中裁剪为正方形。
    
    参数:
        img (numpy.ndarray): 输入图像（BGR格式，形状为`(H, W, C)`）。
        target_size (int): 目标正方形边长，默认为224。
        
    返回:
        numpy.ndarray: 处理后的正方形图像（形状为`(target_size, target_size, 3)`）。
        
    异常:
        ValueError: 输入图像无效或无法裁剪为正方形。
        cv2.error: 图像缩放失败。
    """
    # Step 1: 验证输入图像有效性
    img = img.to_ndarray(format="rgb24")  # 将图像转换为RGB格式的NumPy数组
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("输入图像不是一个有效的NumPy数组。")

    # 确保图像为uint8格式
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    height, width, channels = img.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"图像尺寸无效: {height}x{width}。")
    
    # 计算新的宽度，保持宽高比
    aspect_ratio = width / height

    # 选择原视频中较短的一条边为224，另一条边按照比例缩放
    if width < height:
        target_width = target_length
        target_height = int(target_width / aspect_ratio)
    else:
        target_height = target_length
        target_width = int(target_height * aspect_ratio)
    
    # 调整图像大小
    try:
        resized_img = cv2.resize(img, (target_width, target_height))
    except cv2.error as e:
        raise RuntimeError(f"调整图像大小时出错: {e}")
    
    if target_height == target_length:
        start_x = (target_width - target_height) // 2  # 计算裁剪的起始点
        cropped_img = resized_img[:, start_x:start_x + target_length]
    else:
        start_y = (target_height - target_width) // 2  # 计算裁剪的起始点
        cropped_img = resized_img[start_y:start_y + target_length, :]
    
    # Step 6: 强制验证输出形状
    assert cropped_img.shape == (target_length, target_length, 3), f"裁剪后尺寸错误: {cropped_img.shape}"
    return cropped_img

def read_equidistant_frames_pyav(video_path, num_frames):
    """
    从视频中均匀采样指定数量的帧。
    参数:
        video_path: 视频路径。
        num_frames: 需要采样的帧数。
    返回:
        采样后的帧数组。
    """
    container = av.open(video_path)  # 打开视频文件
    video = container.streams.get(0)[0]  # 获取视频流
    total_frames = int(container.streams.video[0].frames)
    # print(f"视频中的总帧数: {total_frames}")

    # 获取所有帧的时间戳
    av_timestamps = [
        int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
    ]
    av_timestamps.sort()  # 对时间戳排序

    # 确定起始和结束帧索引
    start_id = bisect.bisect_left(av_timestamps, 1)
    end_id = bisect.bisect_left(av_timestamps, 1e10)


    # 如果视频很短，扩展采样范围
    if end_id - start_id < 10:
        end_id += 10
        start_id -= 10
    
    # 保证采样范围在实际的视频帧范围内
    end_id = min(len(av_timestamps) - 1, end_id)
    start_id = max(1, start_id)

    min_id = end_id
    max_id = start_id

    # 保证采样范围在有效解码的视频帧的范围内
    container.seek(0)  # 重置视频流
    for i, frame in enumerate(container.decode(video=0)):
        min_id = min(min_id, i)
        max_id = max(max_id, i)
    
    start_id = max(start_id, min_id+1)
    end_id = min(end_id, max_id-1)

    indices = np.linspace(start_id, end_id, num_frames).astype(int)  # 均匀采样帧索引

    frames = []
    container.seek(0) # 重置视频流，要不然会因为重复解码视频而报错
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(resize_and_crop(frame))  # 调整大小并裁剪帧

    assert len(frames) == num_frames, f"实际获取的帧数为 {len(frames)}，但应为 {num_frames},采样失败的视频名为{video_path}。"
    return np.stack(frames)

def get_frames_for_video(path, num_frames):
    """
    从完整视频中均匀地获取指定数量的帧。
    参数:
        path: 视频路径。
        num_frames: 需要的帧数。
    返回:
        采样后的帧数组。
    """
    # 调用函数读取视频帧
    frames = read_equidistant_frames_pyav(path, num_frames)
    
    return frames

def collate_read_video(example, num_frames):
    """
    从视频中读取帧，并缓存结果。
    参数:
        example: 包含视频信息的字典。
        num_frames: 需要的帧数。
    返回:
        包含视频帧的字典。
    """
    try:
        if example['clip'] is None:
            video_path = os.path.join(VIDEO_ROOT, example['video_path'])
            clip = get_frames_for_video(video_path, num_frames)
            example['clip'] = clip
        return example
    except Exception as e:
        # 记录错误视频路径和原因
        print(f"Error processing video {example['video_path']}: {str(e)}")
        # 返回空数据或标记为无效（后续可过滤）
        example['clip'] = None  # 或设置为占位符
        return example
    

# 加载训练集
file_path = data_path('PreVAD-Instruct27k', 'filter_test.json')
with open(file_path, 'r', encoding='utf-8') as file:
    dataset_list = json.load(file)
print(len(dataset_list))
# 将列表转换为 Dataset 对象
dataset = HF_Dataset.from_pandas(pd.DataFrame(dataset_list))

num_processes = 8  # 设置多进程处理的数量
# 将数据集分块处理
num_blocks = 50  # 设置分块数量
# 计算每个块的大小
block_size = len(dataset) // num_blocks
print(block_size)
remainder = len(dataset) % num_blocks  # 计算余数

def add_clip_key(example):
    example['clip'] = None
    return example
# 使用 map 函数应用这个处理
dataset = dataset.map(add_clip_key)


# 遍历每个块
for i in range(num_blocks):
    start_idx = i * block_size  # 当前块的起始索引
    end_idx = start_idx + block_size  # 当前块的结束索引

    # 对于最后一个块，包含剩余的所有样本
    if i == num_blocks - 1:
        end_idx = len(dataset)

    # 选择当前块的数据
    print(f"Selecting between {start_idx}-{end_idx}")
    save_directory = LOCAL_PATH + f'/train_dataset_small/shard_{i}/'  # 设置保存路径

    # 如果目录已存在，则跳过重新计算（防止程序崩溃后重复处理）
    if os.path.exists(save_directory):
        print("\t Folder for that shard found. Skipping recalculation")
        continue

    # 选择当前块的数据
    curr_shard = dataset.select(range(start_idx, end_idx))
    # 对当前块的数据进行处理（读取视频帧并缓存结果）
    curr_shard = curr_shard.map(
        collate_read_video, 
        batched=False,  # 定义是否批量处理，这里定义为false意味着collate_read_video函数会被单个调用
        fn_kwargs={"num_frames": NUM_FRAMES_VIDEO},  # 给collate_read_video传递查找表和帧数参数（每个视频中采样8帧）
        num_proc=num_processes,  # 使用多进程加速处理
        writer_batch_size=10  # 设置写入批次大小
    )

    # 将当前块的数据保存到磁盘
    curr_shard.save_to_disk(save_directory)
    print(f"Shard {i} - mapping complete")
    dataset.cleanup_cache_files()  # 清理缓存文件

# 重新加载存储的分区数据
num_blocks = 50  # 定义分区数量
sharded_dataset = []  # 创建一个空列表用于存储每个分区的数据集
for i in range(num_blocks):  # 遍历每个分区
    storage_directory = LOCAL_PATH + f'/train_dataset_small/shard_{i}/'
    # 构造每个分区的存储路径
    sharded_dataset.append(load_from_disk(storage_directory))
    # 使用 load_from_disk 函数从磁盘加载每个分区的数据集，并将其添加到列表中

dataset = concatenate_datasets(sharded_dataset)
print(f"load complete - len:{len(dataset)}")

# 从预训练模型加载 VideoLlavaProcessor
processor = VideoLlavaProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_PATH)
processor.tokenizer.padding_side = "right"  # 在训练期间，总是使用右侧填充
superclass_keywords = [
            "Normal",
            "Daily Accident",
            "Violence",
            "Production Accident",
            "Vehicle Accident",
            "Fire-related Accident",
            "Robbery",
            "Animal-related Violence"
        ]
class VideoLlavaDataset(Dataset):
    """
    PyTorch Dataset for VideoLlavaDataset. 
    这个类接收一个 HuggingFace Dataset 作为输入，用于处理视频和语言相关的数据。
    """

    def __init__(self, dataset: str):
        """
        初始化 VideoLlavaDataset 类。

        参数：
        - dataset (str): 输入的 HuggingFace Dataset。
        """
        super().__init__()
        self.dataset = dataset  # 存储输入的 Dataset

    def __len__(self) -> int:
        """
        返回数据集的长度（样本数量）。
        """
        return len(self.dataset)  # 返回输入 Dataset 的长度

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        clip = np.array(sample["clip"])
        task_type = sample['task_type']
        video_class = sample['superclass_name']
        if task_type == "qa":
            question = sample['question'][0]
            answer = sample['answer']
            prompt = f"USER: <video>Based on the video, answer the question: {question}\nASSISTANT: {answer}"
        elif task_type == "description":
            # 新prompt格式：要求先识别事件类型再描述
            description = sample['corrected_descriptions']
            random.shuffle(superclass_keywords)
            prompt = f"""USER: <video>Identify the event type from {', '.join(superclass_keywords)} and describe it.\n
                      ASSISTANT: Event type: {video_class}. Detailed description: {description}"""
        
        return prompt, clip, task_type

def train_collate_fn(examples):
    texts, videos, task_types = list(zip(*examples))
    batch = processor(
        text=texts, 
        videos=videos, 
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH_PROCESSOR,
        return_tensors="pt"
    )
    labels = batch["input_ids"].clone()
    
    # 掩码问题部分和填充符
    for i, text in enumerate(texts):
        question_part = text.split("ASSISTANT:")[0] + "ASSISTANT:"
        question_enc = processor(
        text=question_part, 
        videos=videos, 
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH_PROCESSOR,
        return_tensors="pt")
        q_len = question_enc["input_ids"].size(1)
        labels[i, :q_len] = -100  # 掩码问题部分
        labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100  # 掩码填充符
    batch["labels"] = labels
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values_videos"], labels,task_types

def eval_collate_fn(examples):
    texts, videos, task_types = list(zip(*examples))
    processed_texts = []
    answers = []
    for text in texts:
        processed_texts.append(text.split("ASSISTANT:")[0] + "ASSISTANT:") # 提取去掉答案后的prompt
        answers.append(text.split("ASSISTANT:")[1].strip()) # 提取真实答案
    batch = processor(
        text=processed_texts,
        videos=videos,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH_PROCESSOR,
        return_tensors="pt"
    )
    return (batch["input_ids"], batch["attention_mask"], batch["pixel_values_videos"],answers, task_types)

dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = VideoLlavaDataset(dataset["train"])
eval_dataset = VideoLlavaDataset(dataset["test"])

## 加载模型
# QLoRA: 使用 4 位量化，有助于在保持性能的同时减少内存使用。

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用 4 位量化
    bnb_4bit_quant_type="nf4",  # 使用 NormalFloat4 (NF4) 量化类型，适用于权重分布
    bnb_4bit_compute_dtype=torch.float16,  # 在计算时使用 16 位浮点精度
)

model = VideoLlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,  # 模型 ID
    torch_dtype=torch.float16,  # 指定模型加载时使用的数据类型为 16 位浮点
    quantization_config=bnb_config,  # 应用量化配置
    device_map= 'auto',  # 自动分配模型到可用设备（GPU/CPU）
    cache_dir=CACHE_PATH,  # 指定模型缓存目录
    attn_implementation="flash_attention_2"
)

def find_all_linear_names(model):
    """
    查找模型中所有线性层（nn.Linear）的名称，并排除与多模态投影和视觉模型相关的层。

    参数：
    - model: 输入的模型。

    返回：
    - list: 需要应用 LoRa 的线性层名称列表。
    """
    cls = torch.nn.Linear  # 定义目标层类型为 nn.Linear
    lora_module_names = set()  # 用于存储线性层的名称
    multimodal_keywords = ['multi_modal_projector', 'vision_model']  # 多模态相关层的关键词

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果模块名称包含多模态关键词，则跳过
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        # 如果模块是线性层，则记录其名称
        if isinstance(module, cls):
            names = name.split('.')  # 拆分模块名称
            # 添加线性层的名称（最后一级名称）
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 如果模型包含 'lm_head'，则移除（针对 16 位模型）
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    # print(list(lora_module_names)) # ['o_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'q_proj']
    return list(lora_module_names)  # 返回线性层名称列表

# 定义 LoRa 配置
lora_config = LoraConfig(
    r=8,  # 适配器的秩（rank）
    lora_alpha=8,  # LoRa 的缩放因子
    lora_dropout=0.1,  # LoRa 层的 dropout 概率
    target_modules=find_all_linear_names(model),  # 需要应用 LoRa 的目标模块
    init_lora_weights="gaussian",  # LoRa 权重的初始化方式（高斯初始化）
)

# 准备模型以支持 k-bit 训练（例如 4-bit 或 8-bit）
model = prepare_model_for_kbit_training(model)

# 将 LoRa 配置应用于模型，生成 PEFT 模型
model = get_peft_model(model, lora_config)

config = {
    "max_epochs": 3,  # 最大训练轮数
    "val_check_interval": 0.5,  # 在每个 epoch 中进行验证的频率
    "check_val_every_n_epoch": 1,  # 每隔多少个 epoch 进行一次验证
    "gradient_clip_val": 0.5,  # 梯度裁剪的阈值，防止梯度爆炸
    "accumulate_grad_batches": 8,  # 梯度累积的批次数量，用于放大有效 batch size
    "lr": 2e-5,  
    "batch_size": 1,  # 每个批次的样本数量
    "num_nodes": 1,  # 使用的节点数量（分布式训练时的机器数量）
    "warmup_steps": 500,  # 学习率预热的步数
}

class VideoLlavaModelPLModule(L.LightningModule):
    """
    PyTorch Lightning 模块，用于 Video-LLaVA 模型的训练和评估。
    """

    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")  # 获取批次大小

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, labels, task_types = batch
        print(task_types)
        # 前向计算
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values,
            labels=labels
        )
        
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            input_ids, attention_mask, pixel_values, true_answers, task_types = batch
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values,
                max_length=MAX_LENGTH_PROCESSOR,
                num_beams=1,
                do_sample=False
            )
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            qa_scores = []
            desc_scores = []
            total_scores = []

            for pred, true_answer, task_type in zip(generated_texts, true_answers, task_types):
                pred_answer = pred.split("ASSISTANT:")[1].strip()
                if task_type == "qa":
                    score = self.calculate_rouge(pred_answer, true_answer)
                    qa_scores.append(score)
                    total_scores.append(score)
                elif task_type == "description":
                    score = self.calculate_rouge(pred_answer, true_answer)
                    desc_scores.append(score)
                    total_scores.append(score)
            # 记录指标
            if qa_scores:
                self.log("val_qa_rouge", np.mean(qa_scores), prog_bar=True)
            if desc_scores:
                self.log("val_desc_rouge", np.mean(desc_scores), prog_bar=True)

        self.log("val_total_rouge", np.mean(total_scores), prog_bar=True)
        return  np.mean(total_scores)
        
            

    def calculate_rouge(self, prediction, reference):
        # 实现ROUGE计算逻辑（需要安装rouge-score包）
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure

    def configure_optimizers(self):
        """
        配置优化器。

        返回：
        - optimizer: 优化器实例。
        """
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        """
        训练数据加载器。

        返回：
        - DataLoader: 训练数据加载器。
        """
        return DataLoader(
            train_dataset,  # 训练数据集
            collate_fn=train_collate_fn,  # collate 函数
            batch_size=self.batch_size,  # 批次大小
            shuffle=True,  # 是否打乱数据
            num_workers=8,  # 数据加载器的线程数
            pin_memory=True,  # 加速数据到GPU的传输
            persistent_workers=True  # 保持worker进程活跃
        )

    def val_dataloader(self):
        """
        验证数据加载器。

        返回：
        - DataLoader: 验证数据加载器。
        """
        return DataLoader(
            eval_dataset,  # 验证数据集
            collate_fn=eval_collate_fn,  # collate 函数
            batch_size=self.batch_size,  # 批次大小
            shuffle=False,  # 是否打乱数据
            num_workers=8,  # 数据加载器的线程数
            pin_memory=True,  # 加速数据到GPU的传输
            persistent_workers=True  # 保持worker进程活跃
        )
    
# 实例化 VideoLlavaModelPLModule，传入配置、处理器和模型
model_module = VideoLlavaModelPLModule(config, processor, model)

# 定义早停回调
early_stop_callback = EarlyStopping(
    monitor="val_total_rouge", 
    patience=3,  
    verbose=False,  
    mode="max",
)

from datetime import datetime

class SaveModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # 只在主进程保存
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"{LOCAL_PATH}/weights/{current_time}-checkpoint-{trainer.current_epoch}"
            pl_module.model.save_pretrained(output_dir)
            print(f"Model checkpoint saved at epoch {trainer.current_epoch} to {output_dir}")

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:  # 只在主进程保存
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"{LOCAL_PATH}/weights/{current_time}-checkpoint-final"
            pl_module.model.save_pretrained(output_dir)
            print(f"Model checkpoint saved at the end of the training to {output_dir}")

wandb_logger = WandbLogger(
    project="video-llava-finetuning",  # 你的Wandb项目名称
    log_model="all",  # 记录模型检查点
    save_dir=LOCAL_PATH,  # 日志保存目录
    name="lora-finetune"  # 实验名称
)

trainer = L.Trainer(
    accelerator="gpu",  # 指定加速器类型为 GPU
    devices=1,  
    max_epochs=config.get("max_epochs"),  # 从配置中获取最大训练轮数
    accumulate_grad_batches=config.get("accumulate_grad_batches"),  # 从配置中获取梯度累积的批次数量
    gradient_clip_val=config.get("gradient_clip_val"),  # 从配置中获取梯度裁剪的阈值
    precision="16-mixed",  # 使用 16 位混合精度训练
    limit_val_batches=50, 
    num_sanity_val_steps=1,  # 在训练开始前进行 1 步验证数据的 sanity check
    logger=wandb_logger,
    callbacks=[early_stop_callback, SaveModelCallback()],  # 添加早停回调和自定义的模型保存回调
    val_check_interval=config.get("val_check_interval"),  # 从配置中获取验证检查的间隔
)


trainer.fit(model_module)  
