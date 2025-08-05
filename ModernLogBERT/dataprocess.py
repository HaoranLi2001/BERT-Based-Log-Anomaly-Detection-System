import sys
sys.path.append('../')
import re
import pandas as pd
from transformers import AutoTokenizer
import torch
import os
import random
import numpy as np

# 1. 解析 HDFS.log 文件，保留 block_id
def parse_log_file(log_file_path):
    """
    解析 HDFS.log 文件，提取日志消息和 block_id，去除动态信息。
    返回：parsed_logs 列表，元素为 (日志消息, block_id) 对。
    """
    parsed_logs = []
    block_pattern = re.compile(r'blk_[-]?\d+')
    
    with open(log_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 5)
            if len(parts) < 6:
                continue
            log_message = parts[5]
            block_match = block_pattern.search(log_message)
            if block_match:
                block_id = block_match.group(0)
                log_message = re.sub(r'/\d+\.\d+\.\d+\.\d+:\d+', '', log_message)
                log_message = re.sub(r'blk_[-]?\d+', 'BLK_ID', log_message)
                log_message = log_message.strip()
                parsed_logs.append((log_message, block_id))
    
    return parsed_logs

# 2. 加载 anomaly_label.csv 文件
def load_labels(label_file_path):
    """
    加载 anomaly_label.csv 文件，返回 block_id 到标签的映射。
    """
    labels_df = pd.read_csv(label_file_path)
    label_map = {}
    for _, row in labels_df.iterrows():
        block_id = row['BlockId']
        label = row['Label']
        label_map[block_id] = 0 if label == 'Normal' else 1
    return label_map

# 3. 随机窗口采样
def window_sampling(parsed_logs, label_map, window_size=50, train_ratio=0.8, test_ratio=0.2):
    """
    按随机窗口采样日志，训练集和测试集不重合，训练集只保留正常窗口。
    返回：train_windows, test_windows 列表，元素为 (日志序列, 标签)。
    """
    total_logs = len(parsed_logs)
    train_target = int(total_logs * train_ratio)  # 训练集目标日志量
    test_target = int(total_logs * test_ratio)    # 测试集目标日志量
    
    # 分割日志索引
    train_size = int(total_logs * train_ratio)
    train_logs = parsed_logs[:train_size]
    test_logs = parsed_logs[train_size:]
    
    train_windows = []
    test_windows = []
    train_log_count = 0
    test_log_count = 0
    
    # 训练集随机采样
    train_indices = list(range(len(train_logs) - window_size + 1))
    random.shuffle(train_indices)  # 打乱起点
    for start_idx in train_indices:
        if train_log_count >= train_target:
            break
        window = train_logs[start_idx:start_idx + window_size]
        if len(window) != window_size:  # 确保窗口完整
            continue
        log_texts = [log for log, _ in window]
        block_ids = [block_id for _, block_id in window]
        
        # 检查窗口是否正常
        is_normal = all(label_map.get(block_id, 0) == 0 for block_id in block_ids)
        if is_normal:
            train_windows.append((log_texts, 0))  # 正常窗口，标签为0
            train_log_count += len(log_texts)
    
    # 测试集随机采样
    test_indices = list(range(len(test_logs) - window_size + 1))
    random.shuffle(test_indices)
    for start_idx in test_indices:
        if test_log_count >= test_target:
            break
        window = test_logs[start_idx:start_idx + window_size]
        if len(window) != window_size:
            continue
        log_texts = [log for log, _ in window]
        block_ids = [block_id for _, block_id in window]
        
        # 测试集保留所有窗口，标签基于是否有异常
        is_normal = all(label_map.get(block_id, 0) == 0 for block_id in block_ids)
        label = 0 if is_normal else 1
        test_windows.append((log_texts, label))
        test_log_count += len(log_texts)
    
    print(f"Train windows: {len(train_windows)}, logs: {train_log_count}",flush=True)
    print(f"Test windows: {len(test_windows)}, logs: {test_log_count}",flush=True)
    return train_windows, test_windows

# 4. 预处理并分词
def preprocess_and_tokenize(data, tokenizer, max_length=512):
    """
    将日志序列分词为 input_ids 和 attention_mask。
    返回：处理后的数据列表，元素为 (input_ids, attention_mask, label)。
    """
    processed_data = []
    for log_sequence, label in data:
        log_text = " ".join(log_sequence)
        inputs = tokenizer(
            log_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        input_ids = inputs['input_ids'].squeeze(0)  # [max_length]
        attention_mask = inputs['attention_mask'].squeeze(0)  # [max_length]
        processed_data.append((input_ids, attention_mask, label))
    return processed_data

# 5. 保存数据
def save_data(data, save_path):
    """
    将数据保存为 .pt 文件。
    """
    torch.save(data, save_path)
    print(f"Saved data to {save_path}",flush=True)

# 6. 主函数：数据预处理
def preprocess_data(log_file_path, label_file_path, save_dir="E:\LU\cs_proj\projects\ModernLogBERT\output", model_name="answerdotai/ModernBERT-base", window_size=50, max_length=512):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading tokenizer...",flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Parsing log file...",flush=True)
    parsed_logs = parse_log_file(log_file_path)

    print("Loading labels...",flush=True)
    label_map = load_labels(label_file_path)

    print("Sampling windows...",flush=True)
    train_windows, test_windows = window_sampling(parsed_logs, label_map, window_size=window_size)

    print("Preprocessing and tokenizing data...",flush=True)
    print("Preprocessing train data...",flush=True)
    train_processed = preprocess_and_tokenize(train_windows, tokenizer, max_length)
    print("Preprocessing test data...",flush=True)
    test_processed = preprocess_and_tokenize(test_windows, tokenizer, max_length)

    save_data(train_processed, os.path.join(save_dir, "train.pt"))
    save_data(test_processed, os.path.join(save_dir, "test.pt"))

    tokenizer.save_pretrained(save_dir)
    print(f"Saved tokenizer to {save_dir}",flush=True)

if __name__ == "__main__":
    print("Starting dataprocess...",flush=True)
    log_file_path = "E:\LU\cs_proj\projects\.dataset\hdfs\HDFS.log"
    label_file_path = "E:\LU\cs_proj\projects\.dataset\hdfs/anomaly_label.csv"
    preprocess_data(log_file_path, label_file_path)