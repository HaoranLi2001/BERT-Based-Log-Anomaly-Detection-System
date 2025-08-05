import re
import pandas as pd
from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split
import os
from transformers import AutoTokenizer

# 1. 解析 HDFS.log 文件
def parse_log_file(log_file_path):
    """
    解析 HDFS.log 文件，按 block_id 分组原始日志。
    返回：block_logs 字典，key 为 block_id，value 为原始日志消息列表。
    """
    block_logs = defaultdict(list)
    
    # 正则表达式提取 block_id
    block_pattern = re.compile(r'blk_[-]?\d+')
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # 示例日志：
            # 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
            log_message = line.strip()
            
            # 提取 block_id
            block_match = block_pattern.search(log_message)
            if block_match:
                block_id = block_match.group(0)
                block_logs[block_id].append(log_message)
    
    return block_logs

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

# 3. 按 block_id 分组并关联标签
def group_logs_with_labels(block_logs, label_map):
    """
    将日志按 block_id 分组，并关联标签。
    返回：data 列表，元素为 (日志序列, 标签) 对。
    """
    data = []
    for block_id, logs in block_logs.items():
        if block_id in label_map:
            label = label_map[block_id]
            data.append((logs, label))
    return data

# 4. 预处理并分词（直接使用原始日志）
def preprocess_and_tokenize(data, tokenizer, max_length=512):
    """
    将原始日志序列分词为 input_ids 和 attention_mask。
    返回：处理后的数据列表，元素为 (input_ids, attention_mask, label)。
    """
    processed_data = []
    for log_sequence, label in data:
        # 将日志序列拼接为一个字符串（用空格分隔）
        log_text = " ".join(log_sequence)
        # 分词
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
    torch.save(data, save_path)
    print(f"Saved data to {save_path}")

# 6. 主函数：数据预处理
def preprocess_data(log_file_path, label_file_path, tokenizer_path, save_dir="processed_data", max_length=512):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载新训练的 Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 解析日志文件（直接使用原始日志）
    print("Parsing log file...")
    block_logs = parse_log_file(log_file_path)

    # 加载标签文件
    print("Loading labels...")
    label_map = load_labels(label_file_path)

    # 按 block_id 分组并关联标签
    print("Grouping logs with labels...")
    data = group_logs_with_labels(block_logs, label_map)

    # 划分数据集：训练集（80%）、验证集（10%）、测试集（10%）
    print("Splitting dataset...")
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=[label for _, label in data])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=[label for _, label in temp_data])

    # 预处理并分词
    print("Preprocessing and tokenizing data...")
    train_processed = preprocess_and_tokenize(train_data, tokenizer, max_length)
    val_processed = preprocess_and_tokenize(val_data, tokenizer, max_length)
    test_processed = preprocess_and_tokenize(test_data, tokenizer, max_length)

    # 保存数据
    save_data(train_processed, os.path.join(save_dir, "train.pt"))
    save_data(val_processed, os.path.join(save_dir, "val.pt"))
    save_data(test_processed, os.path.join(save_dir, "test.pt"))

    # 保存分词器（供训练脚本使用）
    tokenizer.save_pretrained(save_dir)
    print(f"Saved tokenizer to {save_dir}")

if __name__ == "__main__":
    print("starting dataprocess...", flush=True)
    log_file_path = "/home/mingtong/.dataset/hdfs/HDFS.log"
    label_file_path = "/home/mingtong/.dataset/hdfs/anomaly_label.csv"
    tokenizer_path = "/home/mingtong/projects/ModernBERT_Log_newtokenizer/tokenizer_log"  # 替换为你的新 Tokenizer 路径
    preprocess_data(log_file_path, label_file_path, tokenizer_path)