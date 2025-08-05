import sys
sys.path.append('../')
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import os


log_file = "output_20250515_2.log"
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", log_file)

class Tee:
    def __init__(self, file_handle, console):
        self.file = file_handle
        self.console = console

    def write(self, message):
        self.file.write(message)
        self.console.write(message)
        self.file.flush()

    def flush(self):
        self.file.flush()
        self.console.flush()

log_handle = open(log_path, 'w')
sys.stdout = Tee(log_handle, sys.stdout)
sys.stderr = Tee(log_handle, sys.stderr)

# 1. 定义数据集类
class LogDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.data[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 加载数据
def load_data(data_path):
    print(f"Loading data from {data_path}...",flush=True)
    data = torch.load(data_path)
    print(f"Loaded data from {data_path}, size: {len(data)}",flush=True)
    return data

# 3. 定义模型
class LogBERT(nn.Module):
    def __init__(self, base_model, hidden_size, vocab_size, tokenizer):
        super(LogBERT, self).__init__()
        self.base_model = base_model
        self.mlkp_head = nn.Linear(hidden_size, vocab_size)  # MLKP 预测 token
        self.vhm_head = nn.Linear(hidden_size, vocab_size)  # VHM 预测分布
        self.dropout = nn.Dropout(0.1)
        self.tokenizer = tokenizer  # 存储 tokenizer

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # MLKP 输出
        mlkp_logits = self.mlkp_head(self.dropout(sequence_output))  # [batch_size, seq_len, vocab_size]
        
        # VHM 输出
        vhm_logits = self.vhm_head(self.dropout(cls_output))  # [batch_size, vocab_size]
        
        return mlkp_logits, vhm_logits
    
# 4. MLKP 掩码
def mask_tokens(input_ids, tokenizer, mask_prob=0.3):
    """
    随机掩码 token，返回掩码后的 input_ids 和 MLKP 标签。
    """
    device = input_ids.device  # 获取 input_ids 的设备
    masked_ids = input_ids.clone()
    labels = input_ids.clone()
    mask_token_id = torch.tensor(tokenizer.mask_token_id, device=device)
    vocab_size = tokenizer.vocab_size
    
    # 随机选择掩码位置
    rand = torch.rand(input_ids.shape, device=device)  # 确保 rand 在同一设备
    # 将 tokenizer 的 token ID 转换为张量并移到同一设备
    cls_token_id = torch.tensor(tokenizer.cls_token_id, device=device)
    sep_token_id = torch.tensor(tokenizer.sep_token_id, device=device)
    pad_token_id = torch.tensor(tokenizer.pad_token_id, device=device)
    
    mask_mask = (rand < mask_prob) & (input_ids != cls_token_id) & (input_ids != sep_token_id) & (input_ids != pad_token_id)
    
    # 替换为 [MASK]
    mask_indices = mask_mask.nonzero(as_tuple=False)
    for idx in mask_indices:
        prob = torch.rand(1, device=device).item()
        if prob < 1:
            masked_ids[idx[0], idx[1]] = mask_token_id
    
    # 非掩码位置的标签设为 -100
    labels[~mask_mask] = -100
    return masked_ids, labels

# 5. VHM 目标分布
def get_histogram(input_ids, vocab_size):
    """
    计算 token 的频率分布。
    """
    batch_size = input_ids.size(0)
    histograms = torch.zeros(batch_size, vocab_size, device=input_ids.device)
    for i in range(batch_size):
        valid_tokens = input_ids[i][input_ids[i] < vocab_size]  # 忽略 padding 等
        hist = torch.bincount(valid_tokens, minlength=vocab_size).float()
        hist = hist / (hist.sum() + 1e-10)  # 归一化
        histograms[i] = hist
    return histograms

# 5. 冻结指定层
def freeze_layers(model, num_layers_to_freeze):
    """
    冻结 ModernBERT 的前 num_layers_to_freeze 层，并打印每层可训练和不可训练参数数量。
    
    Args:
        model: LogClassifier 模型 (LogBERT)
        num_layers_to_freeze: 要冻结的层数
    """
    # ModernBERT 的 Transformer 层位于 base_model.layers
    layers = model.base_model.layers
    total_layers = len(layers)
    print(f"Total Transformer layers: {total_layers}")

    if num_layers_to_freeze >= total_layers:
        print(f"Warning: num_layers_to_freeze ({num_layers_to_freeze}) >= total layers ({total_layers}), freezing all layers.")
        num_layers_to_freeze = total_layers

    # 冻结前 num_layers_to_freeze 层
    for i in range(num_layers_to_freeze):
        layer = layers[i]
        for name, param in layer.named_parameters():
            param.requires_grad = False
            print(f"Froze layer {i} parameter: {name}, shape: {param.shape}")

    # 确保分类 Head 可训练
    for name, param in model.mlkp_head.named_parameters():
        param.requires_grad = True
        print(f"Classifier parameter (trainable): {name}, shape: {param.shape}")
    for name, param in model.vhm_head.named_parameters():
        param.requires_grad = True
        print(f"Classifier parameter (trainable): {name}, shape: {param.shape}")

    # 按层统计可训练和不可训练参数
    print("\nPer-layer Parameter Statistics:")
    print("-" * 50)
    layer_params = {}
    
    # 遍历模型的所有参数，按层分组
    for name, param in model.named_parameters():
        # 提取层级名称（例如，base_model.embeddings, base_model.layers.0）
        layer_name = ".".join(name.split(".")[:3])  # 取前三级，如 base_model.layers.0
        if layer_name not in layer_params:
            layer_params[layer_name] = {"trainable": 0, "frozen": 0}
        
        # 统计参数数量
        num_params = param.numel()
        if param.requires_grad:
            layer_params[layer_name]["trainable"] += num_params
        else:
            layer_params[layer_name]["frozen"] += num_params

    # 打印每层统计
    for layer_name, stats in layer_params.items():
        print(f"Layer: {layer_name}")
        print(f"  Trainable parameters: {stats['trainable']}")
        print(f"  Frozen parameters: {stats['frozen']}")
        print("-" * 50)

    # 打印冻结和可训练参数的总数
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Frozen parameters: {frozen_params}")
    print(f"Total Trainable parameters: {trainable_params}")

# 6. 计算每个窗口的损失
def compute_window_losses(input_ids, attention_mask, mlkp_labels, true_histogram, model, mlkp_loss_fn, vhm_loss_fn, alpha, device, g=5, r=2):
    """
    计算每个窗口的损失、异常键数量、掩码词语和候选集。
    
    Args:
        g: 候选集大小
        r: 异常日志键阈值
    Returns:
        window_losses: 总损失列表
        mlkp_losses: MLKP 损失列表
        vhm_losses: VHM 损失列表
        predicted_tokens: 预测 token 列表
        anomalous_keys_list: 每个窗口的异常键数量列表
        predictions: 预测标签列表
        masked_tokens_list: 每个窗口的掩码词语列表
        candidate_sets_list: 每个窗口的候选集列表
    """
    mlkp_logits, vhm_logits = model(input_ids, attention_mask)
    
    batch_size = input_ids.size(0)
    window_losses = []
    mlkp_losses = []
    vhm_losses = []
    predicted_tokens = []
    anomalous_keys_list = []
    predictions = []
    masked_tokens_list = []
    candidate_sets_list = []
    
    for i in range(batch_size):
        single_mlkp_logits = mlkp_logits[i].view(-1, mlkp_logits.size(-1))
        single_mlkp_labels = mlkp_labels[i].view(-1)
        loss_mlkp = mlkp_loss_fn(single_mlkp_logits, single_mlkp_labels)
        
        single_vhm_logits = vhm_logits[i:i+1]
        single_histogram = true_histogram[i:i+1]
        single_vhm_logits = torch.log_softmax(single_vhm_logits, dim=-1)
        loss_vhm = vhm_loss_fn(single_vhm_logits, single_histogram)
        
        loss = loss_mlkp + alpha * loss_vhm
        window_losses.append(loss.item())
        mlkp_losses.append(loss_mlkp.item())
        vhm_losses.append(loss_vhm.item())
        
        # 统计异常键并记录掩码词语和候选集
        anomalous_keys = 0
        mask_indices = (single_mlkp_labels != -100).nonzero(as_tuple=False).squeeze(-1)
        masked_tokens = []
        candidate_sets = []
        for mask_idx in mask_indices:
            probs = torch.softmax(single_mlkp_logits[mask_idx], dim=-1)
            top_g_indices = torch.topk(probs, k=g, dim=-1).indices
            true_token_id = single_mlkp_labels[mask_idx].item()
            if true_token_id not in top_g_indices:
                anomalous_keys += 1
            # 记录掩码词语（转换为文本）
            masked_token = model.tokenizer.decode([true_token_id], skip_special_tokens=True)
            masked_tokens.append(masked_token)
            # 记录候选集（转换为文本）
            candidate_tokens = [model.tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in top_g_indices]
            candidate_sets.append(','.join(candidate_tokens))
        
        anomalous_keys_list.append(anomalous_keys)
        masked_tokens_list.append(';'.join(masked_tokens))
        candidate_sets_list.append(';'.join(candidate_sets))
        
        # 预测标签
        pred = 1 if anomalous_keys > r else 0
        predictions.append(pred)
        
        # 获取预测 token
        pred_tokens = input_ids[i].clone()
        if mask_indices.numel() > 0:
            pred_token_ids = torch.argmax(single_mlkp_logits[mask_indices], dim=-1)
            pred_tokens[mask_indices] = pred_token_ids
        predicted_tokens.append(pred_tokens)
    
    return window_losses, mlkp_losses, vhm_losses, predicted_tokens, anomalous_keys_list, predictions, masked_tokens_list, candidate_sets_list

# 7. 预测并打印详细信息
def log_prediction(model, tokenizer, test_dataset, g=5, r=2, device="cuda", num_samples=1):
    """
    对测试集的一个随机窗口进行预测，基于候选集机制检测异常，并详细打印预测过程。
    
    Args:
        model: LogBERT 模型
        tokenizer: 分词器
        test_dataset: 测试数据集
        g: 候选集大小（前 g 个高概率 token）
        r: 异常日志键阈值
        device: 设备（cuda 或 cpu）
        num_samples: 打印的样本数
    """
    model.eval()
    mlkp_loss_fn = CrossEntropyLoss(ignore_index=-100)
    
    # 随机选择样本
    indices = random.sample(range(len(test_dataset)), num_samples)
    for idx in indices:
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)  # [1, seq_len]
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        true_label = sample['labels'].item()
        
        # 解码窗口日志（截取前 50 个 token 以简化输出）
        decoded_text = tokenizer.decode(input_ids[0][:50], skip_special_tokens=True)
        
        with torch.no_grad():
            # 掩码 token
            masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
            # 计算 MLKP 预测
            mlkp_logits, _ = model(masked_ids, attention_mask)  # [1, seq_len, vocab_size]
            mlkp_logits = mlkp_logits.view(-1, mlkp_logits.size(-1))  # [seq_len, vocab_size]
            mlkp_labels = mlkp_labels.view(-1)  # [seq_len]
            
            # 统计异常日志键
            anomalous_keys = 0
            mask_indices = (mlkp_labels != -100).nonzero(as_tuple=False).squeeze(-1)
            for mask_idx in mask_indices:
                # 获取前 g 个高概率 token
                probs = torch.softmax(mlkp_logits[mask_idx], dim=-1)
                top_g_indices = torch.topk(probs, k=g, dim=-1).indices
                true_token_id = mlkp_labels[mask_idx].item()
                if true_token_id not in top_g_indices:
                    anomalous_keys += 1
            
            # 判断序列是否异常
            pred = 1 if anomalous_keys > r else 0
            
            # 计算 MLKP 损失（用于参考）
            loss_mlkp = mlkp_loss_fn(mlkp_logits, mlkp_labels)
            
            # 掩码和预测内容
            masked_content = tokenizer.decode(masked_ids[0][:50], skip_special_tokens=True)
            pred_tokens = input_ids[0].clone()
            if mask_indices.numel() > 0:
                pred_token_ids = torch.argmax(mlkp_logits[mask_indices], dim=-1)
                pred_tokens[mask_indices] = pred_token_ids
            predicted_content = tokenizer.decode(pred_tokens[:50], skip_special_tokens=True)
        
        # 打印详细信息
        print(f"\nPrediction Details (Sample {idx}):")
        print(f"  Window content (truncated): {decoded_text}")
        print(f"  Masked content (truncated): {masked_content}")
        print(f"  Predicted content (truncated): {predicted_content}")
        print(f"  True label: {'Anomalous' if true_label == 1 else 'Normal'} ({true_label})")
        print(f"  Predicted label: {'Anomalous' if pred == 1 else 'Normal'} ({pred})")
        print(f"  Anomalous keys: {anomalous_keys} (threshold r={r})")
        print(f"  MLKP loss: {loss_mlkp.item():.4f}")
        print(f"  Correct: {pred == true_label}\n")

# 8. 训练函数
def train_model(train_loader, test_loader, model_name="answerdotai/ModernBERT-base", tokenizer_path="output", num_epochs=20, alpha=1.0, num_layers_to_freeze=21, g=5, r=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}", flush=True)

    # 替换嵌入层
    print("Resizing model embeddings to match new tokenizer...", flush=True)
    new_vocab_size = tokenizer.vocab_size
    base_model.resize_token_embeddings(new_vocab_size)

    checkpoint_path = "E:\\LU\\cs_proj\\projects\\.pretrained_model\\ep165.pt"
    print(f"Loading pretrained checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint, strict=False)
    print("Loaded pretrained checkpoint successfully!")

    model = LogBERT(base_model, config.hidden_size, tokenizer.vocab_size, tokenizer).to(device)

    if num_layers_to_freeze > 0:
        print(f"Freezing the first {num_layers_to_freeze} layers of ModernBERT...")
        freeze_layers(model, num_layers_to_freeze)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    mlkp_loss_fn = CrossEntropyLoss(ignore_index=-100)
    vhm_loss_fn = KLDivLoss(reduction='batchmean')

    print(f"TRAIN START", flush=True)
    for epoch in range(num_epochs):
        # 训练循环（保持不变）
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
            true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
            
            optimizer.zero_grad()
            mlkp_logits, vhm_logits = model(masked_ids, attention_mask)
            
            mlkp_logits = mlkp_logits.view(-1, tokenizer.vocab_size)
            mlkp_labels = mlkp_labels.view(-1)
            loss_mlkp = mlkp_loss_fn(mlkp_logits, mlkp_labels)
            
            vhm_logits = torch.log_softmax(vhm_logits, dim=-1)
            loss_vhm = vhm_loss_fn(vhm_logits, true_histogram)
            
            loss = loss_mlkp + alpha * loss_vhm
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}", flush=True)
        
        print(f"Epoch {epoch+1}, Average Train Loss: {total_loss / len(train_loader)}", flush=True)
        
        # 测试循环
        print(f"Epoch {epoch+1}, Testing...", flush=True)
        model.eval()
        test_losses = []
        mlkp_losses_all = []
        vhm_losses_all = []
        true_labels = []
        predictions = []
        window_data = []
        window_idx = 0
        csv_window_count = 0
        max_csv_windows = 1000  # 限制 CSV 保存 1000 条

        with torch.no_grad():
            for j, batch in enumerate(test_loader):
                if j % 100 == 0:
                    print(f"Epoch {epoch+1}, Test Batch {j+1}", flush=True)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
                true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
                
                window_losses, mlkp_losses, vhm_losses, predicted_tokens, anomalous_keys_list, batch_predictions, masked_tokens_list, candidate_sets_list = compute_window_losses(
                    masked_ids, attention_mask, mlkp_labels, true_histogram,
                    model, mlkp_loss_fn, vhm_loss_fn, alpha, device, g=g, r=r
                )
                test_losses.extend(window_losses)
                mlkp_losses_all.extend(mlkp_losses)
                vhm_losses_all.extend(vhm_losses)
                predictions.extend(batch_predictions)
                true_labels.extend(labels.cpu().numpy())

                # 收集窗口数据（仅前 1000 个窗口写入 CSV）
                for i, (loss, mlkp_loss, vhm_loss, anomalous_keys, label, pred, pred_tokens, masked_tokens, candidate_sets) in enumerate(zip(
                    window_losses, mlkp_losses, vhm_losses, anomalous_keys_list, labels, batch_predictions, predicted_tokens, masked_tokens_list, candidate_sets_list)):
                    if csv_window_count < max_csv_windows:
                        content = tokenizer.decode(input_ids[i][:50], skip_special_tokens=True)
                        masked_content = tokenizer.decode(masked_ids[i][:50], skip_special_tokens=True)
                        predicted_content = tokenizer.decode(pred_tokens[:50], skip_special_tokens=True)
                        
                        window_data.append({
                            'window_idx': window_idx,
                            'content': content,
                            'masked_content': masked_content,
                            'predicted_content': predicted_content,
                            'masked_tokens': masked_tokens,
                            'candidate_sets': candidate_sets,
                            'mlkp_loss': mlkp_loss,
                            'vhm_loss': vhm_loss,
                            'total_loss': loss,
                            'anomalous_keys': anomalous_keys,
                            'true_label': label.item(),
                            'pred_label': pred,
                            'correct': pred == label.item()
                        })
                        csv_window_count += 1
                    window_idx += 1

        # 保存窗口数据到 CSV
        df = pd.DataFrame(window_data)
        csv_path = f"test_results_epoch_{epoch+1}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(window_data)} test window results to {csv_path}", flush=True)

        # 计算评估指标（使用全部测试集）
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1_binary = f1_score(true_labels, predictions, zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        print(f"Epoch {epoch+1}, Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 (binary): {f1_binary:.4f}, F1 (macro): {f1_macro:.4f}", flush=True)
    
    return model, tokenizer, (g, r)

# 9. 推理函数
def predict_log_sequence(log_sequence, model, tokenizer, g=5, r=2, max_length=512, device="cuda"):
    """
    预测单个日志序列是否异常，基于候选集机制。
    
    Args:
        g: 候选集大小
        r: 异常日志键阈值
    """
    log_text = " ".join(log_sequence)
    inputs = tokenizer(
        log_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
        mlkp_logits, _ = model(masked_ids, attention_mask)
        mlkp_logits = mlkp_logits.view(-1, mlkp_logits.size(-1))
        mlkp_labels = mlkp_labels.view(-1)
        
        anomalous_keys = 0
        mask_indices = (mlkp_labels != -100).nonzero(as_tuple=False).squeeze(-1)
        for mask_idx in mask_indices:
            probs = torch.softmax(mlkp_logits[mask_idx], dim=-1)
            top_g_indices = torch.topk(probs, k=g, dim=-1).indices
            true_token_id = mlkp_labels[mask_idx].item()
            if true_token_id not in top_g_indices:
                anomalous_keys += 1
        
        pred = 1 if anomalous_keys > r else 0
    return "Anomalous" if pred == 1 else "Normal"


# 主程序
if __name__ == "__main__":
    print("Start training...",flush=True)
    data_dir = "E:\LU\cs_proj\projects\ModernLogBERT_newtokenizer\output"
    
    train_path = f"{data_dir}/train.pt"
    test_path = f"{data_dir}/test.pt"
    
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    train_dataset = LogDataset(train_data)
    test_dataset = LogDataset(test_data)
    print("dataset created",flush=True)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("dataloader created",flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(data_dir)
    print("tokenizer prepared",flush=True)

    model, tokenizer, threshold = train_model(train_loader, test_loader)
    print("Training completed",flush=True)
    
    # 示例预测
    log_sequence = ["Receiving block src: dest:", "Receiving block src: dest:"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = predict_log_sequence(log_sequence, model, tokenizer, threshold, device=device)
    print(f"Sequence {log_sequence} is {result}",flush=True)

    log_handle.close()