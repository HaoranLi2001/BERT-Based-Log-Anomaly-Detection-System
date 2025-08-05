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
    def __init__(self, base_model, hidden_size, vocab_size):
        super(LogBERT, self).__init__()
        self.base_model = base_model
        self.mlkp_head = nn.Linear(hidden_size, vocab_size)  # MLKP 预测 token
        self.vhm_head = nn.Linear(hidden_size, vocab_size)  # VHM 预测分布
        self.dropout = nn.Dropout(0.1)

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
def mask_tokens(input_ids, tokenizer, mask_prob=0.15):
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
    
    # 80% 替换为 [MASK]，10% 替换为随机 token，10% 保持不变
    mask_indices = mask_mask.nonzero(as_tuple=False)
    for idx in mask_indices:
        prob = torch.rand(1, device=device).item()
        if prob < 0.8:
            masked_ids[idx[0], idx[1]] = mask_token_id
        elif prob < 0.9:
            masked_ids[idx[0], idx[1]] = torch.randint(0, vocab_size, (1,), device=device).item()
    
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
def compute_window_losses(input_ids, attention_mask, mlkp_labels, true_histogram, model, mlkp_loss_fn, vhm_loss_fn, alpha, device):
    """
    计算 batch 内每个窗口的 MLKP 和 VHM 损失。
    返回：窗口级别的综合损失列表。
    """
    mlkp_logits, vhm_logits = model(input_ids, attention_mask)
    
    # MLKP 损失（逐样本）
    batch_size = input_ids.size(0)
    window_losses = []
    for i in range(batch_size):
        single_mlkp_logits = mlkp_logits[i].view(-1, mlkp_logits.size(-1))  # [seq_len, vocab_size]
        single_mlkp_labels = mlkp_labels[i].view(-1)  # [seq_len]
        loss_mlkp = mlkp_loss_fn(single_mlkp_logits, single_mlkp_labels)
        mlkp_losses = []
        vhm_losses = []
        
        # VHM 损失
        single_vhm_logits = vhm_logits[i:i+1]  # [1, vocab_size]
        single_histogram = true_histogram[i:i+1]  # [1, vocab_size]
        single_vhm_logits = torch.log_softmax(single_vhm_logits, dim=-1)
        loss_vhm = vhm_loss_fn(single_vhm_logits, single_histogram)
        
        # 综合损失
        loss = loss_mlkp + alpha * loss_vhm
        window_losses.append(loss.item())
        mlkp_losses.append(loss_mlkp.item())
        vhm_losses.append(loss_vhm.item())
    
    return window_losses, mlkp_losses, vhm_losses

# 7. 预测并打印详细信息
def log_prediction(model, tokenizer, test_dataset, threshold, device, alpha=1.0, num_samples=1):
    """
    对测试集的一个随机窗口进行预测，并详细打印预测过程。
    """
    model.eval()
    mlkp_loss_fn = CrossEntropyLoss(ignore_index=-100)
    vhm_loss_fn = KLDivLoss(reduction='batchmean')
    
    # 随机选择一个窗口
    indices = random.sample(range(len(test_dataset)), num_samples)
    for idx in indices:
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)  # [1, seq_len]
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        true_label = sample['labels'].item()
        
        # 解码窗口日志（截取前 50 个 token 以简化输出）
        decoded_text = tokenizer.decode(input_ids[0][:50], skip_special_tokens=True)
        
        with torch.no_grad():
            masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
            true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
            
            mlkp_logits, vhm_logits = model(masked_ids, attention_mask)
            
            mlkp_logits = mlkp_logits.view(-1, mlkp_logits.size(-1))
            mlkp_labels = mlkp_labels.view(-1)
            loss_mlkp = mlkp_loss_fn(mlkp_logits, mlkp_labels)
            
            vhm_logits = torch.log_softmax(vhm_logits, dim=-1)
            loss_vhm = vhm_loss_fn(vhm_logits, true_histogram)
            
            total_loss = loss_mlkp + alpha * loss_vhm
            pred = 1 if total_loss.item() > threshold else 0
        
        # 打印详细信息
        print(f"\nPrediction Details:",flush=True)
        print(f"  Window content (truncated): {decoded_text}",flush=True)
        print(f"  True label: {'Anomalous' if true_label == 1 else 'Normal'} ({true_label})",flush=True)
        print(f"  Predicted label: {'Anomalous' if pred == 1 else 'Normal'} ({pred})",flush=True)
        print(f"  MLKP loss: {loss_mlkp.item():.4f}",flush=True)
        print(f"  VHM loss: {loss_vhm.item():.4f}",flush=True)
        print(f"  Total loss: {total_loss.item():.4f}",flush=True)
        print(f"  Threshold: {threshold:.4f}",flush=True)
        print(f"  Correct: {pred == true_label}\n",flush=True)


# 8. 训练函数
def train_model(train_loader, test_loader, model_name="answerdotai/ModernBERT-base", num_epochs=20, alpha=1.0, num_layers_to_freeze=21):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}",flush=True)
    
    # 加载 ModernBERT 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, config=config)
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}",flush=True)
    
    # 实例化分类模型
    model = LogBERT(base_model, config.hidden_size, tokenizer.vocab_size).to(device)

    # 冻结指定层
    if num_layers_to_freeze > 0:
        print(f"Freezing the first {num_layers_to_freeze} layers of ModernBERT...")
        freeze_layers(model, num_layers_to_freeze)
    
    mlkp_loss_fn = CrossEntropyLoss(ignore_index=-100)
    vhm_loss_fn = KLDivLoss(reduction='batchmean')

    optimizer = AdamW(model.parameters(), lr=2e-5)

    print(f"TRAIN START",flush=True)
    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
            true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
            
            optimizer.zero_grad()
            mlkp_logits, vhm_logits = model(masked_ids, attention_mask)
            
            # MLKP 损失（batch 级别，用于优化）
            mlkp_logits = mlkp_logits.view(-1, tokenizer.vocab_size)
            mlkp_labels = mlkp_labels.view(-1)
            loss_mlkp = mlkp_loss_fn(mlkp_logits, mlkp_labels)
            
            # VHM 损失
            vhm_logits = torch.log_softmax(vhm_logits, dim=-1)
            loss_vhm = vhm_loss_fn(vhm_logits, true_histogram)
            
            # 综合损失
            loss = loss_mlkp + alpha * loss_vhm
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}",flush=True)
                
                # 在训练中进行预测
                # current_threshold = default_threshold if epoch == 0 else threshold
                # log_prediction(model, tokenizer, test_dataset, current_threshold, device, alpha)
        
        print(f"Epoch {epoch+1}, Average Train Loss: {total_loss / len(train_loader)}",flush=True)
        
        # 收集测试集窗口级别损失
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
                true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
                
                window_losses = compute_window_losses(
                    masked_ids, attention_mask, mlkp_labels, true_histogram,
                    model, mlkp_loss_fn, vhm_loss_fn, alpha, device
                )
                test_losses.extend(window_losses)
        
        # 计算阈值（窗口级别的 95% 分位数）
        threshold = np.percentile(test_losses, 95)
        print(f"Epoch {epoch+1}, Loss threshold (95th percentile): {threshold}",flush=True)
        log_prediction(model, tokenizer, test_dataset, threshold, device, alpha)

        # 测试集评估并打印所有窗口的损失
        test_losses = []
        mlkp_losses_all = []
        vhm_losses_all = []
        true_labels = []
        predictions = []
        window_data = []
        window_idx = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                masked_ids, mlkp_labels = mask_tokens(input_ids, tokenizer)
                true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
                
                window_losses, mlkp_losses, vhm_losses = compute_window_losses(
                    masked_ids, attention_mask, mlkp_labels, true_histogram,
                    model, mlkp_loss_fn, vhm_loss_fn, alpha, device
                )
                test_losses.extend(window_losses)
                mlkp_losses_all.extend(mlkp_losses)
                vhm_losses_all.extend(vhm_losses)
                true_labels.extend(labels.cpu().numpy())
                
                # 为每个窗口生成预测
                batch_predictions = [1 if loss > threshold else 0 for loss in window_losses]
                predictions.extend(batch_predictions)
                # 打印每个窗口的详细信息
                for i, (loss, mlkp_loss, vhm_loss, label, pred) in enumerate(zip(window_losses, mlkp_losses, vhm_losses, labels, batch_predictions)):
                    content = tokenizer.decode(input_ids[i][:50], skip_special_tokens=True)
                    print(f"Test Window {window_idx}:")
                    print(f"  Content (truncated): {content}")
                    print(f"  MLKP loss: {mlkp_loss:.4f}")
                    print(f"  VHM loss: {vhm_loss:.4f}")
                    print(f"  Total loss: {loss:.4f}")
                    print(f"  True label: {'Anomalous' if label.item() == 1 else 'Normal'} ({label.item()})")
                    print(f"  Predicted label: {'Anomalous' if pred == 1 else 'Normal'} ({pred})")
                    print(f"  Correct: {pred == label.item()}\n")
                    
                    # 收集窗口数据用于 CSV
                    window_data.append({
                        'window_idx': window_idx,
                        'content': content,
                        'mlkp_loss': mlkp_loss,
                        'vhm_loss': vhm_loss,
                        'total_loss': loss,
                        'true_label': label.item(),
                        'pred_label': pred,
                        'correct': pred == label.item()
                    })
                    window_idx += 1


        # 保存窗口数据到 CSV
        df = pd.DataFrame(window_data)
        csv_path = f"test_losses_epoch_{epoch+1}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved test window losses to {csv_path}")

                
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1_binary = f1_score(true_labels, predictions, zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        print(f"Epoch {epoch+1}, Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 (binary): {f1_binary:.4f}, F1 (macro): {f1_macro:.4f}")
    
    return model, tokenizer, threshold

# 9. 推理函数
def predict_log_sequence(log_sequence, model, tokenizer, threshold, max_length=512, device="cuda", alpha=1.0):
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
        true_histogram = get_histogram(input_ids, tokenizer.vocab_size).to(device)
        
        mlkp_logits, vhm_logits = model(masked_ids, attention_mask)
        mlkp_logits = mlkp_logits.view(-1, tokenizer.vocab_size)
        mlkp_labels = mlkp_labels.view(-1)
        mlkp_loss_fn = CrossEntropyLoss(ignore_index=-100)
        loss_mlkp = mlkp_loss_fn(mlkp_logits, mlkp_labels)
        vhm_logits = torch.log_softmax(vhm_logits, dim=-1)
        vhm_loss_fn = KLDivLoss(reduction='batchmean')
        loss_vhm = vhm_loss_fn(vhm_logits, true_histogram)
        loss = loss_mlkp + alpha * loss_vhm
        
        pred = 1 if loss.item() > threshold else 0
    return "Anomalous" if pred == 1 else "Normal"

# 主程序
if __name__ == "__main__":
    print("Start training...",flush=True)
    data_dir = "E:\LU\cs_proj\projects\ModernLogBERT\output"
    
    train_path = f"{data_dir}/train.pt"
    test_path = f"{data_dir}/test.pt"
    
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    train_dataset = LogDataset(train_data)
    test_dataset = LogDataset(test_data)
    print("dataset created",flush=True)

    batch_size = 16
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