import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader


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
    data = torch.load(data_path)
    print(f"Loaded data from {data_path}, size: {len(data)}",flush=True)
    return data

# 3. 定义模型
class LogClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_labels=2):
        super(LogClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

# 4. 计算精确度、召回率和 F1 分数
def evaluate_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    true_positives = sum((p == 1 and l == 1) for p, l in zip(preds, labels))
    false_positives = sum((p == 1 and l == 0) for p, l in zip(preds, labels))
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(preds, labels))
    true_negatives = sum((p == 0 and l == 0) for p, l in zip(preds, labels))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# 5. 冻结指定层
def freeze_layers(model, num_layers_to_freeze, freeze_embeddings=False):
    """
    冻结 ModernBERT 的前 num_layers_to_freeze 层。
    model: LogClassifier 模型
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

    # 冻结嵌入层（可选）
    if freeze_embeddings:
        for name, param in model.base_model.embeddings.named_parameters():
            param.requires_grad = False
            print(f"Froze embedding parameter: {name}, shape: {param.shape}")
    else:
        print("Embeddings are not frozen, will be fine-tuned.")

    # 确保分类 Head 可训练
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True
        print(f"Classifier parameter (trainable): {name}, shape: {param.shape}")

    # 打印冻结和可训练参数的数量
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen parameters: {frozen_params}")
    print(f"Trainable parameters: {trainable_params}")


# 6. 训练函数
def train_model(train_loader, val_loader, test_loader, model_name="answerdotai/ModernBERT-base", tokenizer_path="processed_data", num_epochs=10, save_dir="saved_model", num_layers_to_freeze=10, resume_from=None):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}",flush=True)
    if device.type == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # 加载新训练的 Tokenizer
    print("Loading tokenizer...",flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 加载 ModernBERT 模型
    print("Loading ModernBERT model...",flush=True)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)

    # 替换 ModernBERT 的嵌入层以匹配新 Tokenizer 的词汇表
    print("Resizing model embeddings to match new tokenizer...",flush=True)
    new_vocab_size = tokenizer.vocab_size
    model.resize_token_embeddings(new_vocab_size)

    checkpoint_path = "pretrained_model/ep340.pt"
    print(f"Loading pretrained checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print("Keys in checkpoint:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Full model object")
    # 加载状态字典，忽略不匹配的键（例如 MLM Head）
    model.load_state_dict(checkpoint, strict=False)
    print("Loaded pretrained checkpoint successfully!")

    # 实例化分类模型
    hidden_size = config.hidden_size
    classifier = LogClassifier(model, hidden_size, num_labels=2)
    classifier = classifier.to(device)

    # 加载保存的模型（如果继续训练）
    if resume_from:
        print(f"Loading saved model from {resume_from}...", flush=True)
        classifier.load_state_dict(torch.load(resume_from, map_location=device))
        print("Loaded saved model successfully!")

    # 冻结指定层
    if num_layers_to_freeze > 0:
        print(f"Freezing the first {num_layers_to_freeze} layers of ModernBERT...")
        freeze_layers(classifier, num_layers_to_freeze)

    # 处理数据不平衡（加权损失）
    normal_count = sum(1 for batch in train_loader for label in batch['labels'] if label == 0)
    anomaly_count = sum(1 for batch in train_loader for label in batch['labels'] if label == 1)
    class_weights = torch.tensor([1.0, normal_count / anomaly_count]).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)

    # 定义优化器
    print("Loss function defined", flush=True)
    optimizer = AdamW(classifier.parameters(), lr=2e-5)
    print("Loss function defined", flush=True)

    # 训练循环
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        for i,batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = classifier(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 10 == 0:  # 每 10 个 batch 打印一次
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}", flush=True)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}",flush=True)

        # 验证
        classifier.eval()
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = classifier(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                all_preds.extend(preds)
                all_labels.extend(labels)

        accuracy = total_correct / total_samples
        precision, recall, f1 = evaluate_metrics(all_preds, all_labels)
        print(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}",flush=True)

    # 测试
    classifier.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = classifier(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = total_correct / total_samples
    precision, recall, f1 = evaluate_metrics(all_preds, all_labels)
    print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}",flush=True)

    # 保存模型
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "log_classifier_froze20.pt")
    torch.save(classifier.state_dict(), save_path)
    print(f"Saved model state dict to {save_path}")

    return classifier

# 7. 推理函数
def predict_log_sequence(log_sequence, model, tokenizer, max_length=512, device="cuda"):
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
        logits = model(input_ids, attention_mask)
        pred = torch.argmax(logits, dim=1).cpu().item()
    return "Anomalous" if pred == 1 else "Normal"

# 8. 加载保存的模型并进行推理
def load_model_and_predict(model_path, tokenizer_path, model_name="answerdotai/ModernBERT-base", log_sequence=None, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 加载 ModernBERT 模型
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)

    # 替换嵌入层
    new_vocab_size = tokenizer.vocab_size
    model.resize_token_embeddings(new_vocab_size)

    # 实例化分类模型
    hidden_size = config.hidden_size
    classifier = LogClassifier(model, hidden_size, num_labels=2)
    classifier.load_state_dict(torch.load(model_path))
    classifier = classifier.to(device)
    classifier.eval()

    # 进行推理
    if log_sequence:
        result = predict_log_sequence(log_sequence, classifier, tokenizer, max_length=max_length, device=device)
        print(f"Sequence {log_sequence} is {result}")

    return classifier

# 9. 主函数
if __name__ == "__main__":
    print("start training",flush=True)
    # 数据路径
    data_dir = "/home/mingtong/projects/ModernBERT_Log_newtokenizer/processed_data"
    train_path = f"{data_dir}/train.pt"
    val_path = f"{data_dir}/val.pt"
    test_path = f"{data_dir}/test.pt"

    # 加载数据
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)

    # 创建数据集和 DataLoader
    train_dataset = LogDataset(train_data)
    val_dataset = LogDataset(val_data)
    test_dataset = LogDataset(test_data)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 继续训练
    classifier = train_model(
        train_loader, 
        val_loader, 
        test_loader, 
        tokenizer_path=data_dir, 
        num_epochs=10,  # 继续训练5个epoch（可调整）
        save_dir="saved_model", 
        num_layers_to_freeze=10,  # 保持原有冻结层数
        resume_from="saved_model/log_classifier_froze10.pt"  # 加载保存的模型
    )

    # 加载保存的模型并进行推理
    model_path = "saved_model/log_classifier_froze20.pt"
    log_sequence = ["081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"]
    loaded_classifier = load_model_and_predict(model_path, data_dir, log_sequence=log_sequence)