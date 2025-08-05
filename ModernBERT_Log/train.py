import sys
sys.path.append('../')
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

print("Before importing", flush=True)

# 启用 TF32 加速
torch.set_float32_matmul_precision('high')
print("TF32 enabled", flush=True)

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

print("Class defined", flush=True)

# 2. 加载数据
def load_data(data_path):
    print(f"Loading data from {data_path}...", flush=True)
    data = torch.load(data_path)
    print(f"Loaded data from {data_path}, size: {len(data)}", flush=True)
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

print("Model class defined", flush=True)

# 4. 计算精确度、召回率和 F1 分数
def evaluate_metrics(preds, labels):
    """
    计算精确度、召回率和 F1 分数。
    preds: 预测标签 (list or tensor)
    labels: 真实标签 (list or tensor)
    """
    # 确保 preds 和 labels 是列表
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 计算 TP, FP, FN
    true_positives = sum((p == 1 and l == 1) for p, l in zip(preds, labels))  # 预测为正且实际为正
    false_positives = sum((p == 1 and l == 0) for p, l in zip(preds, labels))  # 预测为正但实际为负
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(preds, labels))  # 预测为负但实际为正
    true_negatives = sum((p == 0 and l == 0) for p, l in zip(preds, labels))  # 预测为负且实际为负

    # 计算精确度、召回率和 F1 分数
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# 5. 训练函数
def train_model(train_loader, val_loader, test_loader, model_name="answerdotai/ModernBERT-base", num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU device: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading model...", flush=True)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    print("Model loaded", flush=True)

    hidden_size = config.hidden_size
    classifier = LogClassifier(model, hidden_size, num_labels=2)
    classifier = classifier.to(device)
    print("Classifier initialized", flush=True)

    normal_count = sum(1 for batch in train_loader for label in batch['labels'] if label == 0)
    anomaly_count = sum(1 for batch in train_loader for label in batch['labels'] if label == 1)
    class_weights = torch.tensor([1.0, normal_count / anomaly_count]).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)
    print("Loss function defined", flush=True)

    optimizer = AdamW(classifier.parameters(), lr=2e-5)
    print("Optimizer defined", flush=True)

    all_preds = []
    all_labels = []

    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
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

        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}", flush=True)

        classifier.eval()
        total_correct = 0
        total_samples = 0
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
        print(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    classifier.eval()
    total_correct = 0
    total_samples = 0

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
    print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return classifier

# 6. 推理函数
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

print("Functions defined", flush=True)

print("Start training", flush=True)
data_dir = "E:\LU\cs_proj\projects\ModernBERT_Log\output"
print("Dir set", flush=True)

train_path = f"{data_dir}/train.pt"
val_path = f"{data_dir}/val.pt"
test_path = f"{data_dir}/test.pt"
print("Dir set2", flush=True)

train_data = load_data(train_path)
val_data = load_data(val_path)
test_data = load_data(test_path)
print("Data loaded", flush=True)

train_dataset = LogDataset(train_data)
val_dataset = LogDataset(val_data)
test_dataset = LogDataset(test_data)
print("Datasets created", flush=True)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print("Dataloaders created", flush=True)

tokenizer = AutoTokenizer.from_pretrained(data_dir)
print("Tokenizer loaded", flush=True)

classifier = train_model(train_loader, val_loader, test_loader)
print("Training completed", flush=True)

log_sequence = ["Receiving block src: dest:", "Receiving block src: dest:"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result = predict_log_sequence(log_sequence, classifier, tokenizer, device=device)
print(f"Sequence {log_sequence} is {result}", flush=True)