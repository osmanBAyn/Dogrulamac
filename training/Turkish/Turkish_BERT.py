import pandas as pd
import numpy as np
import re
from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import copy

ds = load_dataset('OsBaran/trSahteHaber1', split='train')
df = ds.to_pandas()
df.drop_duplicates(inplace=True, subset='text')
df = df.dropna(subset=['text', 'label'])

def bert_preprocessing(text):
    if not isinstance(text, str):
        return ""
    text = re.split(r'haberin sonu', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text'].apply(bert_preprocessing)
df.drop('text', inplace=True, axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 16
epochs = 3
model_name = "dbmdz/bert-base-turkish-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

X = np.array(df['cleaned_text'].tolist())
y = np.array(df['label'].tolist())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_texts = X[train_idx].tolist()
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val_texts = X[val_idx].tolist()
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

    train_encodings = tokenizer(X_train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    val_encodings = tokenizer(X_val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=2, persistent_workers = True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device, non_blocking=True)
                attention_mask = batch[1].to(device, non_blocking=True)
                labels = batch[2].to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device, non_blocking=True)
            attention_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = (torch.sigmoid(outputs.logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, zero_division=0)
    rec = recall_score(all_true, all_preds, zero_division=0)
    f1 = f1_score(all_true, all_preds, zero_division=0)

    print(f"Fold {fold+1} - Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    if fold == 0:
        cm = confusion_matrix(all_true, all_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Gerçek (0)', 'Sahte (1)'],
                    yticklabels=['Gerçek (0)', 'Sahte (1)'])
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek Değer')
        plt.title('Fold 1 - Confusion Matrix')
        plt.show()