from datasets import load_dataset
import pandas as pd
import numpy as np
import re
import torch
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from collections import Counter
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ds = load_dataset('OsBaran/trSahteHaber1', split = 'train')
df = ds.to_pandas()
df.drop_duplicates(inplace = True, subset = 'text')
df = df.dropna(subset = ['text', 'label'])

def lstm_preprocessing(text):
  if not isinstance(text, str):
    return ""
  text = text.lower()
  text = re.split(r'haberin sonu', text, flags = re.IGNORECASE)[0]
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
  text = re.sub(r'[^a-zçğıöşü\s]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  return text
df['cleaned_text'] = df['text'].apply(lstm_preprocessing)
df.drop('text', inplace = True, axis = 1)

max_len = 512
vocab_size = 20000
batch_size = 32
all_words = []
for text in df['cleaned_text']:
  all_words.extend(text.split())
word_counts = Counter(all_words)
vocab = {word: i + 2 for i, (word, _) in enumerate(word_counts.most_common(vocab_size))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1
vocab_size = len(vocab)
def pad_and_truncate(text):
  tokens = text.split()
  seq = [vocab.get(word, 1) for word in tokens] #vocabte yoksa unk = 1
  if len(seq) > max_len:
    return seq[:max_len]
  else:
    return seq + [0] * (max_len - len(seq))
df['seqs'] = df['cleaned_text'].apply(pad_and_truncate)
df.drop('cleaned_text', inplace = True, axis = 1)
df = df.sample(frac = 1, random_state = 42)

class LSTMModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 64, padding_idx = 0)
    self.lstm1 = nn.LSTM(64, 64, bidirectional = True, batch_first = True)
    self.lstm2 = nn.LSTM(128, 64, bidirectional = True, batch_first = True)
    self.fc1 = nn.Linear(128, 64)
    self.fc2 = nn.Linear(64, 1)
    self.dropout = nn.Dropout(0.45)
    self.relu = nn.ReLU()
  def forward(self, x):
    x = self.embedding(x)
    output, (hidden, cell) = self.lstm1(x)
    output = self.dropout(output)
    output, (hidden, cell) = self.lstm2(output)
    x, _ = torch.max(output, dim=1) # dimensionum [batch_size, 128] oldu
    x = self.dropout(x)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# print(device)

batch_size = 32
epochs = 10

X = np.array(df['seqs'].tolist())
y = np.array(df['label'].tolist())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train = torch.tensor(X[train_idx], dtype=torch.long)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X[val_idx], dtype=torch.long)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=2)
    
    model = LSTMModel(vocab_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_state)
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            outputs = model(x_batch)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_batch.cpu().numpy())
            
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

