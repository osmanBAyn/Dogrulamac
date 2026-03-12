import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

dataset = load_dataset("OsBaran/try")
df_full = dataset['train'].to_pandas()
df_full.drop_duplicates(inplace = True, subset = ['text'])

df_train_val, df_test = train_test_split(
    df_full,
    test_size=8000, 
    random_state=42, 
    stratify=df_full['label'] 
)
df_cleaned = df_train_val

df_train, df_val = train_test_split(
    df_train_val,
    test_size=8000,
    random_state=42,
    stratify=df_train_val['label']
)

print(f"Train Boyutu: {len(df_train)}")   
print(f"Val Boyutu: {len(df_val)}") 
print(f"Test Boyutu: {len(df_test)}")     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def prepare_hf_dataset(df):
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    ds = ds.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=256), batched=True)
    cols_to_keep = ['input_ids', 'attention_mask', 'label']
    cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
    return ds.remove_columns(cols_to_remove)

hf_cleaned = prepare_hf_dataset(df_cleaned)
hf_test = prepare_hf_dataset(df_test)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', pos_label=1)
    recall = recall_score(labels, predictions, average='binary', pos_label=1)
    f1 = f1_score(labels, predictions, average='binary', pos_label=1)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_cleaned = df_cleaned['label'].values
y_test_true = df_test['label'].values

fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
fold_cms = [] 

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_cleaned)), y_cleaned)):
    print(f"\n FOLD {fold + 1} \n")

    train_dataset = hf_cleaned.select(train_idx)
    val_dataset = hf_cleaned.select(val_idx)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir=f"./bert_fold_{fold+1}",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=500,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        dataloader_num_workers = 2,
        dataloader_persistent_workers = True,
        dataloader_pin_memory = True,
        dataloader_prefetch_factor = 2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(hf_test)
    y_pred = np.argmax(predictions.predictions, axis=-1)

    acc = accuracy_score(y_test_true, y_pred)
    prec = precision_score(y_test_true, y_pred, average='binary', pos_label=1)
    rec = recall_score(y_test_true, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_test_true, y_pred, average='binary', pos_label=1)

    fold_metrics['accuracy'].append(acc)
    fold_metrics['precision'].append(prec)
    fold_metrics['recall'].append(rec)
    fold_metrics['f1'].append(f1)
    cm = confusion_matrix(y_test_true, y_pred, labels=[0, 1])
    fold_cms.append(cm)

    print(f"Fold {fold + 1} Sonuçları: Acc: %{acc*100:.2f} | F1: %{f1*100:.2f}")
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

alpha = 0.05
n = len(fold_metrics['accuracy'])
t_critical = stats.t.ppf(1 - alpha/2, df=n-1)

for metric_name, values in fold_metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    se = std_val / np.sqrt(n)
    margin_of_error = t_critical * se
    print(f"\n{metric_name.capitalize()}: Mean %{mean_val*100:.2f} | CI [%{(mean_val-margin_of_error)*100:.2f} - %{(mean_val+margin_of_error)*100:.2f}]")

fig, axes = plt.subplots(1, 5, figsize=(25, 4))
labels_str = ['Real (0)', 'Fake (1)']
for i, cm in enumerate(fold_cms):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_str, yticklabels=labels_str, ax=axes[i], cbar=False)
    axes[i].set_xlabel('Tahmin Edilen')
    axes[i].set_ylabel('Gerçek')
    axes[i].set_title(f'Fold {i + 1} - df_test Matrisi')

plt.tight_layout()
plt.show()