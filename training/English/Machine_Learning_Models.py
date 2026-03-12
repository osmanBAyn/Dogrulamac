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

X_train_text = df_train['text']
y_train = df_train['label']
X_val_text = df_val['text']
y_val = df_val['label']
X_test_text = df_test['text']
y_test = df_test['label']

cumulative_cm = {name: np.zeros((2, 2), dtype=int) for name in models.keys()}
X_test_tfidf = vectorizer.transform(df_test['text'])
y_test = df_test['label']

for name, model in models.items():
    y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cumulative_cm[name] += cm
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Model: {name}")
    print(f"  Accuracy  : %{acc * 100:.2f}")
    print(f"  Precision : %{precision * 100:.2f}")
    print(f"  Recall    : %{recall * 100:.2f}")
    print(f"  F1-Score  : %{f1 * 100:.2f}")
    print("-" * 30)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
labels = ['Real (0)', 'Fake (1)']
for idx, name in enumerate(models.keys()):
    cm = cumulative_cm[name]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[idx])
    axes[idx].set_xlabel('Tahmin Edilen')
    axes[idx].set_ylabel('Gerçek')
    axes[idx].set_title(f'{name} (Test Toplamı)')

plt.tight_layout()
plt.show()


X = df_cleaned['text']   
y = df_cleaned['label'] 

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(dual=False),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
}

model_accuracies = {name: [] for name in models.keys()}
cumulative_cm = {name: np.zeros((2, 2), dtype=int) for name in models.keys()}

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"🔄 FOLD {fold + 1}\n")
    X_train_text, X_test_text = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        model_accuracies[name].append(acc)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cumulative_cm[name] += cm

        print(f" {name:20s} -> Fold {fold + 1} Başarısı: %{acc*100:.2f}")


alpha = 0.05 
for name in models.keys():
    accuracies = model_accuracies[name]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1) 

    n = len(accuracies)
    se = std_acc / np.sqrt(n) 
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_critical * se

    ci_lower = mean_acc - margin_of_error
    ci_upper = mean_acc + margin_of_error

    print(f"\n Model: {name}")
    print(f"   Ortalama Başarı (Mean)  : %{mean_acc*100:.2f}")
    print(f"   Standart Sapma (Std)    : %{std_acc*100:.2f}")
    print(f"   %95 Güven Aralığı (CI)  : [%{ci_lower*100:.2f} - %{ci_upper*100:.2f}]")


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

labels = ['Real (0)', 'Fake (1)']

for idx, name in enumerate(models.keys()):
    cm = cumulative_cm[name]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[idx])
    axes[idx].set_xlabel('Tahmin Edilen')
    axes[idx].set_ylabel('Gerçek')
    axes[idx].set_title(f'{name} (5 Fold Toplamı)')

plt.tight_layout()
plt.show()