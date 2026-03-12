from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ds = load_dataset("OsBaran/trSahteHaber1")
df = pd.DataFrame(ds['train'])
df = df.dropna(subset=['text', 'label'])
df.drop_duplicates(inplace = True, subset = 'text')
print(df.head()()

def haberin_sonunu_al(text):
    if not isinstance(text, str):
        return ""
    
    bolunmus_metin = re.split(r'haberin sonu', text, flags=re.IGNORECASE)
    ust_kisim = bolunmus_metin[0].strip()
    return ust_kisim

df['hedef_metin'] = df['text'].apply(haberin_sonunu_al)

nltk.download('stopwords')
turkce_stop_words = set(stopwords.words('turkish'))
ekstra_silinecekler = {'bir', 'olan', 'olarak', 'sonra', 'kadar', 'da', 'de', 'ki', 've', 'ile'}
turkce_stop_words.update(ekstra_silinecekler)

def clean_for_ml(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zçğıöşü\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()    
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in turkce_stop_words and len(word) > 2]
    return " ".join(cleaned_tokens)

df['cleaned_text'] = df['text'].apply(clean_for_ml)
print("ORİJİNAL :", df['text'].iloc[0][:150], "...")
print("TEMİZ    :", df['cleaned_text'].iloc[0][:150], "...")

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
df.drop_duplicates('cleaned_text', inplace = True)
df.drop('text', inplace = True, axis = 1)

print(df['label'].value_counts())

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state = 42, class_weight = 'balanced'),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(dual=False, class_weight = 'balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight = 'balanced'),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs = -1)
}

metrics_dict = {
    model_name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} 
    for model_name in models.keys()
}

fold_0_cms = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = df['cleaned_text']
y = df['label']

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"\n FOLD {fold + 1}\n")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]    
    current_vectorizer = clone(vectorizer)
    X_train_tfidf = current_vectorizer.fit_transform(X_train)
    X_val_tfidf = current_vectorizer.transform(X_val)
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)        
        y_pred = model.predict(X_val_tfidf)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        
        metrics_dict[name]['accuracy'].append(acc)
        metrics_dict[name]['precision'].append(prec)
        metrics_dict[name]['recall'].append(rec)
        metrics_dict[name]['f1'].append(f1)
        
        print(f"{name:20s} -> Acc: %{acc*100:.2f} | Prec: %{prec*100:.2f} | Rec: %{rec*100:.2f} | F1: %{f1*100:.2f}")
        
        if fold == 0:
            fold_0_cms[name] = confusion_matrix(y_val, y_pred, labels=[0, 1])

fig, axes = plt.subplots(1, 5, figsize=(25, 4))
labels_str = ['Real (0)', 'Fake (1)']

for i, (name, cm) in enumerate(fold_0_cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_str, yticklabels=labels_str, ax=axes[i], cbar=False)
    axes[i].set_xlabel('Tahmin Edilen')
    axes[i].set_ylabel('Gerçek')
    axes[i].set_title(f'{name}\n(Fold 1)')

plt.tight_layout()
plt.show()

alpha = 0.05
n = 5
t_critical = stats.t.ppf(1 - alpha/2, df=n-1) 
for name in models.keys():
    print(f"\n MODEL: {name.upper()}\n")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        values = metrics_dict[name][metric]
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)      
        se = std_val / np.sqrt(n)
        margin_of_error = t_critical * se
        ci_lower = mean_val - margin_of_error
        ci_upper = mean_val + margin_of_error
        
        print(f"🔹 {metric.capitalize():10s} : "
              f"Ort: %{mean_val*100:.2f} | "
              f"Std: %{std_val*100:.2f} | "
              f"%95 CI: [%{ci_lower*100:.2f} - %{ci_upper*100:.2f}]")