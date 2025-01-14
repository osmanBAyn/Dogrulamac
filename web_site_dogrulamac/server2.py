import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TextClassificationPipeline)
import requests
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import torch
from deep_translator import DeeplTranslator
import torch
import torch.nn.functional as F
api_key_deepl = "69f73328-5f95-4eda-813a-16af8c688404:fx"
# Buraya İngilizce modelinizi yazın
model = AutoModelForSequenceClassification.from_pretrained("OsBaran/Roberta-Classification-Model")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Cihazı kontrol et
def predict_with_roberta(model, tokenizer, input_text):
    # Giriş metnini tokenize et ve tensor'a çevir
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    # Model ile tahmin yap
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits'leri al ve tahmin yap
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()  # 0: yanlış, 1: doğru
    return prediction
def explain_roberta_prediction(model, tokenizer, input_text):
    # Tokenize et

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    # Model ile tahmin yap
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits'leri al
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Tahmin sonucunu ve olasılıkları elde et
    predicted_class = torch.argmax(logits, dim=-1).item()
    result = "Doğru" if predicted_class == 1 else "Yanlış"
    explanation = f"Modelin tahmini: {result} (Olasılık: {probabilities[predicted_class]:.2f})\n"

    # Önemli kelimeleri çıkarma (örnek olarak)
    tokenized_input = tokenizer.tokenize(tokenizer.decode(inputs['input_ids'][0]))
    important_tokens = tokenized_input[:10]  # İlk 10 tokeni al
    explanation += "Modelin kararı aşağıdaki anahtar kelimelere dayanıyor:\n" + ', '.join(important_tokens)

    return explanation

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
def score_and_visualize(text):
  prediction = pipe([text])
  print(prediction[0])

  explainer = shap.Explainer(pipe)
  shap_values = explainer([text])
  shap.plots.text(shap_values)

api_key = '764e3b45715b449a8aedb8cd8018dfed'
def fetch_news_from_api(api_key, query, page_size=100):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}'
    response = requests.get(url)

    # API yanıtını kontrol et
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        print(f"Error: {response}")
        return []
def extract_keywords(text, top_n=5):
    # 1. Metni temizleme
    text = re.sub(r'[^\w\s]', '', text.lower())  # Noktalama işaretlerini kaldırma ve küçük harfe çevirme

    # 2. Tokenizasyon
    words = text.split()

    # 3. Durak kelimeleri kaldırma
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS]

    # 4. Anahtar kelimeleri sayma ve en sık geçenleri alma
    keyword_counts = Counter(keywords)
    most_common_keywords = keyword_counts.most_common(top_n)

    return [keyword for keyword, _ in most_common_keywords]



kw_model = KeyBERT('all-mpnet-base-v2')  # SBERT kullanarak modeli yükleyin

def extract_keywords_keybert(text, num_keywords=2):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]


def filter_trusted_sources(articles, trusted_sources):
    trusted_articles = []
    for article in articles:
        source_name = article['source']['name'].lower()  # Kaynağı küçük harfe çevir
        if any(trusted_source in source_name for trusted_source in trusted_sources):
            trusted_articles.append(article)
    return trusted_articles

def fetch_news_content(link):
    response = requests.get(link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Başlık ve içerik çıkarma
        title = soup.find('h1').get_text() if soup.find('h1') else "Başlık bulunamadı"
        content = ' '.join([p.get_text() for p in soup.find_all('p')])
        return title, content
    else:
        print(f"Error fetching content: {response.status_code}")
        return "", ""
def compare_with_thrusted(input_text, bbc_articles):
    texts = [input_text] + [article[1] for article in bbc_articles]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return similarities

from sentence_transformers import SentenceTransformer, util
def sbert_similarity(input_text, bbc_articles):
# SBERT modelini yükleyin
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Kullanıcı metni ve internetten çekilen metinleri vektörize edin
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    news_embeddings = model.encode([news[1] for news in bbc_articles], convert_to_tensor=True)

    # Benzerlikleri hesaplayın
    cosine_scores = util.pytorch_cos_sim(input_embedding, news_embeddings)

    # En yüksek benzerlik skoru ve karşılık gelen haber
    max_score, most_similar_news = cosine_scores.max(), bbc_articles[cosine_scores.argmax().item()]
    print(f"En benzer haber skoru: {max_score:.2f}")

def translate_text(text, source_lang='tr', target_lang='en'):
    translated = DeeplTranslator(api_key=api_key_deepl, source=source_lang, target=target_lang).translate(text)
    return translated
# Türkçe modelini yükle
# model_tr_name = "dbmdz/bert-base-turkish-cased"  # Buraya Türkçe modelinizi yazın
# model_tr = AutoModelForSequenceClassification.from_pretrained(model_tr_name)
# tokenizer_tr = AutoTokenizer.from_pretrained(model_tr_name)
# classifier_tr = pipeline("sentiment-analysis", model=model_tr, tokenizer=tokenizer_tr)

tokenizer_tr = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model_tr = AutoModelForSequenceClassification.from_pretrained("OsBaran/Bert-Classification-Model-Tr-3", num_labels=2)
def trModelPredictAlgo(input_news):
    inputs = tokenizer(input_news, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

# Modelin tahmin yapması
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Softmax uygulama (olasılık hesaplama)
    probabilities = F.softmax(logits, dim=-1)

    # En yüksek olasılığı ve sınıfı bulma
    predicted_class = torch.argmax(probabilities, dim=-1)
    predicted_probability = probabilities[0, predicted_class].item()
    sonuc = 0
    if(predicted_class.item()==0):
        sonuc = "Yanlış"
    else :
        sonuc = "Doğru"
    # Sonucu yazdırma
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Prediction probability: {predicted_probability * 100:.2f}%")
    return f"Dogruluk tahmini: {sonuc}" + f"Tahmin olasılığı: {predicted_probability * 100:.2f}%"
def enModelPredictAlgo(input_news):
    keywords = extract_keywords_keybert(input_news)
    search_query = ' '.join(keywords)
    news_articles = fetch_news_from_api(api_key, search_query)

    trusted_sources = [
            "bbc news",
            "cnn",
            "reuters.com",
            "theguardian.com",
            "time",
            # Diğer güvenilir kaynaklar...
    ]

    trusted_articles = filter_trusted_sources(news_articles, trusted_sources)
        # # Sonuçları yazdır
    trusted_articles_urls = []
    for i in trusted_articles:
        trusted_articles_urls.append(i["url"])

    if trusted_articles:
        print(f"\nGüvenilir kaynaklardan bulunan haberler:\n")
        print(trusted_articles_urls)
        bbc_articles = [fetch_news_content(link) for link in trusted_articles_urls]
        similarities = compare_with_thrusted(input_news, bbc_articles)
        sbert_similarity(input_news, bbc_articles)
        print(similarities)
        max_similarity = max(similarities)
        threshold = 0.8
        if max_similarity > threshold:
            print(f"Sonuç: Doğru (Benzerlik: {max_similarity:.2f})")
        else:
                # Benzerlik bulunmazsa tahmin algoritmasını kullanın ve açıklama sağlayın
            prediction = predict_with_roberta(model, tokenizer, input_news)
            explanation = explain_roberta_prediction(model, tokenizer, input_news)
            # Tahmin sonucunu yazdır
            # result = "Doğru" if prediction == 1 else "Yanlış"
            # print(f"Haberin durumu: {result}")
            print(explanation)
            return explanation
                
    else:
        print("Güvenilir kaynaklardan hiç haber bulunamadı.")
        prediction = predict_with_roberta(model, tokenizer, input_news)
        explanation = explain_roberta_prediction(model, tokenizer, input_news)
            # Tahmin sonucunu yazdır
        result = "Doğru" if prediction == 1 else "Yanlış"
        print(f"Haberin durumu: {result}")
        print("Haberin açıklaması:")
        print(explanation)
        return explanation
# Gradio ile API oluştur
def predict(input_news, language):
    if language == "en":
        result = enModelPredictAlgo(input_news=input_news)
        return {"Sonuç": result}
    elif language == "tr":
        input_news_en= translate_text(input_news)
        result1 = enModelPredictAlgo(input_news_en)
        
        result2= trModelPredictAlgo(input_news=input_news)
        return {"İngilizce Model Sonucu": result1, "Türkçe Model Sonucu": result2}
    else:
        result = {"error": "Unsupported language"}
    # return result

# Arayüz
gr.Interface(fn=predict, 
             inputs=[gr.Textbox(label="Text Input"), gr.Dropdown(["en", "tr"], label="Language")], 
             outputs="json").launch()
