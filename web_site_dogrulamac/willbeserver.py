from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
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
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
from keybert import KeyBERT
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
api_key = '764e3b45715b449a8aedb8cd8018dfed'
def fetch_news_from_api(api_key, query, page_size=5):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}'
    response = requests.get(url)

    # API yanıtını kontrol et
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        print(f"Error: {response}")
        return []
kw_model = KeyBERT('all-mpnet-base-v2') 
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
    try:
        response = requests.get(link)
    except Exception as e:
        print(f"Error fetching content: {e}")
        return "", ""
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
def sbert_similarity(input_text, bbc_articles):
# SBERT modelini yükleyin
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Kullanıcı metni ve internetten çekilen metinleri vektörize edin
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    news_embeddings = model.encode(bbc_articles, convert_to_tensor=True)

    # Benzerlikleri hesaplayın
    cosine_scores = util.pytorch_cos_sim(input_embedding, news_embeddings)
    return cosine_scores
    # En yüksek benzerlik skoru ve karşılık gelen haber
    max_score, most_similar_news = cosine_scores.max(), bbc_articles[cosine_scores.argmax().item()]
    print(f"En benzer haber skoru: {max_score:.2f}")
@app.route('/similar', methods=['POST'])
def willbeserver():
    data = request.json
    input_news = data.get('input_news', '')
    print(input_news)
    keywords = kw_model.extract_keywords(
        input_news, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5
    )
    print(keywords)
    search_query = ' '.join([kw[0] for kw in keywords])
    news_articles = fetch_news_from_api(api_key, search_query)

    

    trusted_articles_urls = [article["url"] for article in news_articles]

    if trusted_articles_urls:
        bbc_articles = [
            fetch_news_content(link) for link in trusted_articles_urls[:3]
        ]
        
        # similarities = compare_with_thrusted(input_news, [article[1] for article in bbc_articles])
        sbert_result = sbert_similarity(input_news, [article[1] for article in bbc_articles])
        print(sbert_result)
        
        # max_similarity = max(similarities)
        haberveSim = []
        for i in range(len(sbert_result[0])):
            print(sbert_result[0][i])
            haberveSim.append([trusted_articles_urls[i], sbert_result[0][i].item()])
            haberveSim = sorted(haberveSim, key=lambda x: x[1], reverse=True)
            print(haberveSim)

        return jsonify({
            "result": haberveSim
        })
    else:
        return jsonify({
            "result": "No trusted articles found",
        })
if __name__ == '__main__':
    app.run(debug=True)