// Required imports
const axios = require('axios');
const cheerio = require('cheerio');
const { TfidfVectorizer } = require('node-tfidf');
const cosineSimilarity = require('compute-cosine-similarity');
const KeyBERT = require('keybert');
const { SentenceTransformer } = require('sentence-transformers');

const apiKey = '764e3b45715b449a8aedb8cd8018dfed';

async function fetchNewsFromAPI(apiKey, query, pageSize = 100) {
    const url = `https://newsapi.org/v2/everything?q=${query}&pageSize=${pageSize}&apiKey=${apiKey}`;
    try {
        const response = await axios.get(url);
        if (response.status === 200) {
            return response.data.articles || [];
        } else {
            console.error("Error fetching news API", response);
            return [];
        }
    } catch (error) {
        console.error("API Request failed", error);
        return [];
    }
}

async function extractKeywordsKeyBERT(text, numKeywords = 2) {
    const kwModel = new KeyBERT('all-mpnet-base-v2');
    const keywords = await kwModel.extractKeywords(text, {
        keyphraseNgramRange: [1, 2],
        stopWords: 'english',
        topN: numKeywords
    });
    return keywords.map(kw => kw[0]);
}

function filterTrustedSources(articles, trustedSources) {
    return articles.filter(article => {
        const sourceName = article.source.name.toLowerCase();
        return trustedSources.some(trustedSource => sourceName.includes(trustedSource));
    });
}

async function fetchNewsContent(link) {
    try {
        const response = await axios.get(link);
        if (response.status === 200) {
            const $ = cheerio.load(response.data);
            const title = $('h1').text() || "Title not found";
            const content = $('p').map((i, el) => $(el).text()).get().join(' ');
            return { title, content };
        } else {
            console.error("Error fetching content", response.status);
            return { title: "", content: "" };
        }
    } catch (error) {
        console.error("Error fetching content", error);
        return { title: "", content: "" };
    }
}

function compareWithTrusted(inputText, bbcArticles) {
    const texts = [inputText, ...bbcArticles.map(article => article.content)];
    const vectorizer = new TfidfVectorizer();
    const vectors = vectorizer.fitTransform(texts);

    const inputVector = vectors[0];
    return vectors.slice(1).map(vector => cosineSimilarity(inputVector, vector));
}

async function sbertSimilarity(inputText, bbcArticles) {
    const model = new SentenceTransformer('sentence-transformers/all-mpnet-base-v2');
    const inputEmbedding = await model.encode(inputText);
    const newsEmbeddings = await Promise.all(bbcArticles.map(article => model.encode(article.content)));

    const similarities = newsEmbeddings.map(newsEmbedding => cosineSimilarity(inputEmbedding, newsEmbedding));
    const maxSimilarity = Math.max(...similarities);
    console.log(`Highest similarity score: ${maxSimilarity.toFixed(2)}`);
    return { maxSimilarity, mostSimilarArticle: bbcArticles[similarities.indexOf(maxSimilarity)] };
}

async function willBeServer(inputNews) {
    const keywords = await extractKeywordsKeyBERT(inputNews);
    const searchQuery = keywords.join(' ');
    const newsArticles = await fetchNewsFromAPI(apiKey, searchQuery);

    const trustedSources = [
        "bbc news",
        "cnn",
        "reuters.com",
        "theguardian.com",
        "time",
        // Add other trusted sources...
    ];

    const trustedArticles = filterTrustedSources(newsArticles, trustedSources);
    const trustedArticlesContent = await Promise.all(trustedArticles.map(article => fetchNewsContent(article.url)));

    const tfidfSimilarities = compareWithTrusted(inputNews, trustedArticlesContent);
    const sbertResults = await sbertSimilarity(inputNews, trustedArticlesContent);

    const maxSimilarity = Math.max(...tfidfSimilarities);
    const threshold = 0.8;

    if (maxSimilarity > threshold) {
        console.log(`Result: True (Similarity: ${maxSimilarity.toFixed(2)})`);
    } else {
        console.log(`Result: False (Max Similarity: ${maxSimilarity.toFixed(2)})`);
    }
}
