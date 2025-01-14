# Oluşturulan Türkçe Veri Setiyle Geniş Dil Modeli Tabanlı Sahte Haber Tespiti Uygulaması: Doğrulamaç
TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması kapsamında hazırlanan _Oluşturulan Türkçe Veri Setiyle Geniş Dil Modeli Tabanlı Sahte Haber Tespiti Uygulaması: Doğrulamaç_ adlı projenin reposudur. 

Oluşturulan web sitesine http://www.dogrulamac.me/ adresinden ulaşılabilir.

# Dosya Düzeni
```
├── web_site_dogrulamac                    
|   ├── public                             # static files folder in web site
|   ├── chat.html                          # 
|   ├── index.html                         # 
|   ├── package-lock.json                  # 
|   ├── package.json                       # 
|   ├── server.js                          # 
|   ├── server.js                          # 
|   ├── server2.js                         # 
|   ├── server2.js                         # 
|   ├── willbeserver.js                    # 
|   └── willbeserver.py                    # 
├── datasets                               
|   ├── English                            
|   |   └── merged_latest_english.csv      # final merged dataset which used in English model training
|   └── Turkish                            
|   |   ├── teyit.csv                      # scrapping dataset from teyit.org
|   |   ├── dogrula.csv                    # scrapping dataset from dogrula.org
|   |   ├── dogruluk_payi.csv              # scrapping dataset from dogruluk_payi.com
|   |   └── merged_latest_turkish.csv      # final merged dataset which used in Turkish model training
├── training                               
|   ├── English                            
|   |   ├── BERT.ipynb                     # training with BERT English
|   |   ├── roBERTA.ipynb                  # training with roBERTA English
|   |   └── gemma-2.ipynb                  # training with Gemma-2
|   ├── Turkish                            
|   |   ├── lojistik_regression.ipynb        # training with Lineer Regression
|   |   ├── naive_bayes.ipynb              # training with Naive Bayes
|   |   ├── XGBoost.ipynb                  # training with XGBoost
|   |   ├── random_forest.ipynb            # training with Random Forest
|   |   ├── SVM.ipynb                      # training with SVM
|   |   ├── LSTM.ipynb                     # training with LSTM
|   |   ├── BERT.ipynb                     # training with BERT Turkish
|   |   └── gemma-2.ipynb                  # training with Gemma-2
```
