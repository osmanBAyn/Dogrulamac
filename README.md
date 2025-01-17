# Oluşturulan Türkçe Veri Setiyle Geniş Dil Modeli Tabanlı Sahte Haber Tespiti Uygulaması: Doğrulamaç
TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması kapsamında hazırlanan _Sahte Haber Tespitine Yönelik Türkçe Veri Seti Oluşturulması ve Geniş Dil Modeli Tabanlı Tespiti: Doğrulamaç_ adlı projenin reposudur. 

Oluşturulan web sitesine http://www.dogrulamac.me/ adresinden ulaşılabilir.

# Dosya Düzeni
__TR__
```
├── web_site_dogrulamac                    
│   ├── public                             # web sitesinde kullanılan statik dosyalar klasörü
│   ├── chat.html                          
│   ├── index.html                         
│   ├── package-lock.json                  
│   ├── package.json                       
│   ├── server.js                          
│   ├── server2.js                         
│   ├── willbeserver.js                    
│   └── willbeserver.py                    
├── datasets                         
│   ├── English                          
│   │   └── merged_latest_english.csv      # İngilizce model eğitimi için kullanılan nihai birleştirilmiş veri kümesi
│   └── Turkish                             
│       ├── teyit.csv                      # teyit.org'dan alınan veri kümesi
│       ├── dogrula.csv                    # dogrula.org'dan alınan veri kümesi
│       ├── dogruluk_payi.csv              # dogruluk_payi.com'dan alınan veri kümesi
│       └── merged_latest_turkish.csv      # Türkçe model eğitimi için kullanılan nihai birleştirilmiş veri kümesi
├── training                                 
│   ├── English                          
│   │   ├── BERT.ipynb                     # İngilizce için BERT ile eğitim
│   │   ├── roBERTA.ipynb                  # İngilizce için roBERTa ile eğitim
│   │   └── gemma-2.ipynb                  # İngilizce için Gemma-2 ile eğitim
│   └── Turkish                             
│       ├── lojistik_regression.ipynb      # Lojistik Regresyon ile eğitim
│       ├── naive_bayes.ipynb              # Naive Bayes ile eğitim
│       ├── XGBoost.ipynb                  # XGBoost ile eğitim
│       ├── random_forest.ipynb            # Rastgele Orman ile eğitim
│       ├── SVM.ipynb                      # SVM ile eğitim
│       ├── LSTM.ipynb                     # LSTM ile eğitim
│       ├── BERT.ipynb                     # Türkçe için BERT ile eğitim
│       └── gemma-2.ipynb                  # Türkçe için Gemma-2 ile eğitim
````

__EN__
```
├── web_site_dogrulamac                    
│   ├── public                             # static files folder in web site
│   ├── chat.html                          
│   ├── index.html                         
│   ├── package-lock.json                  
│   ├── package.json                       
│   ├── server.js                          
│   ├── server2.js                         
│   ├── willbeserver.js                    
│   └── willbeserver.py                    
├── datasets                               
│   ├── English                            
│   │   └── merged_latest_english.csv      # final merged dataset used for English model training
│   └── Turkish                            
│       ├── teyit.csv                      # scrapped dataset from teyit.org
│       ├── dogrula.csv                    # scrapped dataset from dogrula.org
│       ├── dogruluk_payi.csv              # scrapped dataset from dogruluk_payi.com
│       └── merged_latest_turkish.csv      # final merged dataset used for Turkish model training
├── training                               
│   ├── English                            
│   │   ├── BERT.ipynb                     # training with BERT English
│   │   ├── roBERTA.ipynb                  # training with roBERTA English
│   │   └── gemma-2.ipynb                  # training with Gemma-2
│   └── Turkish                            
│       ├── lojistik_regression.ipynb      # training with Logistic Regression
│       ├── naive_bayes.ipynb              # training with Naive Bayes
│       ├── XGBoost.ipynb                  # training with XGBoost
│       ├── random_forest.ipynb            # training with Random Forest
│       ├── SVM.ipynb                      # training with SVM
│       ├── LSTM.ipynb                     # training with LSTM
│       ├── BERT.ipynb                     # training with BERT Turkish
│       └── gemma-2.ipynb                  # training with Gemma-2
```
```
