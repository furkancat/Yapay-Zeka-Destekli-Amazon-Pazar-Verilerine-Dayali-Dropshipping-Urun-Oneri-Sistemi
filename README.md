# Yapay Zeka Destekli Amazon Pazar Verilerine Dayalı Dropshipping Ürün Öneri Sistemi

Bu proje, lisans **Bitirme Projesi** kapsamında geliştirilmiş, Amazon platformunda dropshipping (stoksuz satış) yapan girişimciler için veri odaklı kararlar almayı sağlayan yapay zeka destekli bir e-ticaret karar destek sistemidir.

Proje kapsamında Amazon.com.tr üzerinden günlük olarak veri toplayan bir web scraping botu, toplanan zaman serisi verilerini depolayan bir PostgreSQL veritabanı, ürünlerin 7 gün sonraki Çok Satanlar Sıralamasını (BSR) tahmin eden Ensemble Makine Öğrenmesi modeli ve bu tahminleri son kullanıcıya sunan bir web arayüzü geliştirilmiştir.

## 🏗️ Sistem Mimarisi

Sistem uçtan uca 3 ana modülden oluşmaktadır:

1. **Veri Toplama (Web Scraping):** Playwright ve BeautifulSoup kullanılarak Amazon'dan dinamik olarak fiyat, BSR, yorum sayısı ve yıldız puanı toplanır.

2. **Makine Öğrenmesi (Modelleme):** Toplanan veriler zaman serisine dönüştürülür. Ağaç tabanlı modeller ve derin öğrenme kullanılarak 7 günlük BSR tahmini yapılır.

3. **Web Arayüzü (FastAPI):** Kullanıcılar kategorilerdeki ürünleri inceler. Sistem, mevcut BSR ile tahmin edilen BSR arasındaki farkı kullanarak ürünlerin beklenen sıralama değişimlerini değerlendirir ve kullanıcıların ürünleri karşılaştırmasına yardımcı olur.

## 📊 Model Performansı ve Sonuçlar

Ürünlerin 7 gün sonraki çok satanlar sıralamasını (BSR) tahmin etmek için XGBoost, CatBoost, LightGBM ve LSTM modelleri eğitilmiştir. Temel modellerin zayıf yönlerini dengelemek amacıyla modellerin çıktıları **Ridge Regresyon tabanlı Stacking (Yığınlama) Ensemble** yöntemiyle birleştirilmiştir.

Test verisi üzerinde yapılan değerlendirme sonucunda, Ensemble modelin temel modellere kıyasla hem Hata (MAE) metriklerinde iyileşme sağladığı hem de BSR varyansını açıklama gücünü (R²) artırdığı gözlemlenmiştir.

| Metrik | XGBoost | CatBoost | LightGBM | LSTM | Ensemble (Stacking) | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **BSR MAE** | 1029.93 | 1031.26 | 1042.94 | 1212.09 | **960.10** | 
| **Log-BSR MAE** | 0.5862 | 0.5694 | 0.5848 | 0.6905 | **0.5571** | 
| **Log-BSR R²** | 0.8524 | 0.8591 | 0.8544 | 0.8175 | **0.8626** | 

*Sonuçlar incelendiğinde, Stacking yaklaşımının en düşük hata oranına (960.10 BSR MAE) ve en yüksek açıklanabilirlik oranına (0.8626 R²) ulaştığı görülmektedir.*

## 🛠️ Kullanılan Teknolojiler

* **Veri Toplama:** Python, Playwright, BeautifulSoup, Pandas

* **Veritabanı:** PostgreSQL, SQLAlchemy, Psycopg2

* **Makine Öğrenmesi:** XGBoost, CatBoost, LightGBM, TensorFlow/Keras (LSTM), Scikit-Learn

* **Backend & Web API:** FastAPI, Uvicorn, JWT Authentication, Jinja2

* **Frontend:** HTML5, CSS3, Bootstrap 5, HTMX

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Depoyu Klonlayın:**

```
git clone https://github.com/furkancat/Yapay-Zeka-Destekli-Amazon-Pazar-Verilerine-Dayali-Dropshipping-Urun-Oneri-Sistemi.git
cd Yapay-Zeka-Destekli-Amazon-Pazar-Verilerine-Dayali-Dropshipping-Urun-Oneri-Sistemi

```

**2. Gerekli Kütüphaneleri Yükleyin:**

```
pip install -r requirements.txt

```

**3. Playwright Tarayıcılarını Kurun (Scraping için):**

```
playwright install

```

**4. Veritabanı Bağlantısını Ayarlayın:**
`database.py` dosyası içerisindeki `VERITABANI_URL` bilgisini kendi PostgreSQL bilgilerinize göre güncelleyin.

**5. Web Sunucusunu Başlatın:**

```
uvicorn main:app --reload

```

Uygulama `http://localhost:8000` adresinde çalışmaya başlayacaktır.

*Bu proje lisans bitirme çalışması olarak geliştirilmiş olup, tüm hakları saklıdır.*
