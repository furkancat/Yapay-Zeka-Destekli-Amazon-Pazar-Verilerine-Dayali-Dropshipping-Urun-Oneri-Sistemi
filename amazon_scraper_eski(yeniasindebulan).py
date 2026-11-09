import requests
from bs4 import BeautifulSoup
import psycopg2
import random
import time
import re
import json
import html as html_module

# -------------------------
# POSTGRESQL BAĞLANTI
# -------------------------
conn = psycopg2.connect(
    host="localhost",
    dbname="dropshipping",
    user="scraper",
    password="guclu_sifre_123"
)
cur = conn.cursor()

# -------------------------
# USER-AGENT HAVUZU
# -------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# -------------------------
# KATEGORİ ÇEVİRİ (İngilizce slug → Türkçe)
# -------------------------
CATEGORY_TRANSLATIONS = {
    "electronics": "Elektronik",
    "kitchen": "Mutfak",
    "beauty": "Kişisel Bakım ve Kozmetik",
    "office-products": "Ofis Ürünleri",
    "toys": "Oyuncak",
    "pet-supplies": "Evcil Hayvan Ürünleri",
    "sports": "Spor ve Outdoor",
    "automotive": "Otomotiv",
    "home": "Ev ve Yaşam",
    "apparel": "Moda",
    "computers": "Bilgisayar",
    "hpc": "Sağlık ve Kişisel Bakım",
    "home-improvement": "Ev ve Yaşam",
    "sporting-goods": "Spor ve Outdoor",
    "books": "Kitap",
    "Oto Parçaları": "Otomotiv",
    "Teşhis ve Sağlık Ölçüm Cihazları": "Sağlık ve Kişisel Bakım",
    "Makyaj Fırçaları ve Aksesuarları": "Kişisel Bakım ve Kozmetik",
    "Otomotiv Parçaları ve Aksesuarları": "Otomotiv",
    "Işıklar, Lambalar ve Göstergeler": "Ev ve Yaşam",
    "Ağız ve Diş Sağlığı": "Sağlık ve Kişisel Bakım",
    "videogames": "Video Oyunları",

    # Türkçe slug'lar (H2 başlığından lower().replace(' ', '-') ile gelen)
    "ev-ve-yaşam": "Ev ve Yaşam",
    "mutfak": "Mutfak",
    "moda": "Moda",
    "ofis-urunleri": "Ofis Ürünleri",
    "elektronik": "Elektronik",
    "kisisel-bakim-ve-kozmetik": "Kişisel Bakım ve Kozmetik",
    "oyuncak": "Oyuncak",
    "evcil-hayvan-urunleri": "Evcil Hayvan Ürünleri",
    "spor-ve-outdoor": "Spor ve Outdoor",
    "otomotiv": "Otomotiv",
}

def translate_category(slug: str) -> str:
    if not slug:
        return "general"
    slug_clean = slug.lower().strip()
    if slug_clean in CATEGORY_TRANSLATIONS:
        return CATEGORY_TRANSLATIONS[slug_clean]
    return slug

# -------------------------
# TÜRKİYE KONUMU COOKIE'LER
# -------------------------
TR_COOKIES = {
    "lc-main": "tr_TR",
    "i18n-prefs": "TRY",
    "sp-cdn": "L5Z9:TR",
    "session-id": "",
    "ubid-main": "",
    "session-token": "",
    "csrf-token": "",
    "skin": "noskin",
}

# -------------------------
# HEADER
# -------------------------
def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com.tr/search?q=amazon+en+çok+satanlar",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

# -------------------------
# ASIN DOĞRULAMA
# -------------------------
def is_valid_asin(asin):
    if not asin or len(asin) != 10:
        return False
    return bool(re.match(r'^B[A-Z0-9]{9}$', asin))

# -------------------------
# FİYAT PARSE (TR formatı)
# -------------------------
def parse_price_text(text):
    if not text:
        return None, None
    try:
        clean = text.replace('₺', '').replace('TL', '').replace('$', '').replace('USD', '').strip()
        match = re.search(r'([\d\.,]+)', clean)
        if not match:
            return text, None
        num_str = match.group(1)
        has_comma = ',' in num_str
        has_dot = '.' in num_str
        if has_comma and has_dot:
            raw = num_str.replace('.', '').replace(',', '.')
            return text, float(raw)
        elif has_comma:
            parts = num_str.split(',')
            if len(parts[-1]) == 2:
                raw = num_str.replace('.', '').replace(',', '.')
                return text, float(raw)
            else:
                raw = num_str.replace(',', '')
                return text, float(raw)
        elif has_dot:
            parts = num_str.split('.')
            if len(parts[-1]) == 2:
                return text, float(num_str)
            else:
                raw = num_str.replace('.', '')
                return text, float(raw)
        else:
            return text, float(num_str)
    except:
        return text, None

# -------------------------
# SAYFA GETİRME (Retry'li)
# -------------------------
def get_page(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.get(
                url,
                headers=get_headers(),
                cookies=TR_COOKIES,
                timeout=20,
                allow_redirects=True
            )
            if r.status_code == 200:
                if "captcha" in r.text.lower() or "robot check" in r.text.lower():
                    print(f"⚠️ Bot algılandı: {url}")
                    time.sleep(random.uniform(10, 15))
                    continue
                time.sleep(random.uniform(2, 4))
                return r.text
            elif r.status_code == 503:
                print(f"⏳ Service Unavailable (503): {url}")
                time.sleep(random.uniform(5, 10))
                continue
            else:
                print(f"⚠️ HTTP {r.status_code}: {url}")
                return None
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout (deneme {attempt + 1}/{max_retries}): {url}")
            time.sleep(random.uniform(3, 6))
        except Exception as e:
            print(f"❌ Hata: {e}")
            return None
    return None

# -------------------------
# JSON + HTML PARSE (Ana mantık)
# -------------------------
def extract_products_from_html(html, source_url):
    soup = BeautifulSoup(html, "html.parser")
    products = []
    
    is_movers = "movers-and-shakers" in source_url
    
    # 1. JSON METADATA parse et (YENİ: data-client-recs-list desteği eklendi)
    json_meta = {}
    
    # Yöntem A: Movers & Shakers / yeni grid sayfaları (data-client-recs-list)
    client_recs_container = soup.find("div", class_="p13n-desktop-grid")
    if client_recs_container and client_recs_container.has_attr("data-client-recs-list"):
        try:
            raw_json = html_module.unescape(client_recs_container["data-client-recs-list"])
            data = json.loads(raw_json)
            for item in data:
                asin = item.get("id", "").strip().upper()
                meta = item.get("metadataMap", {})
                rank = int(meta.get("render.zg.rank", 0))
                if is_movers:
                    bsr_raw = meta.get("render.zg.bsms.currentSalesRank")
                    bsr = int(bsr_raw) if bsr_raw else None
                else:
                    bsr = rank
                
                if is_valid_asin(asin):
                    json_meta[asin] = {"rank": rank, "bsr": bsr}
        except Exception as e:
            print(f"⚠️ data-client-recs-list parse hatası: {e}")
    
    # Yöntem B: Eski carousel yapısı (data-a-carousel-options)
    if not json_meta:
        for carousel_container in soup.find_all("div", attrs={"data-a-carousel-options": True}):
            try:
                raw_json = html_module.unescape(carousel_container["data-a-carousel-options"])
                data = json.loads(raw_json)
                for item_str in data.get("ajax", {}).get("id_list", []):
                    item = json.loads(item_str)
                    asin = item.get("id", "").strip().upper()
                    meta = item.get("metadataMap", {})
                    rank = int(meta.get("render.zg.rank", 0))
                    if is_movers:
                        bsr_raw = meta.get("render.zg.bsms.currentSalesRank")
                        bsr = int(bsr_raw) if bsr_raw else None
                    else:
                        bsr = rank
                    
                    if is_valid_asin(asin):
                        json_meta[asin] = {"rank": rank, "bsr": bsr}
            except Exception:
                continue
    
    # 2. KATEGORİ ÇIKARIMI
    category_map = {}
    h2_list = soup.find_all("h2")
    carousel_list = soup.find_all("div", attrs={"data-a-carousel-options": True})
    
    # Eğer eski carousel yoksa, yeni grid container'ı carousel listesine ekle
    if not carousel_list and client_recs_container:
        carousel_list = [client_recs_container]
    
    for i, carousel in enumerate(carousel_list):
        cat = None
        
        # data-acp-tracking'ten dene
        parent = carousel.find_parent(attrs={"data-acp-tracking": True})
        if parent:
            try:
                tracking = parent["data-acp-tracking"]
                ref_match = re.search(r'"ref_":"([^"]+)"', tracking)
                if ref_match:
                    ref = ref_match.group(1)
                    cat_match = re.search(r'(?:bs_c_|bsms_c_)(.+)', ref)
                    if cat_match:
                        cat = cat_match.group(1)
            except:
                pass
        
        # H2'den dene
        if not cat and i < len(h2_list):
            text = h2_list[i].get_text(strip=True)
            cat_match = re.search(r'^([A-Za-zğüşıöçĞÜŞİÖÇ\s]+?)\s+Kategorisinde', text, re.IGNORECASE)
            if cat_match:
                cat = cat_match.group(1).strip().lower().replace(' ', '-')
        
        # URL'den fallback
        if not cat:
            cat_match = re.search(r'/gp/(?:bestsellers|movers-and-shakers)/([^/?]+)', source_url)
            cat = cat_match.group(1) if cat_match else "general"
        
        category_map[id(carousel)] = cat
    
    # 3. TÜM data-asin KARTLARINI BUL
    all_cards = []
    for carousel in carousel_list:
        cards = carousel.find_all(attrs={"data-asin": True})
        for card in cards:
            card["_category"] = category_map.get(id(carousel), "general")
        all_cards.extend(cards)
    
    # Eğer hiç kart bulunamazsa tüm sayfada ara
    if not all_cards:
        for card in soup.find_all(attrs={"data-asin": True}):
            card["_category"] = "general"
            all_cards.append(card)
    
    # 4. Her kartı parse et
    seen_asins = set()
    for card in all_cards:
        asin = card.get("data-asin", "").strip().upper()
        if not is_valid_asin(asin) or asin in seen_asins:
            continue
        seen_asins.add(asin)
        
        # Rank/BSR: JSON'dan varsa al, yoksa HTML metninden dene
        rank = None
        bsr = None
        if asin in json_meta:
            rank = json_meta[asin]["rank"]
            bsr = json_meta[asin]["bsr"]
        else:
            # HTML'deki "Satış sırası: 325" metninden BSR çıkar
            rank_meta = card.select_one('span._cDEzb_zg-grid-rank-metadata_33jPv')
            if rank_meta:
                txt = rank_meta.get_text()
                sales_match = re.search(r'Satış\s+sırası:\s*([\d\.]+)', txt)
                if sales_match:
                    bsr = int(sales_match.group(1).replace('.', ''))
        
        # Kategori
        category = card.get("_category")
        if not category or category == "general":
            cat_match = re.search(r'/gp/(?:bestsellers|movers-and-shakers)/([^/?]+)', source_url)
            category = cat_match.group(1) if cat_match else "general"
        category = translate_category(category)
        
        p = {
            "asin": asin,
            "source_url": source_url,
            "category": category,
            "rank": rank,
            "bsr": bsr,
            "title": None,
            "price_raw": None,
            "price_num": None,
            "rating": None,
            "review_count": None,
            "image_url": None,
        }
        
        # Title - GÜNCELLENDİ (yeni class eklendi)
        try:
            # Yeni Movers & Shakers / grid title class'ı
            title_div = card.select_one('div._cDEzb_p13n-sc-css-line-clamp-3_g3dy1')
            if title_div:
                p["title"] = title_div.get_text(strip=True)
            else:
                # Eski class'lar veya link text fallback
                title_link = card.select_one('a[role="link"]')
                if title_link:
                    title_div = title_link.select_one('div.p13n-sc-truncate, div.p13n-sc-truncate-desktop-type2')
                    if title_div:
                        p["title"] = title_div.get_text(strip=True)
                    else:
                        txt = title_link.get_text(strip=True)
                        if len(txt) > 5:
                            p["title"] = txt
        except:
            pass
        
        # Price
        try:
            # Direkt yeni price class'ını dene
            price_span = card.select_one('span._cDEzb_p13n-sc-price_3mJ9Z')
            if price_span:
                p["price_raw"], p["price_num"] = parse_price_text(price_span.get_text())
            else:
                price_container = card.select_one('span.a-color-price')
                if price_container:
                    price_span = price_container.select_one('span[class*="p13n-sc-price"]')
                    if price_span:
                        p["price_raw"], p["price_num"] = parse_price_text(price_span.get_text())
        except:
            pass
        
        # Rating & Review Count - GÜNCELLENDİ (daha kesin selector)
        try:
            # Yöntem 1: aria-label'dan rating ve review
            rating_link = card.select_one('a[aria-label*="yıldız üzerinden"]')
            if rating_link:
                aria = rating_link.get("aria-label", "")
                rating_match = re.search(r'(\d+,\d+)', aria)
                if rating_match:
                    p["rating"] = float(rating_match.group(1).replace(',', '.'))
                review_match = re.search(r'([\d\.]+)\s*puan', aria)
                if review_match:
                    review_str = review_match.group(1).replace('.', '').replace(',', '')
                    p["review_count"] = int(review_str)
            
            # Yöntem 2: a-icon-alt'dan rating
            if p["rating"] is None:
                icon_alt = card.select_one('span.a-icon-alt')
                if icon_alt:
                    txt = icon_alt.get_text()
                    m = re.search(r'(\d+,\d+)', txt)
                    if m:
                        p["rating"] = float(m.group(1).replace(',', '.'))
            
            # Yöntem 3: Review count - önce rating link içindeki a-size-small'ı dene
            if p["review_count"] is None:
                review_span = card.select_one('a[aria-label*="yıldız üzerinden"] span.a-size-small')
                if review_span:
                    txt = review_span.get_text(strip=True)
                    nums = re.findall(r'[\d\.]+', txt)
                    if nums:
                        p["review_count"] = int(nums[0].replace('.', '').replace(',', ''))
                else:
                    # Genel a-size-small (sayısal değerse)
                    small = card.select_one('span.a-size-small')
                    if small:
                        txt = small.get_text(strip=True)
                        if txt.replace('.', '').replace(',', '').isdigit():
                            p["review_count"] = int(txt.replace('.', '').replace(',', ''))
        except:
            pass

        # Image URL
        try:
            img = card.select_one('img')
            if img:
                src = img.get('data-src') or img.get('src')
                if src and src.startswith('http'):
                    p["image_url"] = src
        except:
            pass
        
        products.append(p)
    
    return products

# -------------------------
# VERİTABANINA KAYDET
# -------------------------
# -------------------------
# VERİTABANINA KAYDET - GÜNCELLENDİ
# -------------------------
def save_products(products):
    yeni = 0
    guncel = 0
    hata = 0
    
    for p in products:
        try:
            # Önce bu ASIN daha önce işlenmiş mi kontrol et
            cur.execute("SELECT processed FROM asin_pool_tr WHERE asin = %s", (p["asin"],))
            sonuc = cur.fetchone()
            zaten_islenmis = sonuc is not None and sonuc[0] is True
            
            # 1. products tablosu
            # Eğer mevcut kayıtta hiç NULL yoksa güncelleme yapma
            cur.execute("""
                INSERT INTO products (asin, title, brand, category, image_url, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (asin) DO UPDATE 
                SET title = EXCLUDED.title, 
                    brand = EXCLUDED.brand, 
                    category = EXCLUDED.category, 
                    image_url = EXCLUDED.image_url
                WHERE products.title IS NULL 
                   OR products.brand IS NULL 
                   OR products.category IS NULL 
                   OR products.image_url IS NULL
            """, (p["asin"], p["title"], None, p["category"], p["image_url"]))
            
            # 2. product_metrics_tr (her zaman YENİ kayıt - zaman serisi için gerekli)
            # Bugün bu ASIN için zaten kayıt var mı kontrol et
            cur.execute("""
                SELECT 1 FROM product_metrics_tr 
                WHERE asin = %s AND scraped_at::date = CURRENT_DATE
                LIMIT 1
            """, (p["asin"],))
            bugun_kayitli = cur.fetchone() is not None
            
            if not bugun_kayitli:
                cur.execute("""
                    INSERT INTO product_metrics_tr (asin, price, rating, review_count, bestseller_rank, scraped_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (p["asin"], p["price_num"], p["rating"], p["review_count"], p["bsr"]))
                        
            # 3. asin_pool_tr - SADECE hiç yoksa veya processed=FALSE ise
            if not zaten_islenmis:
                cur.execute("""
                    INSERT INTO asin_pool_tr (asin, source_category, processed)
                    VALUES (%s, %s, TRUE)
                    ON CONFLICT (asin) DO UPDATE 
                    SET processed = TRUE
                    WHERE asin_pool_tr.processed IS DISTINCT FROM TRUE
                """, (p["asin"], p["source_url"]))
                yeni += 1
                durum = "✅ YENİ"
            else:
                guncel += 1
                durum = "🔄 GÜNCEL"
            
            conn.commit()
            
            fiyat_str = f"₺{p['price_num']}" if p['price_num'] else "N/A"
            bsr_str = f"#{p['bsr']}" if p['bsr'] else "N/A"
            print(f"   {durum} {p['asin']} | Sıra:#{p['rank']} | BSR:{bsr_str} | {fiyat_str} | ⭐{p['rating']} | {p['review_count']} yorum")
            
        except Exception as e:
            print(f"   ❌ DB Hatası {p['asin']}: {e}")
            conn.rollback()
            hata += 1
    
    print(f"   📊 Yeni: {yeni} | Güncel: {guncel} | Hata: {hata}")
    return yeni + guncel

# -------------------------
# KATEGORİLER (New Releases YOK)
# -------------------------
TR_CATEGORIES = [
    # Genel en çok satanlar
    "https://www.amazon.com.tr/gp/bestsellers",
    
    # Ana kategoriler
    "https://www.amazon.com.tr/gp/bestsellers/electronics",
    "https://www.amazon.com.tr/gp/bestsellers/kitchen",
    "https://www.amazon.com.tr/gp/bestsellers/beauty",
    "https://www.amazon.com.tr/gp/bestsellers/pet-supplies",
    "https://www.amazon.com.tr/gp/bestsellers/sports",
    "https://www.amazon.com.tr/gp/bestsellers/automotive",
    "https://www.amazon.com.tr/gp/bestsellers/toys",
    "https://www.amazon.com.tr/gp/bestsellers/home",
    "https://www.amazon.com.tr/gp/bestsellers/office-products",
    
    # Hızla yükselenler (Movers & Shakers)
    "https://www.amazon.com.tr/gp/movers-and-shakers",
    "https://www.amazon.com.tr/gp/movers-and-shakers/electronics",
    "https://www.amazon.com.tr/gp/movers-and-shakers/kitchen",
    "https://www.amazon.com.tr/gp/movers-and-shakers/beauty",
    "https://www.amazon.com.tr/gp/movers-and-shakers/pet-supplies",
    "https://www.amazon.com.tr/gp/movers-and-shakers/sports",
]

def run_asin_collector(max_per_category=50):
    print("🚀 AMAZON.COM.TR JSON + HTML Toplayıcı Başlıyor...")
    print(f"📦 {len(TR_CATEGORIES)} kategori taranacak (New Releases YOK)")
    print("=" * 60)
    
    total_added = 0
    
    for i, url in enumerate(TR_CATEGORIES, 1):
        print(f"\n[{i}/{len(TR_CATEGORIES)}] Taranıyor: {url}")
        
        try:
            html = get_page(url)
            if not html:
                continue
            
            products = extract_products_from_html(html, url)
            products_to_save = products[:max_per_category]
            
            added = save_products(products_to_save)
            total_added += added
            
            print(f"   📊 Bulunan: {len(products)} | Kaydedilen: {added}")
            
            if i < len(TR_CATEGORIES):
                delay = random.uniform(5, 10)
                print(f"   ⏱️ Sonraki kategori için {delay:.1f}s bekleniyor...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"   ❌ Kategori hatası: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"🏁 TOPLAM: {total_added} ürün kaydedildi")
    print("💡 Veriler direkt products + product_metrics_tr + asin_pool_tr'e yazıldı.")
    print("💡 product_scraper_tr.py artık sadece 'processed=FALSE' kalan ASIN olursa çalışır.")
    
    cur.close()
    conn.close()

# ÇALIŞTIR
if __name__ == "__main__":
    run_asin_collector(max_per_category=100)