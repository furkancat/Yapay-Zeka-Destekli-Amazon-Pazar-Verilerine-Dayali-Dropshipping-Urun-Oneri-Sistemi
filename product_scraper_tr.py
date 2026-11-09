import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import asyncio
try:
    loop = asyncio.get_running_loop()
    if loop:
        print("⚠️ Aktif asyncio loop tespit edildi, yeni loop oluşturuluyor...")
        asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    pass  # Loop yoksa devam et

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import requests
from bs4 import BeautifulSoup
import psycopg2
import psycopg2.extras
import re
import time
import random
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

# -------------------------
# USER-AGENT HAVUZU
# -------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# -------------------------
# TÜRKİYE KONUMU COOKIE'LER
# -------------------------
TR_COOKIES = {
    "lc-main": "tr_TR",
    "i18n-prefs": "TRY",
    "sp-cdn": "L5Z9:TR",
    "skin": "noskin",
}

@dataclass
class AmazonProduct:
    asin: str
    title: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    price_raw: Optional[str] = None
    price_numeric: Optional[float] = None
    currency: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    bestseller_rank: Optional[int] = None
    image_url: Optional[str] = None
    availability: Optional[str] = None
    timestamp: str = ""
    zip_code: str = "10001"
    error: Optional[str] = None

class AmazonScraper:
    def __init__(self, db_config: dict, headless: bool = False, proxy: Optional[str] = None):
        self.db_config = db_config
        self.headless = headless
        self.proxy = proxy
        self.conn = None
        self.cur = None
        self.playwright = None
        self.browser = None
        self.context = None
        self.session = requests.Session()
        self.request_count = 0
        
        self._setup_session_headers()
        self._apply_tr_cookies()
        self._connect_db()
    
    def _setup_session_headers(self):
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Site": "none",
            "Upgrade-Insecure-Requests": "1",
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "max-age=0",
        }
        self.session.headers.update(headers)
    
    def _apply_tr_cookies(self):
        for name, value in TR_COOKIES.items():
            if value:
                self.session.cookies.set(name, value, domain=".amazon.com.tr")
    
    def _refresh_identity(self):
        print(f"   🔄 Kimlik yenileniyor (Yeni UA + Cookies)...")
        self._setup_session_headers()
        print(f"      🎭 Yeni User-Agent: {self.session.headers['User-Agent']}")
        self._apply_tr_cookies()
        time.sleep(random.uniform(1, 3))

    def _clear_dynamic_cookies(self):
        dynamic = {"session-id", "session-id-time", "ubid-main", "session-token"}
        for name in list(self.session.cookies.keys()):
            if name not in dynamic:
                self.session.cookies.pop(name)
        self._apply_tr_cookies()

    def _connect_db(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            print("✅ Veritabanına bağlandı")
        except Exception as e:
            print(f"❌ DB Hatası: {e}")
            raise

    def get_unprocessed_asins(self, limit: int = 5) -> List[str]:
        try:
            self.cur.execute("""
                SELECT asin FROM asin_pool_tr 
                WHERE processed = FALSE 
                ORDER BY created_at ASC 
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (limit,))
            return [row['asin'] for row in self.cur.fetchall()]
        except Exception as e:
            print(f"❌ ASIN hatası: {e}")
            return []

    def mark_as_processed(self, asin: str, success: bool = True, error_msg: str = None):
        try:
            self.cur.execute("UPDATE asin_pool_tr SET processed = TRUE WHERE asin = %s", (asin,))
            self.cur.execute("""
                INSERT INTO scrape_logs (status, message, created_at)
                VALUES (%s, %s, NOW())
            """, ('success' if success else 'error', f"ASIN {asin}: {'OK' if success else error_msg}"))
            self.conn.commit()
            print(f"      📝 asin_pool_tr processed=TRUE yapıldı: {asin}")
        except Exception as e:
            print(f"      ❌ mark_as_processed HATASI: {e}")
            self.conn.rollback()

    def save_product_data(self, product: AmazonProduct):
        try:
            # 1. products tablosuna kaydet
            # Eğer mevcut kayıtta title, brand, category, image_url hepsi doluysa güncelleme yapma
            self.cur.execute("""
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
            """, (product.asin, product.title, product.brand, product.category, product.image_url))
            
            if self.cur.rowcount == 0:
                print(f"      ⏭️ products atlandı (zaten tam dolu): {product.asin}")
            else:
                print(f"      💾 products tablosuna kaydedildi: {product.asin}")

            # 2. product_metrics_tr - SADECE bugün hiç kayıt yoksa yaz
            self.cur.execute("""
                SELECT 1 FROM product_metrics_tr 
                WHERE asin = %s AND scraped_at::date = CURRENT_DATE
                LIMIT 1
            """, (product.asin,))
            bugun_var = self.cur.fetchone() is not None

            # En az bir metrik değeri var mı? (Hepsi None ise metrics tablosuna yazma)
            has_metrics = any([
                product.price_numeric is not None,
                product.rating is not None,
                product.review_count is not None,
                product.bestseller_rank is not None
            ])

            if not bugun_var and has_metrics:
                self.cur.execute("""
                    INSERT INTO product_metrics_tr (asin, price, rating, review_count, bestseller_rank, scraped_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (product.asin, product.price_numeric, product.rating, 
                      product.review_count, product.bestseller_rank))
                print(f"      💾 product_metrics_tr tablosuna kaydedildi: {product.asin} (bugün ilk kez)")
            elif bugun_var:
                print(f"      ⏭️ product_metrics_tr atlandı: {product.asin} (bugün zaten kayıtlı)")
            else:
                print(f"      ⚠️ product_metrics_tr atlandı: {product.asin} (tüm metrikler NULL)")
                return False
                
            self.conn.commit()
            return True
        except Exception as e:
            print(f"      ❌ save_product_data HATASI: {e}")
            self.conn.rollback()
            return False

    def setup_zip_and_cookies(self, zip_code: str = "10001") -> bool:
        """Browser başlat ve Türkiye cookie'lerini Playwright'a aktar"""
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                loop.close()
        except Exception:
            pass
        asyncio.set_event_loop(asyncio.new_event_loop())
        page = None
        try:
            print(f"🌐 Browser başlatılıyor...")
            self.playwright = sync_playwright().start()
            
            selected_ua = random.choice(USER_AGENTS)
            print(f"   🎭 User-Agent: {selected_ua}")
            
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--window-size=1920,1080",
                ],
                proxy={"server": self.proxy} if self.proxy else None
            )
            
            self.context = self.browser.new_context(
                viewport={"width": 800, "height": 600},
                user_agent=selected_ua,
                locale="tr-TR",
                timezone_id="Europe/Istanbul",
                extra_http_headers={
                    "Accept-Language": "tr-TR,tr;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                }
            )
            
            def block_unnecessary(route):
                if route.request.resource_type in ["image", "stylesheet", "font", "media"]:
                    route.abort()
                else:
                    route.continue_()
            self.context.route("**/*", block_unnecessary)
            
            self.context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                window.chrome = { runtime: {} };
            """)
            
            for name, value in TR_COOKIES.items():
                if value:
                    self.context.add_cookies([{
                        'name': name,
                        'value': value,
                        'domain': '.amazon.com.tr',
                        'path': '/'
                    }])
            
            print(f"⏳ Amazon.com.tr'ye bağlanılıyor...")
            page = self.context.new_page()
            
            try:
                page.goto("https://www.amazon.com.tr", wait_until="networkidle", timeout=90000)
                print("   ✅ Ana sayfa yüklendi")
            except PlaywrightTimeout:
                print("   ⚠️ Timeout ama devam ediliyor...")
            
            time.sleep(3)
            
            # Cookie'leri al
            cookies = self.context.cookies()
            self.session.cookies.clear()
            for c in cookies:
                self.session.cookies.set(c['name'], c['value'], domain=c.get('domain', '.amazon.com.tr'))
            
            print(f"   🍪 {len(cookies)} cookie aktarıldı")
            
            if page:
                page.close()
            return True
            
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            return False

    def _close_playwright(self):
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except:
            pass

    def try_submit_checkpoint_form(self, asin: str, zip_code: str, html_content: str) -> tuple:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            target_form = None
            form_type = "unknown"
            
            for form in soup.find_all("form"):
                action = form.get("action", "")
                
                if "/errors/validateCaptcha" in action:
                    target_form = form
                    form_type = "validateCaptcha"
                    break
                
                if "nb_sb" in action or "/s/ref=" in action or "ref=nb" in action:
                    target_form = form
                    form_type = "nb_sb_redirect"
                    break
                
            if not target_form:
                return False, html_content, "Checkpoint formu bulunamadı"
            
            if form_type == "nb_sb_redirect":
                print(f"   📝 nb_sb_redirect tespit edildi, direkt ürüne gidiliyor...")
                
                headers = {
                    "Referer": "https://www.amazon.com.tr/",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "tr-TR,tr;q=0.9",
                    "Cache-Control": "max-age=0",
                }
                
                resp = self.session.get(
                    f"https://www.amazon.com.tr/dp/{asin}", 
                    headers=headers, 
                    timeout=10, 
                    allow_redirects=True
                )
                
                if resp.status_code == 200:
                    check_soup = BeautifulSoup(resp.content, 'html.parser')
                    title = check_soup.find("span", id="productTitle")
                    
                    if title and title.text.strip():
                        print(f"   ✅ Direkt erişim BAŞARILI!")
                        return True, resp.content, None
                    else:
                        content_lower = resp.text.lower()
                        if any(x in content_lower for x in ["validatecaptcha", "nb_sb", "continue shopping", "type the characters you see"]):
                            return False, resp.text, "Hala checkpoint'te"
                        return False, resp.text, "Başlık yok"
                else:
                    return False, html_content, f"HTTP {resp.status_code}"
            
            params = {}
            for inp in target_form.find_all("input"):
                name = inp.get("name")
                value = inp.get("value", "")
                if name:
                    params[name] = value
            
            action_url = target_form.get("action", "")
            submit_url = f"https://www.amazon.com.tr{action_url}" if action_url.startswith("/") else action_url
            
            print(f"   📝 validateCaptcha formu submit ediliyor...")
            if 'amzn' in params:
                print(f"      Token: {params['amzn'][:20]}...")
            
            headers = {
                "Referer": f"https://www.amazon.com.tr/dp/{asin}",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            
            resp = self.session.get(
                submit_url, 
                params=params, 
                headers=headers, 
                timeout=15, 
                allow_redirects=True
            )
            
            if resp.status_code == 200:
                check_soup = BeautifulSoup(resp.content, 'html.parser')
                title = check_soup.find("span", id="productTitle")
                
                if title and title.text.strip():
                    print(f"   ✅ Captcha bypass BAŞARILI!")
                    return True, resp.content, None
                else:
                    return False, resp.text, "Hala captcha sayfasında"
            else:
                return False, html_content, f"HTTP {resp.status_code}"
                
        except Exception as e:
            return False, html_content, str(e)

    def _parse_price_text(self, text: str) -> tuple:
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
                
        except Exception as e:
            print(f"      Fiyat parse hatası: {e}")
            return text, None
        
    def _clean_brand(self, text: str) -> Optional[str]:
        if not text:
            return None
        
        # Baştaki kalıpları temizle: "Marka:", "Visit the", "the"
        text = re.sub(r"^(Marka:|Visit the|the)\s*", "", text, flags=re.I)
        
        # Sondaki "Store'u ziyaret edin", "Store u ziyaret edin", "Store" gibi kalıpları temizle
        text = re.sub(r"\s+Store['’]?u?\s+ziyaret\s+edin\s*$", "", text, flags=re.I)
        text = re.sub(r"\s+Store['’]?u?\s*$", "", text, flags=re.I)
        
        return text.strip() if text.strip() else None

    def _parse_bsr_text(self, text: str) -> Optional[int]:
        if not text:
            return None
        
        try:
            match = re.search(r'#\s*([\d\.,]+)', text)
            if match:
                num_str = match.group(1)
                clean = num_str.replace('.', '').replace(',', '').replace(' ', '')
                if clean.isdigit():
                    return int(clean)
            
            # Türkiye formatı: "1.168. sırada" veya "70. sırada"
            match2 = re.search(r'([\d\.,]+)\.\s*s[ıi]rada', text, re.IGNORECASE)
            if match2:
                num_str = match2.group(1)
                clean = num_str.replace('.', '').replace(',', '').replace(' ', '')
                if clean.isdigit():
                    return int(clean)
                
            return None
        except:
            return None

    def scrape_fallback_playwright(self, asin: str, zip_code: str = "10001", bs4=None) -> AmazonProduct:
        url = f"https://www.amazon.com.tr/dp/{asin}?psc=1"
        
        if not self.context:
            return AmazonProduct(asin=asin, error="Playwright yok", timestamp=datetime.now().isoformat(), zip_code=zip_code)
        
        page = None
        try:
            print(f"   🔄 Playwright: {asin}")
            page = self.context.new_page()
            
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(random.uniform(3, 5))
            
            data = {}
            
            if bs4 is not None:
                data['title'], data['brand'], data['category'], data['price_raw'], data['price_num'], data['rating'], data['review_count'], data['image_url'] = bs4
            else:
                try:
                    title_el = page.locator("#productTitle").first
                    if title_el.count() > 0:
                        data['title'] = title_el.inner_text(timeout=5000).strip()
                    else:
                        data['title'] = None
                except:
                    data['title'] = None

                try:
                    brand_el = page.locator("#bylineInfo").first
                    if brand_el.count() > 0:
                        brand_text = brand_el.inner_text(timeout=3000)
                        data['brand'] = self._clean_brand(brand_text)
                    else:
                        data['brand'] = None
                except:
                    data['brand'] = None

                try:
                    cat_el = page.locator("#wayfinding-breadcrumbs_feature_div li").first
                    if cat_el.count() > 0:
                        data['category'] = cat_el.inner_text(timeout=3000).strip()
                    else:
                        data['category'] = None
                except:
                    data['category'] = None

                try:
                    price_el = page.locator("span.a-price .a-offscreen").first
                    if price_el.count() > 0:
                        price_text = price_el.inner_text(timeout=3000)
                        data['price_raw'], data['price_num'] = self._parse_price_text(price_text)
                    else:
                        data['price_raw'] = None
                        data['price_num'] = None
                except:
                    data['price_raw'] = None
                    data['price_num'] = None

                data['rating'] = None
                try:
                    # Öncelik 1: Saf rating değeri (örn: "4,6/5")
                    rating_el = page.locator("span[data-hook='rating-out-of-text']").first
                    if rating_el.count() > 0:
                        rating_text = rating_el.inner_text(timeout=3000).strip()
                        match = re.search(r"(\d+[,.]\d+)", rating_text)
                        if match:
                            data['rating'] = float(match.group(1).replace(',', '.'))
                    
                    # Öncelik 2: a-icon-alt içindeki metin (örn: "4,6/5 yıldız")
                    if data['rating'] is None:
                        rating_el = page.locator("span.a-icon-alt").first
                        if rating_el.count() > 0:
                            rating_text = rating_el.inner_text(timeout=3000).strip()
                            match = re.search(r"(\d+[,.]\d+)", rating_text)
                            if match:
                                data['rating'] = float(match.group(1).replace(',', '.'))
                    
                    # Fallback: acrPopover title'ından son sayıyı al
                    if data['rating'] is None:
                        rating_el = page.locator("#acrPopover").first
                        if rating_el.count() > 0:
                            rating_text = rating_el.get_attribute("title")
                            if rating_text:
                                # "5 yıldız üzerinden en 4,6" -> son sayıyı yakala
                                match = re.search(r"(\d+[,.]\d+)(?=\D*$)", rating_text)
                                if match:
                                    data['rating'] = float(match.group(1).replace(',', '.'))
                except Exception as e:
                    print(f"      Rating hatası: {e}")
                    data['rating'] = None

                try:
                    review_el = page.locator("#acrCustomerReviewText").first
                    if review_el.count() > 0:
                        review_text = review_el.inner_text(timeout=3000)
                        nums = re.findall(r"[\d\.]+", review_text)
                        data['review_count'] = int(nums[0].replace('.', '')) if nums else None
                    else:
                        data['review_count'] = None
                except:
                    data['review_count'] = None

            data['bsr'] = None
            try:
                scroll_positions = [0.4]
                
                for pos in scroll_positions:
                    page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pos})")
                    time.sleep(2)
                    
                    for sel in ["#SalesRank", "[data-hook='best-sellers-rank']", 
                               "th:has-text('En Çok Satanlar') + td",
                               "#productDetails_detailBullets_sections1 tr:has(td:has-text('#'))",
                               "li:has-text('Best Sellers Rank:')", "li:has-text('En Çok Satanlar Sıralaması:')"]:
                        try:
                            el = page.locator(sel).first
                            if el.count() > 0:
                                text = el.inner_text(timeout=3000)
                                bsr = self._parse_bsr_text(text)
                                if bsr:
                                    data['bsr'] = bsr
                                    print(f"      🏆 BSR: #{data['bsr']} (%{int(pos*100)} scroll)")
                                    break
                        except:
                            continue
                    
                    if data['bsr']:
                        break
                        
                if not data['bsr']:
                    print(f"      ⚠️ BSR bulunamadı")
                    
            except Exception as e:
                print(f"      BSR hatası: {e}")
                data['bsr'] = None
            
            try:
                avail_el = page.locator("#availability span").first
                if avail_el.count() > 0:
                    data['availability'] = avail_el.inner_text(timeout=5000).strip()
            except:
                data['availability'] = None
            
            try:
                img_el = page.locator("#landingImage").first
                if img_el.count() > 0:
                    data['image_url'] = img_el.get_attribute("data-old-hires") or img_el.get_attribute("src")
            except:
                data['image_url'] = None
            
            page.close()
            
            return AmazonProduct(
                asin=asin,
                title=data.get('title'),
                brand=data.get('brand'),
                category=data.get('category'),
                price_raw=data.get('price_raw'),
                price_numeric=data.get('price_num'),
                currency='TRY' if data.get('price_raw') else None,
                rating=data.get('rating'),
                review_count=data.get('review_count'),
                bestseller_rank=data.get('bsr'),
                image_url=data.get('image_url'),
                availability=data.get('availability'),
                timestamp=datetime.now().isoformat(),
                zip_code=zip_code
            )
            
        except Exception as e:
            if page:
                page.close()
            return AmazonProduct(asin=asin, error=f"Playwright: {str(e)[:50]}", timestamp=datetime.now().isoformat(), zip_code=zip_code)

    def scrape_with_bs4(self, asin: str, zip_code: str = "10001", request_count: int = 0) -> AmazonProduct:
        if request_count > 0 and request_count % 64 == 0:
            self._refresh_identity()

        if request_count % 64 == 0:
            print("   🔄 Cookie temizleniyor...")
            self._clear_dynamic_cookies()
            self._apply_tr_cookies()
        
        url = f"https://www.amazon.com.tr/dp/{asin}?psc=1"
        
        try:
            start = time.time()
            
            headers = {
                "User-Agent": self.session.headers.get("User-Agent"),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document",
            }
            self.session.headers.update({"Referer": "https://www.amazon.com.tr/"})
            resp = self.session.get(url, timeout=8, stream=True)
            
            size_header = resp.headers.get("Content-Length")
            if size_header:
                print(f"   📦 Network size: {int(size_header)/1024:.1f} KB (compressed)")
            else:
                print(f"   📦 Response size: {len(resp.content)/1024:.1f} KB (decompressed)")

            
            if resp.status_code == 404:
                print(f"   ❌ 404 Not Found: {asin}, sonraki ASIN'e geçiliyor...")
                return AmazonProduct(asin=asin, error="404", timestamp=datetime.now().isoformat(), zip_code=zip_code)
            
            if resp.status_code == 500:
                print(f"   ❌ 500 Internal Server Error: {asin}, Playwright ile tekrar deneniyor...")
                return self.scrape_fallback_playwright(asin, zip_code)

            content_lower = resp.text.lower()
            
            checkpoint_keywords = ["continue shopping", "validateCaptcha", "to continue", "devam etmek"]
            is_checkpoint = any(k in content_lower for k in checkpoint_keywords)
            
            if is_checkpoint:
                print(f"\n   🛑 Checkpoint: {asin}")
                
                success, new_html, error_msg = self.try_submit_checkpoint_form(asin, zip_code, resp.text)
                
                if success:
                    print(f"   ✅ Form submit OK")
                    resp = type('Response', (), {
                        'content': new_html if isinstance(new_html, bytes) else new_html.encode('utf-8'), 
                        'text': new_html if isinstance(new_html, str) else new_html.decode('utf-8')
                    })()
                else:
                    print(f"   ⚠️ {error_msg}, Playwright...")
                    return self.scrape_fallback_playwright(asin, zip_code)
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            t = soup.find("span", id="productTitle")
            title = t.text.strip() if t else None
            if not title:
                return AmazonProduct(asin=asin, error="Başlık yok", timestamp=datetime.now().isoformat(), zip_code=zip_code)
            
            b = soup.find("a", id="bylineInfo")
            brand = self._clean_brand(b.text) if b else None
            
            cat = None
            bc = soup.find("div", id="wayfinding-breadcrumbs_feature_div")
            if bc:
                items = bc.find_all("li")
                cat = items[0].text.strip() if items else None
            
            price_raw, price_num = self._extract_price_robust(soup)
            rating, review_count = self._extract_rating(soup)
            bsr = self._extract_bsr_robust(soup)

            image_url = None
            try:
                img = soup.select_one("#landingImage")
                if img:
                    image_url = img.get("data-old-hires") or img.get("src")
            except:
                pass

            if price_num:
                print(f"   💰 Price: {price_raw} | Rating: {rating} | Reviews: {review_count} | BSR: {bsr}")
            
            if bsr is None:
                print(f"   ⚠️ Captcha bypass sonrası BSR eksik, Playwright ile tekrar çekiliyor...")
                return self.scrape_fallback_playwright(asin, zip_code, bs4=(title, brand, cat, price_raw, price_num, rating, review_count, image_url))
            
            avail = None
            av = soup.find("div", id="availability")
            if av:
                sp = av.find("span")
                avail = sp.text.strip() if sp else None
            
            elapsed = time.time() - start
            tag = "[Form+BS4]" if is_checkpoint else "[BS4]"
            print(f"   ✅ {title[:45]} ({elapsed:.1f}s) {tag}")
            
            return AmazonProduct(
                asin=asin, title=title, brand=brand, category=cat,
                price_raw=price_raw, price_numeric=price_num, currency='TRY' if price_raw else None,
                rating=rating, review_count=review_count, bestseller_rank=bsr, image_url=image_url, availability=avail,
                timestamp=datetime.now().isoformat(), zip_code=zip_code
            )
            
        except Exception as e:
            return AmazonProduct(asin=asin, error=str(e)[:50], timestamp=datetime.now().isoformat(), zip_code=zip_code)

    def _extract_price_robust(self, soup):
        selectors = [
            "span.a-offscreen", "span.a-price .a-offscreen",
            "#priceblock_dealprice", "#priceblock_ourprice",
            ".a-price-range .a-offscreen", ".a-price.a-text-price .a-offscreen"
        ]
        
        for sel in selectors:
            try:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text().strip()
                    price_raw, price_num = self._parse_price_text(text)
                    if price_num is not None:
                        return price_raw, price_num
            except Exception as e:
                print(f"      Fiyat selector hatası: {e}")
                continue
        
        return None, None

    def _extract_rating(self, soup):
        rating = None
        
        # 1. Öncelik: data-hook="rating-out-of-text" (saf değer, örn: "4,6/5")
        r = soup.find("span", {"data-hook": "rating-out-of-text"})
        if r:
            match = re.search(r"(\d+[,.]\d+)", r.text)
            if match:
                rating = float(match.group(1).replace(',', '.'))
        
        # 2. Alternatif: a-icon-alt (örn: "4,6/5 yıldız")
        if rating is None:
            r = soup.find("span", class_="a-icon-alt")
            if r:
                match = re.search(r"(\d+[,.]\d+)", r.text)
                if match:
                    rating = float(match.group(1).replace(',', '.'))
        
        # Review count
        rc = soup.find("span", id="acrCustomerReviewText")
        reviews = None
        if rc:
            nums = re.findall(r"[\d\.]+", rc.text)
            reviews = int(nums[0].replace('.', '')) if nums else None
        
        return rating, reviews

    def _extract_bsr_robust(self, soup):
        # 1. En Çok Satanlar Sıralaması TH'sini bul
        try:
            for th in soup.find_all("th"):
                th_text = th.get_text(strip=True)
                if "en çok satanlar" in th_text.lower() or "siralamasi" in th_text.lower():
                    td = th.find_next_sibling("td")
                    if td:
                        td_text = td.get_text()
                        match = re.search(r'([\d\.,]+)\.\s*s[ıi]rada', td_text, re.IGNORECASE)
                        if match:
                            clean = match.group(1).replace('.', '').replace(',', '').replace(' ', '')
                            if clean.isdigit():
                                return int(clean)
        except:
            pass
        
        # 2. prodDetails tablosu
        try:
            table = soup.find("table", {"class": lambda x: x and "prodDetTable" in x})
            if table:
                for tr in table.find_all("tr"):
                    tr_text = tr.get_text()
                    if "en çok satan" in tr_text.lower() or "siralamasi" in tr_text.lower():
                        match = re.search(r'([\d\.,]+)\.\s*s[ıi]rada', tr_text, re.IGNORECASE)
                        if match:
                            clean = match.group(1).replace('.', '').replace(',', '').replace(' ', '')
                            if clean.isdigit():
                                return int(clean)
        except:
            pass
        
        # 3. Sayfa genelinde "X. sırada" ara
        try:
            text = soup.get_text()
            idx = text.lower().find("en çok satanlar")
            if idx != -1:
                snippet = text[idx:idx+500]
                match = re.search(r'([\d\.,]+)\.\s*s[ıi]rada', snippet, re.IGNORECASE)
                if match:
                    clean = match.group(1).replace('.', '').replace(',', '').replace(' ', '')
                    if clean.isdigit():
                        return int(clean)
        except:
            pass
        
        # 4. Genel fallback - # işaretli format
        try:
            text = soup.get_text()
            match = re.search(r'#\s*([\d\.,]+)', text)
            if match:
                clean = match.group(1).replace('.', '').replace(',', '').replace(' ', '')
                if clean.isdigit():
                    return int(clean)
        except:
            pass
        
        # 5. SalesRank div
        try:
            sr = soup.find("div", id="SalesRank")
            if sr:
                bsr = self._parse_bsr_text(sr.get_text())
                if bsr:
                    return bsr
        except:
            pass
        
        # 6. Detail bullets
        try:
            db = soup.find("div", id="detailBullets_feature_div")
            if db:
                for li in db.find_all("li"):
                    txt = li.get_text()
                    if "best seller" in txt.lower() or "en çok satan" in txt.lower():
                        bsr = self._parse_bsr_text(txt)
                        if bsr:
                            return bsr
        except:
            pass
        
        # 7. Table yapısı
        try:
            for th in soup.find_all("th"):
                th_txt = th.get_text()
                if "best sellers rank" in th_txt.lower() or "en çok satanlar" in th_txt.lower():
                    td = th.find_next_sibling("td")
                    if td:
                        bsr = self._parse_bsr_text(td.get_text())
                        if bsr:
                            return bsr
        except:
            pass
        
        # 8. DetailBulletsWrapper
        try:
            wrapper = soup.find("div", id="detailBulletsWrapper_feature_div")
            if wrapper:
                for span in wrapper.find_all("span"):
                    txt = span.get_text()
                    if "best sellers rank" in txt.lower() or "en çok satan" in txt.lower():
                        bsr = self._parse_bsr_text(txt)
                        if bsr:
                            return bsr
        except:
            pass

        # 9. ProductDetails tablosu
        try:
            prod_details = soup.find("div", id="productDetails_detailBullets_sections1")
            if prod_details:
                for tr in prod_details.find_all("tr"):
                    txt = tr.get_text()
                    if "best sellers rank" in txt.lower() or "en çok satan" in txt.lower():
                        bsr = self._parse_bsr_text(txt)
                        if bsr:
                            return bsr
        except:
            pass

        # 10. Sayfa metni geniş ara
        try:
            text = soup.get_text()
            patterns = [
                r"Best Sellers? Rank[:\s]+#?([\d\.,]+)",
                r"En Çok Satanlar? Sıralaması[:\s]+#?([\d\.,]+)",
                r"Amazon Best Sellers Rank[:\s]+#?([\d\.,]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    bsr = self._parse_bsr_text(match.group(0))
                    if bsr:
                        return bsr
        except:
            pass

        # 11. Regex fallback
        try:
            text = soup.get_text()
            match = re.search(r'#\s*([\d\.,]+)\s+in\s+[A-Za-z0-9&\-\s\(\)&çğıöşüÇĞİÖŞÜ]+', text)
            if match:
                bsr = self._parse_bsr_text(match.group(0))
                if bsr:
                    return bsr
        except:
            pass
        
        # 12. data-hook attribute
        try:
            rank_span = soup.find("span", {"data-hook": "best-sellers-rank"})
            if rank_span:
                bsr = self._parse_bsr_text(rank_span.get_text())
                if bsr:
                    return bsr
        except:
            pass

        try:
            li = soup.find("li", string=re.compile("Best Sellers Rank|En Çok Satan", re.I))
            if li:
                bsr = self._parse_bsr_text(li.get_text())
                if bsr:
                    return bsr
        except:
            pass
        
        try:
            bold_spans = soup.find_all("span", class_="a-text-bold")
            for sp in bold_spans:
                txt = sp.get_text()
                if "best sellers rank" in txt.lower() or "en çok satan" in txt.lower():
                    bsr = self._parse_bsr_text(sp.find_parent().get_text())
                    if bsr:
                        return bsr
        except:
            pass

        try:
            mobile_rank = soup.select_one("div[data-asin] span:-soup-contains('Best Sellers Rank')")
            if mobile_rank:
                bsr = self._parse_bsr_text(mobile_rank.get_text())
                if bsr:
                    return bsr
        except:
            pass
        
        try:
            prod_details = soup.find("table", id="prodDetails")
            if prod_details:
                for tr in prod_details.find_all("tr"):
                    txt = tr.get_text()
                    if "best sellers rank" in txt.lower() or "en çok satan" in txt.lower():
                        bsr = self._parse_bsr_text(txt)
                        if bsr:
                            return bsr
        except:
            pass
        
        return None

    def run_batch(self, batch_size: int = 5, zip_code: str = "10001"):
        print(f"\n🚀 HYBRID SCRAPER (TR Cookies + UA Rotation + Form Submit + Playwright)")
        print("="*70)
        
        if not self.setup_zip_and_cookies(zip_code):
            print("❌ Browser başlatılamadı")
            return
        
        asins = self.get_unprocessed_asins(limit=batch_size)
        if not asins:
            print("ℹ️ İşlenecek ASIN yok")
            return
        
        print(f"⚡ {len(asins)} ürün (Her 10 üründe kimlik yenilenir)\n")
        
        success_count = 0
        fallback_count = 0
        
        for i, asin in enumerate(asins, 1):
            product = self.scrape_with_bs4(asin, zip_code, request_count=i)
            
            if product.error and "Playwright" in product.error:
                fallback_count += 1
            
            if product.error:
                print(f"[{i}/{len(asins)}] {asin} ❌ {product.error}")
                self.mark_as_processed(asin, False, product.error)
            else:
                if self.save_product_data(product):
                    success_count += 1
                    self.mark_as_processed(asin, True)
                    price = f"₺{product.price_numeric}" if product.price_numeric else "N/A"
                    bsr = f"#{product.bestseller_rank}" if product.bestseller_rank else "N/A"
                    print(f"[{i}/{len(asins)}] {asin} 💾 {price} | BSR:{bsr}")
                else:
                    if product.error:
                        print(f"[{i}/{len(asins)}] {asin} ❌ DB kaydetme hatası")
                        self.mark_as_processed(asin, False, "DB kaydetme hatası")
                    else:
                        print(f"[{i}/{len(asins)}] {asin} ⚠️ Metrikler eksik, tekrar denenecek")
            
            if i < len(asins):
                time.sleep(random.uniform(3.0, 6.0))
        
        print(f"\n🏁 {success_count}/{len(asins)} başarılı")
        print(f"🔄 Playwright fallback: {fallback_count}")

    def close(self):
        self._close_playwright()
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("🔌 Kapatıldı")

def _scrape_worker(db_config: dict, asin: str, zip_code: str):
    """Tek ASIN'i işleyen worker. Her worker kendi DB + Browser'ını açar."""
    scraper = AmazonScraper(db_config, headless=True)
    try:
        product = scraper.scrape_with_bs4(asin, zip_code, request_count=0)
        
        if product.error and product.error not in ["404", "Başlık yok"]:
            scraper.mark_as_processed(asin, False, product.error)
            return asin, False, product.error
        elif not product.error:
            if scraper.save_product_data(product):
                scraper.mark_as_processed(asin, True)
                price = f"₺{product.price_numeric}" if product.price_numeric else "N/A"
                bsr = f"#{product.bestseller_rank}" if product.bestseller_rank else "N/A"
                return asin, True, f"{price} | BSR:{bsr}"
            else:
                return asin, False, "DB kaydetme hatası"
        else:
            scraper.mark_as_processed(asin, False, product.error)
            return asin, False, product.error
    except Exception as e:
        return asin, False, str(e)
    finally:
        scraper.close()


def get_next_asins(db_config: dict, limit: int = 5) -> list:
    """Ana process'den kilitli ASIN çek (FOR UPDATE SKIP LOCKED)"""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("""
        SELECT asin FROM asin_pool_tr 
        WHERE processed = FALSE 
        ORDER BY created_at ASC 
        LIMIT %s
        FOR UPDATE SKIP LOCKED
    """, (limit,))
    asins = [row[0] for row in cur.fetchall()]
    conn.commit()
    cur.close()
    conn.close()
    return asins


def run_parallel(db_config: dict, batch_size: int = 5, zip_code: str = "10001", max_workers: int = 5):
    """Paralel batch işleme. Her batch'te 5 ürün, 5 worker ile aynı anda."""
    print(f"\n🚀 PARALEL SCRAPER (TR) - {max_workers} worker")
    print("="*70)
    
    total_success = 0
    total_fail = 0
    batch_no = 1
    
    while True:
        asins = get_next_asins(db_config, batch_size)
        if not asins:
            print(f"\n🏁 Tüm ürünler işlendi! Toplam: {total_success} başarılı, {total_fail} hatalı")
            break
        
        print(f"\n📦 Batch #{batch_no}: {len(asins)} ürün alındı, {max_workers} worker ile işleniyor...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_scrape_worker, db_config, asin, zip_code): asin 
                for asin in asins
            }
            
            for future in as_completed(futures):
                asin, success, msg = future.result()
                if success:
                    total_success += 1
                    print(f"   ✅ {asin}: {msg}")
                else:
                    total_fail += 1
                    print(f"   ❌ {asin}: {msg}")
        
        batch_no += 1
        time.sleep(random.uniform(2, 4))

if __name__ == "__main__":
    DB_CONFIG = {
        "host": "localhost",
        "dbname": "dropshipping",
        "user": "scraper",
        "password": "guclu_sifre_123"
    }

    # PARALEL MOD (5 worker, her seferinde 5 ürün)
    #run_parallel(DB_CONFIG, batch_size=5, zip_code="10001", max_workers=5)
    
    import psycopg2
    
    while True:
        scraper = AmazonScraper(DB_CONFIG, headless=True)
        try:
            scraper.run_batch(batch_size=65, zip_code="10001")
        except KeyboardInterrupt:
            print("\n🛑 Durduruldu")
            break
        except Exception as e:
            print(f"\n💥 Hata: {e}")
        finally:
            scraper.close()
        
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM asin_pool_tr WHERE processed = FALSE")
        remaining = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        if remaining == 0:
            print("🏁 Tüm ürünler işlendi!")
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("UPDATE asin_pool_tr SET processed = FALSE")
            conn.commit()
            updated = cur.rowcount
            cur.close()
            conn.close()
            print(f"✅ {updated} ASIN processed = FALSE yapıldı. Program sonlanıyor.")
            break
        
        print(f"⏳ Kalan {remaining} ürün, 5 saniye sonra devam ediliyor...")
        time.sleep(5)