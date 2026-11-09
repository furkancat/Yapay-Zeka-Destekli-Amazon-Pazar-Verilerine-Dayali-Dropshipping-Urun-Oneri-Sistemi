from fastapi import FastAPI, Depends, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, text
from datetime import datetime, timedelta
from typing import Optional
import json

import os
import pandas as pd

from database import veritabani_al, motor, Taban
from models import (
    Kullanici, KrediHareketi, UrunErisimLogu,
    Product, ProductMetricsTR, LatestProductMetricsTR
)
from auth import (
    sifre_hashle, sifre_dogrula, erisim_tokeni_olustur,
    mevcut_kullanici, mevcut_kullanici_opsiyonel, admin_mi
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
Taban.metadata.create_all(bind=motor)

# ------------------ AUTH ------------------

@app.get("/", response_class=HTMLResponse)
def ana_sayfa(request: Request):
    return RedirectResponse(url="/dashboard")

@app.get("/login", response_class=HTMLResponse)
def login_sayfasi(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/token")
def token_al(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(veritabani_al)
):
    kullanici = db.query(Kullanici).filter(Kullanici.kullanici_adi == form_data.username).first()
    if not kullanici or not sifre_dogrula(form_data.password, kullanici.sifre_hash):
        raise HTTPException(status_code=401, detail="Kullanici adi veya sifre hatali")
    token = erisim_tokeni_olustur(kullanici.kullanici_adi, kullanici.id, kullanici.rol)
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True, max_age=86400, path="/", samesite="lax")
    return response

@app.get("/kayit", response_class=HTMLResponse)
def kayit_sayfasi(request: Request):
    return templates.TemplateResponse("kayit.html", {"request": request})

@app.post("/kayit")
def kayit_ol(
    kullanici_adi: str = Form(...),
    email: str = Form(...),
    sifre: str = Form(...),
    db: Session = Depends(veritabani_al)
):
    mevcut = db.query(Kullanici).filter(
        (Kullanici.kullanici_adi == kullanici_adi) | (Kullanici.email == email)
    ).first()
    if mevcut:
        raise HTTPException(status_code=400, detail="Kullanici zaten var")
    yeni = Kullanici(
        kullanici_adi=kullanici_adi, email=email,
        sifre_hash=sifre_hashle(sifre), kredi=50
    )
    db.add(yeni); db.commit(); db.refresh(yeni)
    db.add(KrediHareketi(kullanici_id=yeni.id, miktar=50, sebep="kayit_bonusu"))
    db.commit()
    token = erisim_tokeni_olustur(yeni.kullanici_adi, yeni.id, yeni.rol)
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True, max_age=86400, path="/", samesite="lax")
    return response

@app.get("/cikis")
def cikis_yap():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# ------------------ KATEGORI SAYFALARI ------------------

@app.get("/kategoriler", response_class=HTMLResponse)
def kategoriler_listesi(
    request: Request,
    db: Session = Depends(veritabani_al),
    kullanici: Optional[Kullanici] = Depends(mevcut_kullanici_opsiyonel)
):
    if not kullanici:
        return RedirectResponse(url="/login")
    kategori_sorgu = db.query(
        Product.category,
        func.count(distinct(Product.asin)).label("urun_sayisi")
    ).filter(Product.category != None).group_by(Product.category).all()
    kategoriler = []
    for kat, sayi in kategori_sorgu:
        ornek = db.query(LatestProductMetricsTR, Product).join(
            Product, LatestProductMetricsTR.asin == Product.asin
        ).filter(Product.category == kat).order_by(
            LatestProductMetricsTR.bestseller_rank.asc()
        ).first()
        kategoriler.append({
            "adi": kat,
            "urun_sayisi": sayi,
            "top_urun": ornek[1].title[:40] + "..." if ornek and ornek[1].title else ""
        })
    return templates.TemplateResponse("kategoriler.html", {
        "request": request, "kullanici": kullanici, "kategoriler": kategoriler
    })

@app.get("/kategori/{kategori_adi}", response_class=HTMLResponse)
def kategori_detay(
    request: Request,
    kategori_adi: str,
    sayfa: int = Query(1, ge=1),
    siralama: str = Query("firsat_desc"), # VARSAYILAN SIRALAMA ARTIK FIRSAT SKORU
    db: Session = Depends(veritabani_al),
    kullanici: Optional[Kullanici] = Depends(mevcut_kullanici_opsiyonel)
):
    if not kullanici:
        return RedirectResponse(url="/login")
    
    # 1. AI Tahminleri ve Fırsat Skorlarını CSV'den Oku
    csv_yolu = r"C:\Users\user\Desktop\deneme\dropshipping_model\output\oneri_ensemble.csv"
    ai_dict = {}
    if os.path.exists(csv_yolu):
        df_ai = pd.read_csv(csv_yolu)
        for _, row in df_ai.iterrows():
            asin_temiz = str(row['asin']).strip() # BOŞLUKLAR TEMİZLENDİ
            ai_dict[asin_temiz] = {
                "tahmini_bsr": int(row.get('bsr_tahmin_ortalama', 0)) if pd.notna(row.get('bsr_tahmin_ortalama')) else None,
                "firsat_skoru": float(row.get('firsat_skoru_ortalama', 0)) if pd.notna(row.get('firsat_skoru_ortalama')) else -999
            }

    sayfa_basina = 20
    
    # 2. Veritabanından Tüm Ürünleri Çek
    sorgu = db.query(LatestProductMetricsTR, Product).join(
        Product, LatestProductMetricsTR.asin == Product.asin
    ).filter(
        Product.category == kategori_adi,
        LatestProductMetricsTR.bestseller_rank != None
    ).all()

    # 3. Verileri Birleştir
    urunler_liste = []
    for metrik, urun in sorgu:
        db_asin_temiz = metrik.asin.strip() # VERİTABANINDAKİ BOŞLUKLAR DA TEMİZLENDİ
        ai_veri = ai_dict.get(db_asin_temiz, {"tahmini_bsr": None, "firsat_skoru": -999})
        urunler_liste.append({
            "asin": db_asin_temiz,
            "title": urun.title or "Basliksiz",
            "category": urun.category or "Kategorisiz",
            "price": float(metrik.price) if metrik.price else None,
            "bsr": metrik.bestseller_rank,
            "review_count": metrik.review_count,
            "rating": float(metrik.rating) if metrik.rating else None,
            "tahmini_bsr": ai_veri["tahmini_bsr"],
            "firsat_skoru": ai_veri["firsat_skoru"]
        })

    # 4. Seçilen Kriterine Göre Sıralama (Varsayılan: Fırsat Skoru)
    if siralama == "firsat_desc":
        urunler_liste.sort(key=lambda x: x["firsat_skoru"], reverse=True)
    elif siralama == "bsr_asc":
        urunler_liste.sort(key=lambda x: (x["bsr"] is None, x["bsr"]))
    elif siralama == "bsr_desc":
        urunler_liste.sort(key=lambda x: (x["bsr"] is None, x["bsr"]), reverse=True)
    elif siralama == "fiyat_asc":
        urunler_liste.sort(key=lambda x: (x["price"] is None, x["price"]))
    elif siralama == "fiyat_desc":
        urunler_liste.sort(key=lambda x: (x["price"] is None, x["price"]), reverse=True)

    # 5. Sayfalama (Pagination)
    toplam = len(urunler_liste)
    toplam_sayfa = (toplam + sayfa_basina - 1) // sayfa_basina
    offset = (sayfa - 1) * sayfa_basina
    urunler_sayfa = urunler_liste[offset:offset + sayfa_basina]

    return templates.TemplateResponse("kategori_detay.html", {
        "request": request, "kullanici": kullanici, "kategori_adi": kategori_adi,
        "urunler": urunler_sayfa, "sayfa": sayfa, "toplam_sayfa": toplam_sayfa,
        "toplam": toplam, "siralama": siralama
    })

# ------------------ DASHBOARD (SADECE CAROUSEL) ------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(veritabani_al),
    kullanici: Optional[Kullanici] = Depends(mevcut_kullanici_opsiyonel)
):
    if not kullanici:
        return RedirectResponse(url="/login")
        
    kategoriler = db.query(distinct(Product.category)).filter(Product.category != None).all()
    kategoriler = [k[0] for k in kategoriler if k[0]]
    kategori_carouseller = []
    
    # 1. AI Tahminleri ve Fırsat Skorlarını CSV'den Oku
    csv_yolu = r"C:\Users\user\Desktop\deneme\dropshipping_model\output\oneri_ensemble.csv"
    ai_dict = {}
    if os.path.exists(csv_yolu):
        df_ai = pd.read_csv(csv_yolu)
        for _, row in df_ai.iterrows():
            asin_temiz = str(row['asin']).strip() # BOŞLUKLAR TEMİZLENDİ
            ai_dict[asin_temiz] = {
                "tahmini_bsr": int(row.get('bsr_tahmin_ortalama', 0)) if pd.notna(row.get('bsr_tahmin_ortalama')) else None,
                "firsat_skoru": float(row.get('firsat_skoru_ortalama', 0)) if pd.notna(row.get('firsat_skoru_ortalama')) else -999
            }
            
    for kat in kategoriler[:6]:
        sonuc = db.query(LatestProductMetricsTR, Product).join(
            Product, LatestProductMetricsTR.asin == Product.asin
        ).filter(
            Product.category == kat,
            LatestProductMetricsTR.bestseller_rank != None
        ).all()
        
        urunler_liste = []
        for metrik, urun in sonuc:
            db_asin_temiz = metrik.asin.strip() # VERİTABANINDAKİ BOŞLUKLAR DA TEMİZLENDİ
            ai_veri = ai_dict.get(db_asin_temiz, {"tahmini_bsr": None, "firsat_skoru": -999})
            urunler_liste.append({
                "asin": db_asin_temiz,
                "title": urun.title or "Basliksiz",
                "price": float(metrik.price) if metrik.price else None,
                "bsr": metrik.bestseller_rank,
                "review_count": metrik.review_count,
                "rating": float(metrik.rating) if metrik.rating else None,
                "tahmini_bsr": ai_veri["tahmini_bsr"],
                "firsat_skoru": ai_veri["firsat_skoru"]
            })
            
        # 2. Sadece Sıralama Yap (Fırsat Skoruna Göre Diz) ve İlk 10'u Al
        urunler_liste.sort(key=lambda x: x["firsat_skoru"], reverse=True)
        top_10_urunler = urunler_liste[:10]
        
        kategori_carouseller.append({"kategori": kat, "urunler": top_10_urunler})
        
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "kullanici": kullanici,
        "kategori_carouseller": kategori_carouseller
    })

# ------------------ SEARCH (BEDAVA) ------------------

@app.get("/search", response_class=HTMLResponse)
def search_sayfasi(
    request: Request,
    db: Session = Depends(veritabani_al),
    kullanici: Optional[Kullanici] = Depends(mevcut_kullanici_opsiyonel)
):
    if not kullanici:
        return RedirectResponse(url="/login")
    kategoriler = db.query(distinct(Product.category)).filter(Product.category != None).all()
    kategoriler = [k[0] for k in kategoriler if k[0]]
    return templates.TemplateResponse("search.html", {
        "request": request, "kullanici": kullanici,
        "kategoriler": kategoriler, "urunler": []
    })

@app.post("/search")
def search_arama(
    request: Request,
    kategori: Optional[str] = Form(None),
    q: Optional[str] = Form(None),
    max_bsr: Optional[int] = Form(None),
    min_fiyat: Optional[float] = Form(None),
    max_fiyat: Optional[float] = Form(None),
    siralama: str = Form("bsr_asc"),
    db: Session = Depends(veritabani_al),
    kullanici: Optional[Kullanici] = Depends(mevcut_kullanici_opsiyonel)
):
    if not kullanici:
        return RedirectResponse(url="/login")
    sorgu = db.query(LatestProductMetricsTR, Product).join(
        Product, LatestProductMetricsTR.asin == Product.asin
    ).filter(LatestProductMetricsTR.bestseller_rank != None)
    if kategori:
        sorgu = sorgu.filter(Product.category == kategori)
    if q:
        sorgu = sorgu.filter(Product.title.ilike(f"%{q}%"))
    if max_bsr:
        sorgu = sorgu.filter(LatestProductMetricsTR.bestseller_rank < max_bsr)
    if min_fiyat:
        sorgu = sorgu.filter(LatestProductMetricsTR.price >= min_fiyat)
    if max_fiyat:
        sorgu = sorgu.filter(LatestProductMetricsTR.price <= max_fiyat)
    if siralama == "bsr_desc":
        sorgu = sorgu.order_by(LatestProductMetricsTR.bestseller_rank.desc())
    elif siralama == "fiyat_asc":
        sorgu = sorgu.order_by(LatestProductMetricsTR.price.asc())
    elif siralama == "fiyat_desc":
        sorgu = sorgu.order_by(LatestProductMetricsTR.price.desc())
    else:
        sorgu = sorgu.order_by(LatestProductMetricsTR.bestseller_rank.asc())
    sonuc = sorgu.limit(20).all()
    urunler = []
    for metrik, urun in sonuc:
        urunler.append({
            "asin": metrik.asin,
            "title": urun.title or "Basliksiz",
            "category": urun.category or "Kategorisiz",
            "price": float(metrik.price) if metrik.price else None,
            "bsr": metrik.bestseller_rank,
            "review_count": metrik.review_count,
            "rating": float(metrik.rating) if metrik.rating else None
        })
    return templates.TemplateResponse("parca_urun_kartlari.html", {
        "request": request, "urunler": urunler, "kullanici": kullanici
    })

# ------------------ URUN DETAY (KREDILI) ------------------

@app.get("/urun/{asin}", response_class=HTMLResponse)
def urun_detay(
    request: Request,
    asin: str,
    db: Session = Depends(veritabani_al),
    kullanici: Optional[Kullanici] = Depends(mevcut_kullanici_opsiyonel)
):
    if not kullanici:
        return RedirectResponse(url="/login")
    
    # Admin kredisi azalmaz
    if kullanici.rol != "admin":
        if kullanici.kredi < 2:
            return templates.TemplateResponse("hata.html", {
                "request": request,
                "mesaj": "Yetersiz kredi. Detaylı analiz için 2 kredi gerekli.",
                "kullanici": kullanici
            })
        kullanici.kredi -= 2
        db.add(KrediHareketi(kullanici_id=kullanici.id, miktar=-2, sebep="urun_detay", referans_asin=asin))
        db.add(UrunErisimLogu(kullanici_id=kullanici.id, asin=asin, harcanan_kredi=2))
        db.commit()

    asin = asin.strip()
    urun = db.query(Product).filter(Product.asin == asin).first()
    if not urun:
        return templates.TemplateResponse("hata.html", {
            "request": request, "mesaj": f"ASIN '{asin}' bulunamadı.", "kullanici": kullanici
        })
        
    son_metrik = db.query(LatestProductMetricsTR).filter(LatestProductMetricsTR.asin == asin).first()
    gecmis = db.query(ProductMetricsTR).filter(
        ProductMetricsTR.asin == asin,
        ProductMetricsTR.scraped_at >= datetime.now() - timedelta(days=90)
    ).order_by(ProductMetricsTR.scraped_at.asc()).all()

    # --- YAPAY ZEKA (ENSEMBLE) VERİSİNİ FARKLI KLASÖRDEN OKUMA ---
    ai_verisi = None
    # Model klasörünün tam yolu (Absolute Path)
    csv_yolu = r"C:\Users\user\Desktop\deneme\dropshipping_model\output\oneri_ensemble.csv"
    
    if os.path.exists(csv_yolu):
        df_oneri = pd.read_csv(csv_yolu)
        urun_ai = df_oneri[df_oneri['asin'] == asin]
        if not urun_ai.empty:
            ai_row = urun_ai.iloc[0]
            ai_verisi = {
                "tahmini_bsr": int(ai_row['bsr_tahmin_ortalama'])
            }
    # -------------------------------------------------------------

    urun_dict = {
        "asin": asin,
        "title": urun.title or "Başlıksız",
        "category": urun.category or "Kategorisiz",
        "brand": urun.brand,
        "price": float(son_metrik.price) if son_metrik and son_metrik.price else None,
        "bsr": son_metrik.bestseller_rank if son_metrik else None,
        "review_count": son_metrik.review_count if son_metrik else 0,
        "rating": float(son_metrik.rating) if son_metrik and son_metrik.rating else None,
        "ai": ai_verisi  # AI verisini arayüze gönderiyoruz
    }

    grafik_tarihler = [m.scraped_at.strftime("%Y-%m-%d") for m in gecmis if m.scraped_at]
    grafik_fiyatlar = [float(m.price) if m.price else None for m in gecmis]
    grafik_bsr = [m.bestseller_rank for m in gecmis]
    grafik_yorum = [m.review_count for m in gecmis]
    grafik_rating = [float(m.rating) if m.rating else None for m in gecmis]

    return templates.TemplateResponse("urun_detay.html", {
        "request": request, "kullanici": kullanici, "urun": urun_dict,
        "grafik_tarihler": json.dumps(grafik_tarihler),
        "grafik_fiyatlar": json.dumps(grafik_fiyatlar),
        "grafik_bsr": json.dumps(grafik_bsr),
        "grafik_yorum": json.dumps(grafik_yorum),
        "grafik_rating": json.dumps(grafik_rating)
    })

# ------------------ ADMIN ------------------

@app.get("/yonetici", response_class=HTMLResponse)
def yonetici_paneli(
    request: Request,
    db: Session = Depends(veritabani_al),
    kullanici: Kullanici = Depends(admin_mi)
):
    toplam_kullanici = db.query(Kullanici).count()
    toplam_urun = db.query(Product).count()
    kullanicilar = db.query(Kullanici).order_by(Kullanici.id.desc()).all()

    return templates.TemplateResponse("yonetici.html", {
        "request": request, "kullanici": kullanici,
        "toplam_kullanici": toplam_kullanici,
        "toplam_urun": toplam_urun,
        "kullanicilar": kullanicilar
    })

@app.post("/admin/kullanici/{kullanici_id}/rol")
def kullanici_rol_degistir(
    kullanici_id: int,
    yeni_rol: str = Form(...),
    db: Session = Depends(veritabani_al),
    admin_kullanici: Kullanici = Depends(admin_mi)
):
    hedef = db.query(Kullanici).filter(Kullanici.id == kullanici_id).first()
    if not hedef:
        raise HTTPException(status_code=404, detail="Kullanici bulunamadi")
    if hedef.id == admin_kullanici.id:
        raise HTTPException(status_code=400, detail="Kendi rolunuzu degistiremezsiniz")
    if yeni_rol not in ("kullanici", "admin"):
        raise HTTPException(status_code=400, detail="Gecersiz rol")

    hedef.rol = yeni_rol
    db.commit()
    return RedirectResponse(url="/yonetici", status_code=302)

@app.post("/admin/kullanici/{kullanici_id}/kredi")
def kullanici_kredi_ekle(
    kullanici_id: int,
    miktar: int = Form(...),
    db: Session = Depends(veritabani_al),
    admin_kullanici: Kullanici = Depends(admin_mi)
):
    hedef = db.query(Kullanici).filter(Kullanici.id == kullanici_id).first()
    if not hedef:
        raise HTTPException(status_code=404, detail="Kullanici bulunamadi")

    hedef.kredi += miktar
    db.commit()

    db.add(KrediHareketi(
        kullanici_id=hedef.id,
        miktar=miktar,
        sebep="admin_ekleme",
        referans_asin=None
    ))
    db.commit()

    return RedirectResponse(url="/yonetici", status_code=302)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)