from sqlalchemy import Column, Integer, String, Float, DateTime, Numeric, Boolean, Text
from sqlalchemy.orm import relationship
from database import Taban
from datetime import datetime

# ------------------ AUTH TABLOLARI (Sen ekledin) ------------------

class Kullanici(Taban):
    __tablename__ = "kullanicilar"

    id = Column(Integer, primary_key=True, index=True)
    kullanici_adi = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    sifre_hash = Column(String(255))
    rol = Column(String(20), default="kullanici")
    kredi = Column(Integer, default=50)
    olusturulma_tarihi = Column(DateTime, default=datetime.now)

class KrediHareketi(Taban):
    __tablename__ = "kredi_hareketleri"

    id = Column(Integer, primary_key=True, index=True)
    kullanici_id = Column(Integer)
    miktar = Column(Integer, nullable=False)
    sebep = Column(String(100))
    referans_asin = Column(String(20))
    olusturulma_tarihi = Column(DateTime, default=datetime.now)

class UrunErisimLogu(Taban):
    __tablename__ = "urun_erisim_loglari"

    id = Column(Integer, primary_key=True, index=True)
    kullanici_id = Column(Integer)
    asin = Column(String(20))
    harcanan_kredi = Column(Integer, default=2)
    olusturulma_tarihi = Column(DateTime, default=datetime.now)

class Favori(Taban):
    __tablename__ = "favoriler"

    id = Column(Integer, primary_key=True, index=True)
    kullanici_id = Column(Integer)
    asin = Column(String(20))
    olusturulma_tarihi = Column(DateTime, default=datetime.now)

# ------------------ SENİN AWS VERİTABANIN ------------------

class Product(Taban):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    asin = Column(String(20), nullable=False)
    title = Column(Text)
    brand = Column(Text)
    category = Column(Text)
    image_url = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

class ProductMetricsTR(Taban):
    __tablename__ = "product_metrics_tr"

    id = Column(Integer, primary_key=True)
    asin = Column(String(20), nullable=False)
    price = Column(Numeric(10, 2))
    rating = Column(Numeric(3, 2))
    review_count = Column(Integer)
    bestseller_rank = Column(Integer)  # BSR
    scraped_at = Column(DateTime, default=datetime.now)

class LatestProductMetricsTR(Taban):
    """latest_product_metrics_tr VIEW'ı - her ASIN için son kayıt"""
    __tablename__ = "latest_product_metrics_tr"

    id = Column(Integer, primary_key=True)
    asin = Column(String(20), nullable=False)
    price = Column(Numeric(10, 2))
    rating = Column(Numeric(3, 2))
    review_count = Column(Integer)
    bestseller_rank = Column(Integer)
    scraped_at = Column(DateTime)