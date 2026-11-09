"""
BSR Tahmin - Ortak Fonksiyonlar (v5.0)
- products.created_at bazlı zaman penceresi
- Target Leakage Önlemi
- Mutual Information + Correlation Feature Selection
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== AYARLAR ====================
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dropshipping2',
    'user': 'postgres',
    'password': ''
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AZ_VERI_ESIK = 30
KORELASYON_ESIK = 0.90
MI_ESIK = 0.01
LSTM_WINDOW = 90
TARGET_SHIFT_DAYS = 7  # Kaç gün sonrasını tahmin edeceğiz?


# ==================== 1. VERİ ÇEKME (created_at ile) ====================
def veri_cek():
    """products.created_at ile birlikte veri çek."""
    sorgu_ana = """
    SELECT 
        p.asin, p.title, p.brand, p.category, p.created_at,
        pm.price, pm.rating, pm.review_count, pm.bestseller_rank, pm.scraped_at
    FROM products p
    INNER JOIN latest_product_metrics_tr pm ON p.asin = pm.asin
    WHERE p.category IS NOT NULL AND p.category != ''
      AND pm.bestseller_rank IS NOT NULL;
    """

    # SADECE created_at'tan SONRAKİ veriler (ürün yokken veri olamaz)
    sorgu_zaman = """
    SELECT p.asin, pm.scraped_at, pm.bestseller_rank, pm.review_count, pm.price, p.created_at
    FROM products p
    INNER JOIN product_metrics_tr pm ON p.asin = pm.asin
    WHERE pm.scraped_at >= p.created_at
    ORDER BY p.asin, pm.scraped_at;
    """

    conn = psycopg2.connect(**DB_CONFIG)
    df_ana = pd.read_sql(sorgu_ana, conn)
    df_zaman = pd.read_sql(sorgu_zaman, conn)
    conn.close()

    df_ana['created_at'] = pd.to_datetime(df_ana['created_at'])
    df_zaman['created_at'] = pd.to_datetime(df_zaman['created_at'])
    df_zaman['scraped_at'] = pd.to_datetime(df_zaman['scraped_at'])

    print(f"[OK] Ana: {len(df_ana)} ürün, Zaman serisi: {len(df_zaman)} kayıt.")
    return df_ana, df_zaman


# ==================== 2. TRAIN/TEST SPLIT ====================
def veri_ayir(df_ana):
    le_cat = LabelEncoder()
    df_ana['kategori_kod'] = le_cat.fit_transform(df_ana['category'])
    df_ana['log_bsr'] = np.log1p(df_ana['bestseller_rank'])

    asin_df = df_ana[['asin', 'kategori_kod', 'category']].drop_duplicates('asin')

    idx_tr, idx_ts = train_test_split(
        asin_df.index, test_size=0.2, random_state=RANDOM_STATE,
        stratify=asin_df['kategori_kod']
    )

    train_asin = set(asin_df.loc[idx_tr, 'asin'])
    test_asin = set(asin_df.loc[idx_ts, 'asin'])

    df_train = df_ana[df_ana['asin'].isin(train_asin)].copy()
    df_test = df_ana[df_ana['asin'].isin(test_asin)].copy()

    cutoff_tarih = pd.to_datetime(df_train['scraped_at']).max()

    print(f"[OK] Train ASIN: {len(train_asin)}, Test ASIN: {len(test_asin)}")
    print(f"[OK] Zaman cutoff: {cutoff_tarih}")
    print(f"     Test ürünlerinin gelecek scrape'leri GİZLENDİ.")

    return df_train, df_test, le_cat, cutoff_tarih, train_asin, test_asin


# ==================== 3. ZAMAN SERİSİ FEATURE (GELECEK TAHMİNİ UYUMLU) ====================
def zaman_serisi_feature(df_zaman, df_ana, train_asin, test_asin, cutoff_tarih, is_inference=False):
    df_z = df_zaman.copy()
    mask_train = df_z['asin'].isin(train_asin)
    mask_test = df_z['asin'].isin(test_asin) & (df_z['scraped_at'] <= cutoff_tarih)
    df_z = df_z[mask_train | mask_test].copy()

    df_target_dates = df_ana[['asin', 'scraped_at']].rename(columns={'scraped_at': 'target_date'})
    df_z = df_z.merge(df_target_dates, on='asin')
    
    if not is_inference:
        df_z = df_z[df_z['scraped_at'] <= (df_z['target_date'] - pd.Timedelta(days=TARGET_SHIFT_DAYS))]

    if len(df_z) == 0:
        return pd.DataFrame({'asin': []})

    son_tarihler = df_z.groupby('asin')['scraped_at'].max().reset_index()
    son_tarihler.columns = ['asin', 'asin_son_tarih']
    df_z = df_z.merge(son_tarihler, on='asin')

    df_z['urun_yasi_gun'] = (df_z['asin_son_tarih'] - df_z['created_at']).dt.total_seconds() / (24*3600)
    df_z['gun_farki'] = (df_z['asin_son_tarih'] - df_z['scraped_at']).dt.total_seconds() / (24*3600)

    son_7 = (df_z['gun_farki'] >= 0) & (df_z['gun_farki'] <= 7)
    onceki_7 = (df_z['gun_farki'] > 7) & (df_z['gun_farki'] <= 14)

    # --- YENİ: LAG ÖZELLİKLERİ (Son 3 günün net verisi) ---
    # Sıralamayı garanti altına al
    df_z = df_z.sort_values(['asin', 'scraped_at'])
    
    # Lag'leri hesapla
    df_z['bsr_lag_1'] = df_z.groupby('asin')['bestseller_rank'].shift(1)
    df_z['bsr_lag_2'] = df_z.groupby('asin')['bestseller_rank'].shift(2)
    df_z['bsr_lag_3'] = df_z.groupby('asin')['bestseller_rank'].shift(3)
    
    # NaN değerleri o anki BSR ile veya grubun ortalamasıyla doldur
    df_z['bsr_lag_1'] = df_z['bsr_lag_1'].fillna(df_z['bestseller_rank'])
    df_z['bsr_lag_2'] = df_z['bsr_lag_2'].fillna(df_z['bestseller_rank'])
    df_z['bsr_lag_3'] = df_z['bsr_lag_3'].fillna(df_z['bestseller_rank'])

    # Aggregate işlemleri
    ozet = df_z.groupby('asin').agg({
        'bestseller_rank': ['mean', 'std', 'count'],
        'review_count': ['first', 'last'],
        'price': ['std'],
        'scraped_at': ['min', 'max'],
        'urun_yasi_gun': 'first',
        'bsr_lag_1': 'last', # Son scrape tarihindeki lag'i al
        'bsr_lag_2': 'last',
        'bsr_lag_3': 'last'
    })
    ozet.columns = ['_'.join(c) if isinstance(c, tuple) and c[1] else c[0] for c in ozet.columns] # Tuple handle
    
    # Sütun isimlerini düzelt (_last eklenenleri temizle)
    ozet = ozet.rename(columns={
        'bestseller_rank_mean': 'bsr_14d_mean',
        'bestseller_rank_std': 'bsr_stability',
        'bestseller_rank_count': 'scrape_count',
        'review_count_first': 'review_first',
        'review_count_last': 'review_last',
        'price_std': 'price_volatility',
        'scraped_at_min': 'takip_baslangic',
        'scraped_at_max': 'takip_bitis',
        'urun_yasi_gun_first': 'urun_yasi_gun',
        'bsr_lag_1_last': 'bsr_lag_1',
        'bsr_lag_2_last': 'bsr_lag_2',
        'bsr_lag_3_last': 'bsr_lag_3'
    })

    # ... (log_bsr, pct_change, growth rate, trend slope hesaplamaları aynı kalıyor)
    df_z['log_bsr'] = np.log1p(df_z['bestseller_rank'])
    son7_log = df_z[son_7].groupby('asin')['log_bsr'].mean()
    onceki7_log = df_z[onceki_7].groupby('asin')['log_bsr'].mean()
    ozet['bsr_velocity_log'] = son7_log.reindex(ozet.index) - onceki7_log.reindex(ozet.index)
    ozet['bsr_velocity_log'] = ozet['bsr_velocity_log'].fillna(0)

    # Percent change
    son7_rank = df_z[son_7].groupby('asin')['bestseller_rank'].mean()
    onceki7_rank = df_z[onceki_7].groupby('asin')['bestseller_rank'].mean()
    pct_change = ((son7_rank - onceki7_rank) / (onceki7_rank + 1e-8)).reindex(ozet.index)
    ozet['bsr_pct_change'] = pct_change.fillna(0)

    # Review growth rate
    ozet['takip_suresi_gun'] = (ozet['takip_bitis'] - ozet['takip_baslangic']).dt.total_seconds() / (24*3600)
    ozet['takip_suresi_gun'] = ozet['takip_suresi_gun'].replace(0, 1)
    ozet['review_growth_rate'] = (ozet['review_last'] - ozet['review_first']) / ozet['takip_suresi_gun']

    # Trend slope
    def son3_trend(grup):
        son3 = grup.tail(3)
        if len(son3) < 2:
            return 0
        x = np.arange(len(son3))
        y = np.log1p(son3['bestseller_rank'].values)
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(x, y)
        return slope

    trend = df_z.groupby('asin').apply(son3_trend)
    ozet['bsr_trend_slope_log'] = trend.reindex(ozet.index).fillna(0)

    # Null handling
    ozet['bsr_stability'] = ozet['bsr_stability'].fillna(0)
    ozet['price_volatility'] = ozet['price_volatility'].fillna(0)
    ozet['review_growth_rate'] = ozet['review_growth_rate'].fillna(0)
    
    # Lag'lerde oluşabilecek olası eksiklikleri doldur
    ozet['bsr_lag_1'] = ozet['bsr_lag_1'].fillna(0)
    ozet['bsr_lag_2'] = ozet['bsr_lag_2'].fillna(0)
    ozet['bsr_lag_3'] = ozet['bsr_lag_3'].fillna(0)

    ozet = ozet.reset_index()
    
    # Kolon listesine lag'leri ekle
    kolonlar = ['asin', 'bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
                'review_growth_rate', 'bsr_stability', 'price_volatility',
                'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean', 'urun_yasi_gun',
                'bsr_lag_1', 'bsr_lag_2', 'bsr_lag_3'] # YENİ
    
    print(f"[OK] Zaman serisi feature: {len(ozet)} ürün.")
    return ozet[kolonlar]

# ==================== 4. LSTM SEQUENCE (GELECEK TAHMİNİ UYUMLU) ====================
def zaman_serisi_sequence(df_zaman, df_ana, train_asin, test_asin, cutoff_tarih, is_inference=False, window=LSTM_WINDOW):
    """
    Hedef kaydırma (Target Shifting) uygulandığı için LSTM'e giren dizinin son 7 günü
    eğitim aşamasında gizlenir. Böylece model geleceği görmeden tahmine zorlanır.
    """
    df_z = df_zaman.copy()

    mask_train = df_z['asin'].isin(train_asin)
    mask_test = df_z['asin'].isin(test_asin) & (df_z['scraped_at'] <= cutoff_tarih)
    df_z = df_z[mask_train | mask_test].copy()

    # 7 Günlük Target Shift Mantığı
    df_target_dates = df_ana[['asin', 'scraped_at']].rename(columns={'scraped_at': 'target_date'})
    df_z = df_z.merge(df_target_dates, on='asin')
    
    if not is_inference:
        df_z = df_z[df_z['scraped_at'] <= (df_z['target_date'] - pd.Timedelta(days=TARGET_SHIFT_DAYS))]

    print(f"[OK] Sequence veri: {len(df_z)} kayıt (7 günlük gelecek gizlendi).")

    if len(df_z) == 0:
        return {}

    df_z['scraped_at'] = pd.to_datetime(df_z['scraped_at'])
    df_z['created_at'] = pd.to_datetime(df_z['created_at'])

    seq_dict = {}
    for asin, grup in df_z.groupby('asin'):
        grup = grup.sort_values('scraped_at').set_index('scraped_at')
        gunluk = grup.resample('D').last()[['bestseller_rank', 'price', 'review_count']]
        gunluk = gunluk.ffill()

        degerler = gunluk.values

        if len(degerler) >= window:
            arr = degerler[-window:]
        else:
            pad = np.zeros((window - len(degerler), 3))
            arr = np.vstack([pad, degerler])

        arr = np.nan_to_num(arr, nan=0.0)
        seq_dict[asin] = arr

    print(f"[OK] Sequence: {len(seq_dict)} ürün, shape=({window}, 3)")
    return seq_dict


# ==================== 5. ÖN İŞLEME ====================
def on_isleme(df_ana, df_zaman, train_asin, test_asin, cutoff_tarih, is_inference=False):
    df = df_ana.copy()
    df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce').fillna(0)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['bestseller_rank'] = pd.to_numeric(df['bestseller_rank'], errors='coerce')

    baslangic = len(df)
    df = df.dropna(subset=['bestseller_rank'])
    if len(df) < baslangic:
        print(f"[UYARI] {baslangic - len(df)} satır BSR NULL atıldı.")

    df['category'] = df['category'].astype(str).str.strip()
    df = df[df['category'] != '']

    simdi = pd.Timestamp.now()
    df['veri_tazeligi_gun'] = (simdi - df['scraped_at']).dt.total_seconds() / (24*3600)
    df['veri_tazeligi_gun'] = df['veri_tazeligi_gun'].fillna(df['veri_tazeligi_gun'].median())
    df['scraped_saat'] = df['scraped_at'].dt.hour.fillna(12)
    df['scraped_hafta_ici'] = df['scraped_at'].dt.weekday.fillna(0)

    brand_freq = df['brand'].value_counts()
    df['brand_freq'] = df['brand'].map(brand_freq).fillna(1)

    # DİKKAT: df_ana ve is_inference parametreleri eklendi!
    df_z = zaman_serisi_feature(df_zaman, df_ana, train_asin, test_asin, cutoff_tarih, is_inference)
    
    if len(df_z) > 0:
        df = df.merge(df_z, on='asin', how='left')
        z_kolonlar = ['bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
                      'review_growth_rate', 'bsr_stability', 'price_volatility',
                      'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean', 'urun_yasi_gun']
        for kol in z_kolonlar:
            df[kol] = df[kol].fillna(0)
    else:
        for kol in ['bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
                    'review_growth_rate', 'bsr_stability', 'price_volatility',
                    'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean', 'urun_yasi_gun']:
            df[kol] = 0

    df['urun_yasi_gun'] = (simdi - df['created_at']).dt.total_seconds() / (24*3600)
    df['urun_yasi_gun'] = df['urun_yasi_gun'].fillna(df['urun_yasi_gun'].median())

    print(f"[OK] Ön işleme: {len(df)} satır, {df['category'].nunique()} kategori.")
    return df


# ==================== 6. KATEGORİ GRUPLAMA ====================
def kategori_grupla(df):
    sayilar = df['category'].value_counts()
    print("\n--- Kategori Dağılımı ---")
    print(sayilar.to_string())

    harita = {
        'Sağlık ve Kişisel Bakım': 'Kişisel Bakım ve Kozmetik',
        'Ofis Ürünleri': 'Ofis ve Kırtasiye',
        'Kitap': 'Diğer',
        'Müzik Enstrümanları ve DJ': 'Diğer',
        'Bahçe': 'Diğer',
        '': 'Diğer'
    }
    df['category_raw'] = df['category']
    df['category'] = df['category'].replace(harita)

    yeni = df['category'].value_counts()
    az = yeni[yeni < AZ_VERI_ESIK].index.tolist()
    if 'Diğer' not in az and len(az) > 0:
        df.loc[df['category'].isin(az), 'category'] = 'Diğer'
        print(f"[OK] {len(az)} az verili kategori 'Diğer' yapıldı.")

    le = LabelEncoder()
    df['kategori_kod'] = le.fit_transform(df['category'])
    print(f"[OK] Son kategori: {df['category'].nunique()}")
    return df, le


# ==================== 7. KATEGORİ İSTATİSTİKLERİ ====================
def kategori_istatistik_hesapla(df_train):
    ist = df_train.groupby('category')['bestseller_rank'].agg([
        ('kat_bsr_ort', 'mean'),
        ('kat_bsr_std', 'std'),
        ('kat_bsr_min', 'min'),
        ('kat_bsr_max', 'max'),
        ('kat_urun_sayisi', 'count')
    ])
    g_ort = df_train['bestseller_rank'].mean()
    g_std = df_train['bestseller_rank'].std()
    ist['kat_bsr_std'] = ist['kat_bsr_std'].fillna(g_std)
    return ist, g_ort, g_std


def istatistik_uygula(df, ist, g_ort, g_std):
    df = df.merge(ist, left_on='category', right_index=True, how='left')
    df['kat_bsr_ort'] = df['kat_bsr_ort'].fillna(g_ort)
    df['kat_bsr_std'] = df['kat_bsr_std'].fillna(g_std)
    df['kat_bsr_min'] = df['kat_bsr_min'].fillna(0)
    df['kat_bsr_max'] = df['kat_bsr_max'].fillna(g_ort * 2)
    df['kat_urun_sayisi'] = df['kat_urun_sayisi'].fillna(1)
    return df


# ==================== 8. ÖZELLİK MÜHENDİSLİĞİ ====================
def ozellik_mühendisliği(df_train, df_test):
    kat_fiyat_med = df_train.groupby('category')['price'].median()
    g_fiyat_med = df_train['price'].median()
    for df_ in [df_train, df_test]:
        for kat in df_['category'].unique():
            mask = (df_['category'] == kat) & (df_['price'].isna())
            df_.loc[mask, 'price'] = kat_fiyat_med.get(kat, g_fiyat_med)
        df_['price'] = df_['price'].fillna(g_fiyat_med)

    ist, g_ort, g_std = kategori_istatistik_hesapla(df_train)
    df_train = istatistik_uygula(df_train, ist, g_ort, g_std)
    df_test = istatistik_uygula(df_test, ist, g_ort, g_std)

    for df_ in [df_train, df_test]:
        df_['bsr_kategori_zscore'] = (df_['bestseller_rank'] - df_['kat_bsr_ort']) / (df_['kat_bsr_std'] + 1e-8)

    for df_ in [df_train, df_test]:
        df_['deger_puani'] = df_['rating'] * np.log1p(df_['review_count'] + 1)
        df_['log_price'] = np.log1p(df_['price'])
        df_['log_review'] = np.log1p(df_['review_count'])
        df_['fiyat_kat_ort_orani'] = df_['price'] / (df_['kat_bsr_ort'] + 1e-8)
        df_['bsr_stabilite_skoru'] = df_['kat_bsr_ort'] / (df_['bsr_stability'] + 1e-8)

    print("[OK] Özellik mühendisliği tamam.")
    return df_train, df_test


# ==================== 9. FEATURE SELECTION ====================
def feature_select(df_train, tum_ozellikler, hedef_kolon='log_bsr'):
    corr = df_train[tum_ozellikler].corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask)

    drop_corr = set()
    for col in high.columns:
        pairs = high[col].dropna()
        yuksek = pairs[pairs > KORELASYON_ESIK]
        for idx, val in yuksek.items():
            if col not in drop_corr and idx not in drop_corr:
                v_col = df_train[col].var()
                v_idx = df_train[idx].var()
                drop_corr.add(col if v_col < v_idx else idx)
                print(f"  [CORR] '{col}' vs '{idx}' r={val:.3f} -> düşürüldü.")

    kalan = [c for c in tum_ozellikler if c not in drop_corr]

    X_mi = df_train[kalan].fillna(0)
    y_mi = df_train[hedef_kolon].values

    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({'ozellik': kalan, 'mi': mi_scores}).sort_values('mi', ascending=False)

    print("\n--- Mutual Information Skorları ---")
    print(mi_df.to_string(index=False))

    dusuk_mi = mi_df[mi_df['mi'] < MI_ESIK]['ozellik'].tolist()
    if dusuk_mi:
        print(f"\n[MI] Düşük MI ile düşürülen: {dusuk_mi}")

    final = [c for c in kalan if c not in dusuk_mi]
    print(f"\n[OK] Feature selection: {len(tum_ozellikler)} -> {len(final)} özellik.")
    return final, mi_df


# ==================== 10. MODEL HAZIRLIK (XGBoost) ====================
def model_hazirlik(df_train, df_test):
    tum = [
        'price', 'log_price', 'rating', 'review_count', 'log_review',
        'deger_puani', 'fiyat_kat_ort_orani',
        'veri_tazeligi_gun', 'brand_freq', 'urun_yasi_gun', 
        'bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
        'review_growth_rate', 'bsr_stability', 'price_volatility',
        'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean',
        'bsr_stabilite_skoru',
        'kat_bsr_ort', 'kat_bsr_std', 'kat_urun_sayisi',
        'bsr_lag_1', 'bsr_lag_2', 'bsr_lag_3' 
    ]

    # Fonksiyonun geri kalanı aynı...
    final_ozellikler, mi_df = feature_select(df_train, tum)
    
    X_tr_s = df_train[final_ozellikler].values
    X_ts_s = df_test[final_ozellikler].values
    X_tr_k = df_train[['kategori_kod']].values
    X_ts_k = df_test[['kategori_kod']].values
    y_tr = df_train['log_bsr'].values
    y_ts = df_test['log_bsr'].values

    scaler = StandardScaler()
    X_tr_s_s = scaler.fit_transform(X_tr_s)
    X_ts_s_s = scaler.transform(X_ts_s)

    return (X_tr_s_s, X_tr_k, y_tr, X_ts_s_s, X_ts_k, y_ts,
            scaler, final_ozellikler, mi_df)

# ==================== 11. MODEL HAZIRLIK (LSTM) ====================
def model_hazirlik_lstm(df_train, df_test, seq_dict):
    tum = [
        'price', 'log_price', 'rating', 'review_count', 'log_review',
        'deger_puani', 'fiyat_kat_ort_orani',
        'veri_tazeligi_gun', 'brand_freq', 'urun_yasi_gun',
        'bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
        'review_growth_rate', 'bsr_stability', 'price_volatility',
        'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean',
        'bsr_stabilite_skoru',
        'kat_bsr_ort', 'kat_bsr_std', 'kat_urun_sayisi'
    ]

    # Fonksiyonun geri kalanı aynı...
    # (LSTM Masking'i ayrı bir dosyada yapacağız)
    final_ozellikler, mi_df = feature_select(df_train, tum)

    X_tr_s = df_train[final_ozellikler].values
    X_ts_s = df_test[final_ozellikler].values
    X_tr_k = df_train[['kategori_kod']].values
    X_ts_k = df_test[['kategori_kod']].values
    y_tr = df_train['log_bsr'].values
    y_ts = df_test['log_bsr'].values

    scaler = StandardScaler()
    X_tr_s_s = scaler.fit_transform(X_tr_s)
    X_ts_s_s = scaler.transform(X_ts_s)

    X_tr_seq = np.array([seq_dict.get(a, np.zeros((LSTM_WINDOW, 3))) for a in df_train['asin']])
    X_ts_seq = np.array([seq_dict.get(a, np.zeros((LSTM_WINDOW, 3))) for a in df_test['asin']])

    seq_scaler = StandardScaler()
    seq_flat = X_tr_seq.reshape(-1, 3)
    seq_scaler.fit(seq_flat)
    X_tr_seq_s = seq_scaler.transform(seq_flat).reshape(X_tr_seq.shape)
    X_ts_seq_s = seq_scaler.transform(X_ts_seq.reshape(-1, 3)).reshape(X_ts_seq.shape)

    return (X_tr_s_s, X_tr_k, y_tr, X_ts_s_s, X_ts_k, y_ts,
            scaler, final_ozellikler, mi_df, X_tr_seq_s, X_ts_seq_s, seq_scaler)