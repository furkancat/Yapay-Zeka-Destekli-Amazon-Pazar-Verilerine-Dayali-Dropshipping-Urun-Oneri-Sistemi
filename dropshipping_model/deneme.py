"""
BSR Tahmin ve Dropshipping Ürün Öneri Sistemi v4.0
- Time-Series Aware Tabular ML (LSTM yok, aggregate features)
- Per-ASIN Window Logic (her ürün kendi son scrape'ine göre)
- Log-Scale BSR Velocity + Percent Change
- Target Leakage Önlemi: Zaman serisi verisi train/test cutoff'a göre filtrelenir
- Mutual Information + Correlation Feature Selection
- Basit DL (2-layer, düşük kapasite - overfitting önlemi)
- Model-Based Scoring (manuel heuristic kaldırıldı)
- Data Leakage Önlemli Kategori İstatistikleri
"""

import os
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import linregress
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import warnings
warnings.filterwarnings('ignore')

# ==================== ÇIKTI DİZİNİ AYARI ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== AYARLAR ====================
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dropshipping2',
    'user': 'postgres',
    'password': ''
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

AZ_VERI_ESIK = 30
KORELASYON_ESIK = 0.90
MI_ESIK = 0.01  # Mutual information threshold


# ==================== 1. VERİ ÇEKME ====================
def veri_cek():
    """SADECE PostgreSQL. ASIN listesi + zaman serisi geçmişi."""
    sorgu_ana = """
    SELECT 
        p.asin, p.title, p.brand, p.category,
        pm.price, pm.rating, pm.review_count, pm.bestseller_rank, pm.scraped_at
    FROM products p
    INNER JOIN latest_product_metrics_tr pm ON p.asin = pm.asin
    WHERE p.category IS NOT NULL AND p.category != ''
      AND pm.bestseller_rank IS NOT NULL;
    """

    # TÜM geçmiş (target leakage önlemi için train/test split sonrası filtrelenir)
    sorgu_zaman = """
    SELECT asin, scraped_at, bestseller_rank, review_count, price
    FROM product_metrics_tr
    ORDER BY asin, scraped_at;
    """

    conn = psycopg2.connect(**DB_CONFIG)
    df_ana = pd.read_sql(sorgu_ana, conn)
    df_zaman = pd.read_sql(sorgu_zaman, conn)
    conn.close()

    print(f"[OK] Ana: {len(df_ana)} ürün, Zaman serisi: {len(df_zaman)} kayıt.")
    return df_ana, df_zaman


# ==================== 2. TRAIN/TEST SPLIT (ASIN bazlı, zaman bilinci) ====================
def veri_ayir(df_ana):
    """
    ÖNCE ASIN'leri split et. Zaman serisi verisi SONRA cutoff'a göre filtrelenir.
    Stratify: kategori bazlı.
    """
    le_cat = LabelEncoder()
    df_ana['kategori_kod'] = le_cat.fit_transform(df_ana['category'])
    df_ana['log_bsr'] = np.log1p(df_ana['bestseller_rank'])

    # ASIN bazlı split (her ürün ya train ya test, asla ikisi birden)
    asin_df = df_ana[['asin', 'kategori_kod', 'category']].drop_duplicates('asin')

    idx_tr, idx_ts = train_test_split(
        asin_df.index, test_size=0.2, random_state=RANDOM_STATE,
        stratify=asin_df['kategori_kod']
    )

    train_asin = set(asin_df.loc[idx_tr, 'asin'])
    test_asin = set(asin_df.loc[idx_ts, 'asin'])

    df_train = df_ana[df_ana['asin'].isin(train_asin)].copy()
    df_test = df_ana[df_ana['asin'].isin(test_asin)].copy()

    # Cutoff tarihi = train setindeki max scraped_at
    # Test ASIN'lerinin zaman serisi verisi SADECE bu tarihten önceki kullanılabilir
    cutoff_tarih = pd.to_datetime(df_train['scraped_at']).max()

    print(f"[OK] Train ASIN: {len(train_asin)}, Test ASIN: {len(test_asin)}")
    print(f"[OK] Zaman cutoff: {cutoff_tarih}")
    print(f"     Test ürünlerinin gelecek scrape'leri GİZLENDİ (target leakage önlendi).")

    return df_train, df_test, le_cat, cutoff_tarih, train_asin, test_asin


# ==================== 3. ZAMAN SERİSİ FEATURE (Per-ASIN, Cutoff'a Göre) ====================
def zaman_serisi_feature(df_zaman, train_asin, test_asin, cutoff_tarih):
    """
    TARGET LEAKAGE ÖNLEME:
    - Train ASIN'ler: tüm geçmiş
    - Test ASIN'ler: SADECE cutoff_tarih'den ÖNCEKİ kayıtlar

    Per-ASIN window: her ürünün kendi SON kaydına göre 7g/önceki 7g.
    BSR velocity: log-scale difference (rank linear fark yanıltıcı).
    """
    df_z = df_zaman.copy()
    df_z['scraped_at'] = pd.to_datetime(df_z['scraped_at'])

    # Cutoff filtreleme (test ASIN'ler için)
    mask_train = df_z['asin'].isin(train_asin)
    mask_test_before = df_z['asin'].isin(test_asin) & (df_z['scraped_at'] <= cutoff_tarih)
    df_z = df_z[mask_train | mask_test_before].copy()

    print(f"[OK] Filtrelenmiş zaman serisi: {len(df_z)} kayıt (test future'ı gizlendi).")

    if len(df_z) == 0:
        return pd.DataFrame({'asin': []})

    # Per-ASIN son tarih
    son_tarihler = df_z.groupby('asin')['scraped_at'].max().reset_index()
    son_tarihler.columns = ['asin', 'asin_son_tarih']
    df_z = df_z.merge(son_tarihler, on='asin')

    # Window: her ürünün kendi son tarihine göre
    df_z['gun_farki'] = (df_z['asin_son_tarih'] - df_z['scraped_at']).dt.total_seconds() / (24*3600)

    son_7 = (df_z['gun_farki'] >= 0) & (df_z['gun_farki'] <= 7)
    onceki_7 = (df_z['gun_farki'] > 7) & (df_z['gun_farki'] <= 14)

    # Aggregate
    ozet = df_z.groupby('asin').agg({
        'bestseller_rank': ['mean', 'std', 'count'],
        'review_count': ['first', 'last'],
        'price': ['std'],
        'scraped_at': ['min', 'max']
    })
    ozet.columns = ['_'.join(c) for c in ozet.columns]
    ozet = ozet.rename(columns={
        'bestseller_rank_mean': 'bsr_14d_mean',
        'bestseller_rank_std': 'bsr_stability',
        'bestseller_rank_count': 'scrape_count',
        'review_count_first': 'review_first',
        'review_count_last': 'review_last',
        'price_std': 'price_volatility',
        'scraped_at_min': 'takip_baslangic',
        'scraped_at_max': 'takip_bitis'
    })

    # Log-scale BSR velocity (per-ASIN)
    df_z['log_bsr'] = np.log1p(df_z['bestseller_rank'])

    son7_log = df_z[son_7].groupby('asin')['log_bsr'].mean()
    onceki7_log = df_z[onceki_7].groupby('asin')['log_bsr'].mean()

    ozet['bsr_velocity_log'] = son7_log.reindex(ozet.index) - onceki7_log.reindex(ozet.index)
    ozet['bsr_velocity_log'] = ozet['bsr_velocity_log'].fillna(0)

    # Percent change (BSR rank üzerinden, ama normalize)
    son7_rank = df_z[son_7].groupby('asin')['bestseller_rank'].mean()
    onceki7_rank = df_z[onceki_7].groupby('asin')['bestseller_rank'].mean()
    pct_change = ((son7_rank - onceki7_rank) / (onceki7_rank + 1e-8)).reindex(ozet.index)
    ozet['bsr_pct_change'] = pct_change.fillna(0)

    # Review growth rate
    ozet['takip_suresi_gun'] = (ozet['takip_bitis'] - ozet['takip_baslangic']).dt.total_seconds() / (24*3600)
    ozet['takip_suresi_gun'] = ozet['takip_suresi_gun'].replace(0, 1)
    ozet['review_growth_rate'] = (ozet['review_last'] - ozet['review_first']) / ozet['takip_suresi_gun']

    # Trend slope (son 3 kayıt, log BSR)
    def son3_trend(grup):
        son3 = grup.tail(3)
        if len(son3) < 2:
            return 0
        x = np.arange(len(son3))
        y = np.log1p(son3['bestseller_rank'].values)
        slope, _, _, _, _ = linregress(x, y)
        return slope

    trend = df_z.groupby('asin').apply(son3_trend)
    ozet['bsr_trend_slope_log'] = trend.reindex(ozet.index).fillna(0)

    # Null handling
    ozet['bsr_stability'] = ozet['bsr_stability'].fillna(0)
    ozet['price_volatility'] = ozet['price_volatility'].fillna(0)
    ozet['review_growth_rate'] = ozet['review_growth_rate'].fillna(0)

    ozet = ozet.reset_index()
    kolonlar = ['asin', 'bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
                'review_growth_rate', 'bsr_stability', 'price_volatility',
                'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean']
    print(f"[OK] Zaman serisi feature: {len(ozet)} ürün.")
    return ozet[kolonlar]


# ==================== 4. ÖN İŞLEME ====================
def on_isleme(df_ana, df_zaman, train_asin, test_asin, cutoff_tarih):
    """rating/review=0, price null kategori median, brand freq encoding."""
    df = df_ana.copy()
    df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')

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

    # Zaman özellikleri (scraped_at'tan)
    simdi = pd.Timestamp.now()
    df['veri_tazeligi_gun'] = (simdi - df['scraped_at']).dt.total_seconds() / (24*3600)
    df['veri_tazeligi_gun'] = df['veri_tazeligi_gun'].fillna(df['veri_tazeligi_gun'].median())
    df['scraped_saat'] = df['scraped_at'].dt.hour.fillna(12)
    df['scraped_hafta_ici'] = df['scraped_at'].dt.weekday.fillna(0)

    # Brand frequency encoding
    brand_freq = df['brand'].value_counts()
    df['brand_freq'] = df['brand'].map(brand_freq).fillna(1)

    # Zaman serisi feature'larını birleştir (leakage önlemli)
    df_z = zaman_serisi_feature(df_zaman, train_asin, test_asin, cutoff_tarih)
    if len(df_z) > 0:
        df = df.merge(df_z, on='asin', how='left')
        z_kolonlar = ['bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
                      'review_growth_rate', 'bsr_stability', 'price_volatility',
                      'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean']
        for kol in z_kolonlar:
            df[kol] = df[kol].fillna(0)
    else:
        for kol in ['bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
                    'review_growth_rate', 'bsr_stability', 'price_volatility',
                    'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean']:
            df[kol] = 0

    print(f"[OK] Ön işleme: {len(df)} satır, {df['category'].nunique()} kategori.")
    return df


# ==================== 5. KATEGORİ GRUPLAMA ====================
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

    # Re-encode
    le = LabelEncoder()
    df['kategori_kod'] = le.fit_transform(df['category'])
    print(f"[OK] Son kategori: {df['category'].nunique()}")
    return df, le


# ==================== 6. KATEGORİ İSTATİSTİKLERİ (SADECE TRAIN) ====================
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


# ==================== 7. ÖZELLİK MÜHENDİSLİĞİ ====================
def ozellik_mühendisliği(df_train, df_test):
    # Price doldurma (train'den)
    kat_fiyat_med = df_train.groupby('category')['price'].median()
    g_fiyat_med = df_train['price'].median()
    for df_ in [df_train, df_test]:
        for kat in df_['category'].unique():
            mask = (df_['category'] == kat) & (df_['price'].isna())
            df_.loc[mask, 'price'] = kat_fiyat_med.get(kat, g_fiyat_med)
        df_['price'] = df_['price'].fillna(g_fiyat_med)

    # Kategori istatistikleri (sadece train)
    ist, g_ort, g_std = kategori_istatistik_hesapla(df_train)
    df_train = istatistik_uygula(df_train, ist, g_ort, g_std)
    df_test = istatistik_uygula(df_test, ist, g_ort, g_std)

    # Normalize hedef
    for df_ in [df_train, df_test]:
        df_['bsr_kategori_zscore'] = (df_['bestseller_rank'] - df_['kat_bsr_ort']) / (df_['kat_bsr_std'] + 1e-8)

    # Etkileşim özellikleri
    for df_ in [df_train, df_test]:
        df_['deger_puani'] = df_['rating'] * np.log1p(df_['review_count'] + 1)
        df_['log_price'] = np.log1p(df_['price'])
        df_['log_review'] = np.log1p(df_['review_count'])
        df_['fiyat_kat_ort_orani'] = df_['price'] / (df_['kat_bsr_ort'] + 1e-8)
        df_['bsr_stabilite_skoru'] = df_['kat_bsr_ort'] / (df_['bsr_stability'] + 1e-8)

    print("[OK] Özellik mühendisliği tamam.")
    return df_train, df_test


# ==================== 8. FEATURE SELECTION (MI + Correlation) ====================
def feature_select(df_train, tum_ozellikler, hedef_kolon='log_bsr'):
    """
    1. Correlation pruning (|r| > 0.90)
    2. Mutual Information filtering (MI < 0.01)
    """
    # 1. Correlation pruning
    corr = df_train[tum_ozellikler].corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high = corr.where(mask)

    drop_corr = set()
    for col in high.columns:
        pairs = high[col].dropna()
        yuksek = pairs[pairs > KORELASYON_ESIK]
        for idx, val in yuksek.items():
            if col not in drop_corr and idx not in drop_corr:
                # Düşük variance olanı at
                v_col = df_train[col].var()
                v_idx = df_train[idx].var()
                drop_corr.add(col if v_col < v_idx else idx)
                print(f"  [CORR] '{col}' vs '{idx}' r={val:.3f} -> düşürüldü.")

    kalan = [c for c in tum_ozellikler if c not in drop_corr]

    # 2. Mutual Information (sadece train)
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


# ==================== 9. MODEL HAZIRLIK ====================
def model_hazirlik(df_train, df_test):
    tum = [
        'price', 'log_price', 'rating', 'review_count', 'log_review',
        'deger_puani', 'fiyat_kat_ort_orani',
        'veri_tazeligi_gun', 'scraped_saat', 'scraped_hafta_ici',
        'brand_freq',
        'bsr_velocity_log', 'bsr_pct_change', 'bsr_trend_slope_log',
        'review_growth_rate', 'bsr_stability', 'price_volatility',
        'scrape_count', 'takip_suresi_gun', 'bsr_14d_mean',
        'bsr_stabilite_skoru',
        'kat_bsr_ort', 'kat_bsr_std', 'kat_urun_sayisi'
    ]

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


# ==================== 10. ML: XGBOOST ====================
def xgboost_egit(X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, ozellikler, df_test):
    X_tr = np.hstack([X_tr_s, X_tr_k])
    X_ts = np.hstack([X_ts_s, X_ts_k])

    # 5-Fold CV
    print("\n[XGBoost] 5-Fold CV...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = []

    for fold, (tr_i, val_i) in enumerate(kf.split(X_tr, X_tr_k[:, 0])):
        m = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        m.fit(X_tr[tr_i], y_tr[tr_i], verbose=False)
        pred = m.predict(X_tr[val_i])
        cv_mae.append(mean_absolute_error(y_tr[val_i], pred))
        print(f"  Fold {fold+1} Log-MAE: {cv_mae[-1]:.4f}")

    print(f"  CV Ort: {np.mean(cv_mae):.4f} (+/- {np.std(cv_mae):.4f})")

    # Final
    final = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    final.fit(X_tr, y_tr, verbose=False)

    y_p_tr = final.predict(X_tr)
    y_p_ts = final.predict(X_ts)
    bsr_pred = np.expm1(y_p_ts)
    bsr_g = np.expm1(y_ts)

    print("\n=== XGBoost SONUÇLARI ===")
    print(f"Train Log-MAE: {mean_absolute_error(y_tr, y_p_tr):.4f}")
    print(f"Test  Log-MAE: {mean_absolute_error(y_ts, y_p_ts):.4f}")
    print(f"Test  R²     : {r2_score(y_ts, y_p_ts):.4f}")
    print(f"Test  BSR-MAE: {mean_absolute_error(bsr_g, bsr_pred):.2f}")

    # Feature Importance
    tum_isim = ozellikler + ['kategori_kod']
    onem = pd.DataFrame({'ozellik': tum_isim, 'onem': final.feature_importances_})
    onem = onem.sort_values('onem', ascending=False)
    print("\n--- XGBoost Özellik Önemleri (Top 15) ---")
    print(onem.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(final, ax=ax, max_num_features=20, importance_type='gain')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_importance_v4.png'), dpi=150)
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_test[ozellikler].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix (v4.0)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_v4.png'), dpi=150)
    plt.close()

    # Kategori bazlı
    df_test = df_test.copy()
    df_test['bsr_tahmin_xgb'] = bsr_pred
    df_test['bsr_gercek'] = bsr_g

    print("\n--- Kategori Bazlı XGBoost BSR-MAE ---")
    for kat in sorted(df_test['category'].unique()):
        mask = df_test['category'] == kat
        if mask.sum() >= 5:
            mae = mean_absolute_error(df_test.loc[mask, 'bsr_gercek'],
                                      df_test.loc[mask, 'bsr_tahmin_xgb'])
            print(f"{kat:35s}: MAE={mae:8.2f} (n={mask.sum()})")

    return final, bsr_pred, onem


# ==================== 11. DL: BASİT 2-LAYER NN ====================
def dl_model_olustur(sayisal_sayisi, kategori_sayisi):
    """Overfitting önlemi: sadece 2 layer, düşük kapasite."""
    emb_dim = min(50, kategori_sayisi // 2)
    print(f"[DL] Kategori: {kategori_sayisi}, Embedding: {emb_dim}")

    inp_s = layers.Input(shape=(sayisal_sayisi,), name='sayisal')
    inp_k = layers.Input(shape=(1,), name='kategori')

    emb = layers.Embedding(kategori_sayisi, emb_dim,
                           embeddings_regularizer=regularizers.l2(0.01))(inp_k)
    emb = layers.Flatten()(emb)

    x = layers.concatenate([inp_s, emb])

    # SADECE 2 layer, düşük kapasite
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(16, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation='linear')(x)

    model = Model([inp_s, inp_k], out)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model


def dl_egit(X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, kategori_sayisi, df_test):
    model = dl_model_olustur(X_tr_s.shape[1], kategori_sayisi)

    cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=5, min_lr=1e-6, verbose=1)
    ]

    print("\n[DL] Eğitim başlıyor (basit 2-layer)...")
    hist = model.fit([X_tr_s, X_tr_k], y_tr, validation_split=0.2,
                     epochs=100, batch_size=32, callbacks=cb, verbose=1)

    y_p_ts = model.predict([X_ts_s, X_ts_k], verbose=0).flatten()
    bsr_pred = np.expm1(y_p_ts)
    bsr_g = np.expm1(y_ts)

    print("\n=== DL (Keras) SONUÇLARI ===")
    print(f"Test Log-MAE: {mean_absolute_error(y_ts, y_p_ts):.4f}")
    print(f"Test R²     : {r2_score(y_ts, y_p_ts):.4f}")
    print(f"Test BSR-MAE: {mean_absolute_error(bsr_g, bsr_pred):.2f}")

    df_test = df_test.copy()
    df_test['bsr_tahmin_dl'] = bsr_pred
    # bsr_gercek yoksa ekle (XGBoost'tan önce çalışırsa diye)
    if 'bsr_gercek' not in df_test.columns:
        df_test['bsr_gercek'] = bsr_g

    print("\n--- Kategori Bazlı DL BSR-MAE ---")
    for kat in sorted(df_test['category'].unique()):
        mask = df_test['category'] == kat
        if mask.sum() >= 5:
            mae = mean_absolute_error(df_test.loc[mask, 'bsr_gercek'],
                                      df_test.loc[mask, 'bsr_tahmin_dl'])
            print(f"{kat:35s}: MAE={mae:8.2f} (n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist.history['loss'], label='Train')
    ax.plot(hist.history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('DL Eğitim Geçmişi (Basit 2-Layer)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dl_training_v4.png'), dpi=150)
    plt.close()

    return model, bsr_pred

# ==================== 12. MODEL-BASED ÖNERİ SİSTEMİ ====================
def oneri_sistemi(df, bsr_tahmin, model_adi):
    """
    Manuel heuristic YOK. Sadece model output kullanılır.
    Öneri skoru = predicted BSR'nin kategori z-score'u.
    Negatif = kategorisinden İYİ satıyor.
    """
    df = df.copy()
    df['bsr_tahmin'] = bsr_tahmin
    df['oneri_skoru'] = (df['bsr_tahmin'] - df['kat_bsr_ort']) / (df['kat_bsr_std'] + 1e-8)
    df['oneri_yuzdelik'] = df.groupby('category')['bsr_tahmin'].rank(pct=True) * 100

    # Fırsat: düşük BSR + düşük velocity_log (iyileşme) + düşük stability (güvenilir)
    df['firsat_skoru'] = (
        -df['oneri_skoru'] * 0.5 +
        (-df['bsr_velocity_log'] * 0.3) +  # Negatif velocity = iyileşme
        (-df['bsr_stability'] * 0.0001)     # Düşük stability = güvenilir
    )

    print(f"\n{'='*70}")
    print(f"  DROPSHIPPING ÖNERİ - {model_adi}")
    print(f"{'='*70}")

    print("\n--- GENEL TOP 10 FIRSAT ÜRÜNÜ ---")
    top10 = df.nlargest(10, 'firsat_skoru')[['asin', 'title', 'category',
                                              'price', 'rating', 'review_count',
                                              'bestseller_rank', 'bsr_tahmin',
                                              'bsr_velocity_log', 'firsat_skoru']]
    print(top10.to_string(index=False))

    print("\n--- HER KATEGORİDEN TOP 2 ---")
    top_kat = df.groupby('category', group_keys=False).apply(
        lambda x: x.nlargest(2, 'firsat_skoru')
    )
    print(top_kat[['category', 'asin', 'price', 'rating',
                   'bestseller_rank', 'bsr_tahmin', 'firsat_skoru']].to_string(index=False))

    return df


# ==================== 13. ANA AKIŞ ====================
def main():
    print("="*70)
    print("BSR TAHMİN v4.0 - Target Leakage Önlemli | Per-ASIN Window | MI Selection")
    print("="*70)

    # 1. Veri çek
    df_ana, df_zaman = veri_cek()

    # 2. ASIN bazlı split + cutoff tarihi (target leakage önlemi)
    df_train_raw, df_test_raw, le_cat, cutoff, train_asin, test_asin = veri_ayir(df_ana)

    # 3. Ön işleme (zaman serisi cutoff'a göre filtrelenmiş)
    df_train = on_isleme(df_train_raw, df_zaman, train_asin, set(), cutoff)
    df_test = on_isleme(df_test_raw, df_zaman, set(), test_asin, cutoff)

    # 4. Kategori grupla (her ikisi ayrı ayrı re-encode edilmeli ama train'den fit)
    # Önce birleştir, sonra fit, sonra split
    df_all = pd.concat([df_train, df_test])
    df_all, le_cat = kategori_grupla(df_all)
    df_train = df_all[df_all['asin'].isin(train_asin)].copy()
    df_test = df_all[df_all['asin'].isin(test_asin)].copy()

    # 5. Özellik mühendisliği
    df_train, df_test = ozellik_mühendisliği(df_train, df_test)

    # 6. Model hazırlık + Feature selection
    (X_tr_s, X_tr_k, y_tr,
     X_ts_s, X_ts_k, y_ts,
     scaler, ozellikler, mi_df) = model_hazirlik(df_train, df_test)

    # 7. XGBoost
    print("\n" + "="*70)
    print("MAKİNE ÖĞRENMESİ: XGBOOST")
    print("="*70)
    xgb_m, xgb_pred, xgb_onem = xgboost_egit(
        X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts,
        ozellikler, df_test
    )

    # 8. DL (basit 2-layer)
    print("\n" + "="*70)
    print("DERİN ÖĞRENME: BASİT 2-LAYER NN")
    print("="*70)
    dl_m, dl_pred = dl_egit(
        X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts,
        len(le_cat.classes_), df_test  # df_test artık bsr_gercek içeriyor
    )

    # 9. Öneri
    df_oneri_xgb = oneri_sistemi(df_test, xgb_pred, "XGBoost")
    df_oneri_dl = oneri_sistemi(df_test, dl_pred, "Deep Learning")

    # 10. Karşılaştırma
    print("\n" + "="*70)
    print("MODEL KARŞILAŞTIRMA")
    print("="*70)
    bsr_g = df_test['bestseller_rank'].values
    mae_x = mean_absolute_error(bsr_g, xgb_pred)
    mae_d = mean_absolute_error(bsr_g, dl_pred)
    r2_x = r2_score(np.log1p(bsr_g), np.log1p(xgb_pred))
    r2_d = r2_score(np.log1p(bsr_g), np.log1p(dl_pred))

    kars = pd.DataFrame({
        'Metrik': ['BSR MAE', 'Log-BSR R²', 'Target Leakage', 'Window Logic', 'Feature Select'],
        'XGBoost': [f"{mae_x:.2f}", f"{r2_x:.4f}", "Önlendi (cutoff)", "Per-ASIN", "MI+Corr"],
        'DL': [f"{mae_d:.2f}", f"{r2_d:.4f}", "Önlendi (cutoff)", "Per-ASIN", "MI+Corr"]
    })
    print(kars.to_string(index=False))

    # 11. Kaydet
    xgb_m.save_model(os.path.join(OUTPUT_DIR, 'bsr_xgb_v4.json'))
    dl_m.save(os.path.join(OUTPUT_DIR, 'bsr_dl_v4.keras'))
    np.save(os.path.join(OUTPUT_DIR, 'encoder_v4.npy'), le_cat.classes_)
    np.save(os.path.join(OUTPUT_DIR, 'scaler_mean_v4.npy'), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, 'scaler_scale_v4.npy'), scaler.scale_)
    mi_df.to_csv(os.path.join(OUTPUT_DIR, 'mi_scores_v4.csv'), index=False)

    print("\n[OK] Kaydedildi: bsr_xgb_v4.json, bsr_dl_v4.keras, *.png, mi_scores_v4.csv")
    print("="*70)


if __name__ == "__main__":
    main()