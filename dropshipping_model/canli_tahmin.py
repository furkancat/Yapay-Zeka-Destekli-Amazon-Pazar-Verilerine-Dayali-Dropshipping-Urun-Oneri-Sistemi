"""
BSR Tahmin - CANLI SİSTEM (Inference)
Veritabanındaki TÜM ürünler için bugünün en son verilerini kullanarak
gerçek anlamda 7 gün sonrasını tahmin eder. (Arayüz İçin)
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
import joblib # YENİ
import warnings
warnings.filterwarnings('ignore')

from common import (
    veri_cek, veri_ayir, on_isleme, kategori_grupla,
    ozellik_mühendisliği, model_hazirlik, model_hazirlik_lstm,
    zaman_serisi_sequence, OUTPUT_DIR, LSTM_WINDOW
)

def firsat_skoru_hesapla(df, bsr_tahmin, prefix):
    df_temp = df.copy()
    df_temp[f'tahmin_{prefix}'] = np.round(bsr_tahmin).astype(int)
    
    # 1. Kategori Z-Skoru
    oneri_skoru = (df_temp[f'tahmin_{prefix}'] - df_temp['kat_bsr_ort']) / (df_temp['kat_bsr_std'] + 1e-8)
    
    # 2. İyileşme/Kötüleşme Oranı (YENİ EKLENEN FİLTRE)
    # Eğer ürün geriliyorsa (Tahmin > Mevcut), bu oran negatif olur ve skoru ciddi şekilde cezalandırır.
    # Eğer ürün iyileşiyorsa (Tahmin < Mevcut), bu oran pozitif olur ve skoru şahlandırır.
    gelisim_farki = df_temp['bestseller_rank'] - df_temp[f'tahmin_{prefix}']
    gelisim_orani = gelisim_farki / (df_temp['bestseller_rank'] + 1e-8)
    
    df_temp[f'skor_{prefix}'] = (
        (-oneri_skoru * 0.3) +             # Kategori içi gücünün ağırlığını %50'den %30'a çektik
        (gelisim_orani * 0.6) +            # Asıl ağırlığı (%60) ürünün kendini ne kadar geliştirdiğine verdik
        (-df_temp['bsr_stability'] * 0.0001)
    )
    return df_temp[f'tahmin_{prefix}'], df_temp[f'skor_{prefix}']

def main():
    print("="*70)
    print("BSR TAHMİN - CANLI SİSTEM (TÜM ÜRÜNLER İÇİN)")
    print("="*70)

    # 1. Verileri Çek
    df_ana, df_zaman = veri_cek()
    tum_asinler = set(df_ana['asin'].unique())
    print(f"Toplam {len(tum_asinler)} ürün işleniyor...")

    # Doğru 22 özelliği bulabilmesi için eski eğitim setini de "şablon" olarak ayırıyoruz
    df_train_raw, _, le_cat, cutoff, train_asin, _ = veri_ayir(df_ana)

    # 2. Ön İşleme
    # Eğitim şablonu (Sadece model_hazirlik içindeki Scaler ve Feature Selection'ın doğru çalışması için mecburi)
    df_train = on_isleme(df_train_raw, df_zaman, train_asin, set(), cutoff, is_inference=False)
    
    # Asıl Canlı Veri (is_inference=True -> Geleceği gizle, bugünün en taze verisini al)
    df_canli = on_isleme(df_ana, df_zaman, set(), tum_asinler, cutoff, is_inference=True)

    # Kategorileri kodlamak için geçici birleştirme
    df_all = pd.concat([df_train, df_canli])
    df_all, le_cat = kategori_grupla(df_all)
    
    # Tekrar ayır
    df_train = df_all.iloc[:len(df_train)].copy()
    df_canli = df_all.iloc[len(df_train):].copy()

    # Özellik mühendisliği
    df_train, df_canli = ozellik_mühendisliği(df_train, df_canli)

    # 3. Model Hazırlık (İşte Sihrin Olduğu Yer)
    # Fonksiyon df_train'e bakıp 22 özelliği seçecek ve scaler'ı ayarlayacak,
    # ama dönüştürme işlemini bizim hedefimiz olan X_canli'ye uygulayacak!
    (_, _, _, X_canli_s, X_canli_k, _, _, _, _) = model_hazirlik(df_train, df_canli)

    # 4. LSTM İçin Hazırlık
    seq_dict = zaman_serisi_sequence(df_zaman, df_ana, train_asin, tum_asinler, cutoff, is_inference=True, window=LSTM_WINDOW)
    (_, _, _, X_canli_s_l, X_canli_k_l, _, _, _, _, _, X_canli_seq, _) = model_hazirlik_lstm(df_train, df_canli, seq_dict)

    df_sonuc = df_canli.copy()
    print("\nModeller yükleniyor ve tüm ürünler için gerçek 7 gün sonrası tahmin ediliyor...")

    # --- XGBoost ---
    try:
        xgb_m = xgb.XGBRegressor()
        xgb_m.load_model(os.path.join(OUTPUT_DIR, 'bsr_xgb_v5.json'))
        X_canli_xgb = np.hstack([X_canli_s, X_canli_k])
        xgb_pred = np.expm1(xgb_m.predict(X_canli_xgb))
        df_sonuc['tahmin_xgb'], df_sonuc['skor_xgb'] = firsat_skoru_hesapla(df_sonuc, xgb_pred, 'xgb')
        print("[OK] XGBoost tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] XGBoost yüklenemedi: {e}")

    # --- CatBoost ---
    try:
        cb_m = CatBoostRegressor()
        cb_m.load_model(os.path.join(OUTPUT_DIR, 'bsr_catboost_v6.cbm'))
        X_canli_cb = pd.DataFrame(X_canli_s)
        X_canli_cb['kategori'] = X_canli_k.astype(int)
        cb_pred = np.expm1(cb_m.predict(X_canli_cb))
        df_sonuc['tahmin_cb'], df_sonuc['skor_cb'] = firsat_skoru_hesapla(df_sonuc, cb_pred, 'cb')
        print("[OK] CatBoost tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] CatBoost yüklenemedi: {e}")

    # --- LightGBM ---
    try:
        lgb_m = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, 'bsr_lgbm_v6.txt'))
        X_canli_lgb = np.hstack([X_canli_s, X_canli_k])
        lgb_pred = np.expm1(lgb_m.predict(X_canli_lgb))
        df_sonuc['tahmin_lgb'], df_sonuc['skor_lgb'] = firsat_skoru_hesapla(df_sonuc, lgb_pred, 'lgb')
        print("[OK] LightGBM tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] LightGBM yüklenemedi: {e}")

    # --- LSTM ---
    try:
        lstm_m = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'bsr_lstm_v5.keras'))
        lstm_pred_log = lstm_m.predict([X_canli_seq, X_canli_s_l, X_canli_k_l], verbose=0).flatten()
        lstm_pred = np.expm1(lstm_pred_log)
        df_sonuc['tahmin_lstm'], df_sonuc['skor_lstm'] = firsat_skoru_hesapla(df_sonuc, lstm_pred, 'lstm')
        print("[OK] LSTM tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] LSTM yüklenemedi: {e}")

    # ==========================================
    # ENSEMBLE (ORTALAMA) VE TIE-BREAKER MANTIĞI
    # ==========================================
    print("\nMeta-Learner ile ortak tahmin hesaplanıyor...")
    
    stack_df = pd.DataFrame()
    if 'tahmin_xgb' in df_sonuc.columns: stack_df['xgb'] = np.log1p(df_sonuc['tahmin_xgb'])
    if 'tahmin_cb' in df_sonuc.columns:  stack_df['cb'] = np.log1p(df_sonuc['tahmin_cb'])
    if 'tahmin_lgb' in df_sonuc.columns: stack_df['lgb'] = np.log1p(df_sonuc['tahmin_lgb'])
    if 'tahmin_lstm' in df_sonuc.columns: stack_df['lstm'] = np.log1p(df_sonuc['tahmin_lstm'])
    
    stack_df = stack_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    
    try:
        # Önceden evaluate.py'de eğitilmiş modeli yükle
        meta_model = joblib.load(os.path.join(OUTPUT_DIR, 'meta_learner_ridge.pkl'))
        meta_pred_log = meta_model.predict(stack_df)
        df_sonuc['bsr_tahmin_ortalama'] = np.round(np.expm1(meta_pred_log)).astype(int)
        print("[OK] Stacking Meta-Learner başarıyla çalıştı.")
    except Exception as e:
        print(f"[UYARI] Meta-Learner yüklenemedi, düz ortalamaya dönülüyor: {e}")
        # Hata durumunda Fallback (Düz Ortalama)
        tahmin_kolonlari = ['tahmin_xgb', 'tahmin_cb', 'tahmin_lgb', 'tahmin_lstm']
        df_sonuc['bsr_tahmin_ortalama'] = np.round(df_sonuc[tahmin_kolonlari].mean(axis=1)).astype(int)

    # Yeni Fırsat Skoru'nu tahmin üzerinden dinamik olarak hesapla
    df_sonuc['bsr_tahmin_ortalama'], df_sonuc['firsat_skoru_ortalama'] = firsat_skoru_hesapla(df_sonuc, df_sonuc['bsr_tahmin_ortalama'].values, 'ortalama')

    # Adaletli Sıralama (Tie-Breaker)
    df_sonuc = df_sonuc.sort_values(
        by=['firsat_skoru_ortalama', 'bestseller_rank', 'review_count'],
        ascending=[False, True, False]
    )
    df_sonuc['oneri_yuzdelik'] = df_sonuc.groupby('category')['firsat_skoru_ortalama'].rank(pct=True) * 100

    print("\n" + "="*70)
    print("  CANLI SİSTEM - TOP 10 FIRSAT ÜRÜNÜ (TÜM VERİTABANI)")
    print("="*70)
    top10 = df_sonuc.head(10)[['asin', 'category', 'price', 'bestseller_rank', 'bsr_tahmin_ortalama', 'firsat_skoru_ortalama']]
    print(top10.to_string(index=False))

    csv_yolu = os.path.join(OUTPUT_DIR, 'oneri_ensemble.csv')
    df_sonuc.to_csv(csv_yolu, index=False)
    print(f"\n[OK] Tüm ürünler arayüz için kaydedildi: {csv_yolu}")
    print("="*70)

if __name__ == "__main__":
    main()