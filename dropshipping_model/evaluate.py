"""
BSR Tahmin - Ensemble (Ortak Akıl) ve Değerlendirme (v6.0)
4 Modeli (XGBoost, LightGBM, CatBoost, LSTM) birleştirip
Tie-Breaker mantığıyla nihai Fırsat Skorunu hesaplar.
Tüm modellerin karşılaştırmalı MAE/R² tablosunu sunar.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge # YENİ EKLENDİ
import joblib # Modeli kaydetmek için
import warnings
warnings.filterwarnings('ignore')

from common import (
    veri_cek, veri_ayir, on_isleme, kategori_grupla,
    ozellik_mühendisliği, model_hazirlik, model_hazirlik_lstm,
    zaman_serisi_sequence, OUTPUT_DIR, LSTM_WINDOW
)

def firsat_skoru_hesapla(df, bsr_tahmin, prefix):
    """Modelin tahmini ile o modele özel fırsat skorunu hesaplar"""
    df_temp = df.copy()
    df_temp[f'tahmin_{prefix}'] = np.round(bsr_tahmin).astype(int)
    
    oneri_skoru = (df_temp[f'tahmin_{prefix}'] - df_temp['kat_bsr_ort']) / (df_temp['kat_bsr_std'] + 1e-8)
    df_temp[f'skor_{prefix}'] = (
        -oneri_skoru * 0.5 +
        (-df_temp['bsr_velocity_log'] * 0.3) +
        (-df_temp['bsr_stability'] * 0.0001)
    )
    return df_temp[f'tahmin_{prefix}'], df_temp[f'skor_{prefix}']

def karsilastirma(df_sonuc, preds_dict):
    """Tüm modellerin tahminlerini alır ve yatay MAE/R² tablosu çizer"""
    bsr_g = df_sonuc['bsr_gercek'].values
    log_bsr_g = np.log1p(bsr_g)
    
    # Tablo için temel iskelet
    kars_dict = {
        'Metrik': ['BSR MAE', 'Log-BSR MAE', 'Log-BSR R²']
    }
    
    # Modelleri sütun olarak ekle
    for model_adi, pred in preds_dict.items():
        if pred is None or np.isnan(pred).all():
            continue
            
        log_pred = np.log1p(pred)
        mae = mean_absolute_error(bsr_g, pred)
        log_mae = mean_absolute_error(log_bsr_g, log_pred)
        r2 = r2_score(log_bsr_g, log_pred)
        
        # Her modelin sütun verileri
        kars_dict[model_adi] = [f"{mae:.2f}", f"{log_mae:.4f}", f"{r2:.4f}"]
        
    kars = pd.DataFrame(kars_dict)

    print("\n" + "="*85)
    print("MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*85)
    print(kars.to_string(index=False))

def main():
    print("="*70)
    print("BSR TAHMİN v6.0 - ENSEMBLE (ORTAK AKIL) DEĞERLENDİRME")
    print("="*70)

    # 1. Verileri Çek
    df_ana, df_zaman = veri_cek()
    df_train_raw, df_test_raw, le_cat, cutoff, train_asin, test_asin = veri_ayir(df_ana)

    # 2. Ön İşleme (Test verisi üzerinden değerlendirme)
    df_train = on_isleme(df_train_raw, df_zaman, train_asin, set(), cutoff, is_inference=False)
    df_test = on_isleme(df_test_raw, df_zaman, set(), test_asin, cutoff, is_inference=False)

    df_all = pd.concat([df_train, df_test])
    df_all, le_cat = kategori_grupla(df_all)
    df_train = df_all[df_all['asin'].isin(train_asin)].copy()
    df_test = df_all[df_all['asin'].isin(test_asin)].copy()

    df_train, df_test = ozellik_mühendisliği(df_train, df_test)

    # 3. Tablo (Ağaç) Modelleri İçin Hazırlık
    (X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts,
     scaler, ozellikler, mi_df) = model_hazirlik(df_train, df_test)

    # 4. LSTM İçin Hazırlık
    seq_dict = zaman_serisi_sequence(df_zaman, df_ana, train_asin, test_asin, cutoff, is_inference=False, window=LSTM_WINDOW)
    (X_tr_s_l, X_tr_k_l, y_tr_l, X_ts_s_l, X_ts_k_l, y_ts_l,
     scaler_l, ozellikler_l, mi_df_l, X_tr_seq, X_ts_seq, seq_scaler) = model_hazirlik_lstm(df_train, df_test, seq_dict)

    df_sonuc = df_test.copy()
    df_sonuc['bsr_gercek'] = np.expm1(y_ts)
    
    print("\nModeller yükleniyor ve tahminler alınıyor...")
    preds_dict = {}

    # --- XGBoost ---
    try:
        xgb_m = xgb.XGBRegressor()
        xgb_m.load_model(os.path.join(OUTPUT_DIR, 'bsr_xgb_v5.json'))
        X_ts_xgb = np.hstack([X_ts_s, X_ts_k])
        xgb_pred = np.expm1(xgb_m.predict(X_ts_xgb))
        df_sonuc['tahmin_xgb'], df_sonuc['skor_xgb'] = firsat_skoru_hesapla(df_sonuc, xgb_pred, 'xgb')
        preds_dict['XGBoost'] = xgb_pred
        print("[OK] XGBoost tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] XGBoost yüklenemedi: {e}")
        df_sonuc['tahmin_xgb'], df_sonuc['skor_xgb'] = np.nan, np.nan

    # --- CatBoost ---
    try:
        cb_m = CatBoostRegressor()
        cb_m.load_model(os.path.join(OUTPUT_DIR, 'bsr_catboost_v6.cbm'))
        X_ts_cb = pd.DataFrame(X_ts_s)
        X_ts_cb['kategori'] = X_ts_k.astype(int)
        cb_pred = np.expm1(cb_m.predict(X_ts_cb))
        df_sonuc['tahmin_cb'], df_sonuc['skor_cb'] = firsat_skoru_hesapla(df_sonuc, cb_pred, 'cb')
        preds_dict['CatBoost'] = cb_pred
        print("[OK] CatBoost tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] CatBoost yüklenemedi: {e}")
        df_sonuc['tahmin_cb'], df_sonuc['skor_cb'] = np.nan, np.nan

    # --- LightGBM ---
    try:
        lgb_m = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, 'bsr_lgbm_v6.txt'))
        X_ts_lgb = np.hstack([X_ts_s, X_ts_k])
        lgb_pred = np.expm1(lgb_m.predict(X_ts_lgb))
        df_sonuc['tahmin_lgb'], df_sonuc['skor_lgb'] = firsat_skoru_hesapla(df_sonuc, lgb_pred, 'lgb')
        preds_dict['LightGBM'] = lgb_pred
        print("[OK] LightGBM tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] LightGBM yüklenemedi: {e}")
        df_sonuc['tahmin_lgb'], df_sonuc['skor_lgb'] = np.nan, np.nan

    # --- LSTM ---
    try:
        lstm_m = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'bsr_lstm_v5.keras'))
        lstm_pred_log = lstm_m.predict([X_ts_seq, X_ts_s_l, X_ts_k_l], verbose=0).flatten()
        lstm_pred = np.expm1(lstm_pred_log)
        df_sonuc['tahmin_lstm'], df_sonuc['skor_lstm'] = firsat_skoru_hesapla(df_sonuc, lstm_pred, 'lstm')
        preds_dict['LSTM'] = lstm_pred
        print("[OK] LSTM tahmini tamamlandı.")
    except Exception as e:
        print(f"[HATA] LSTM yüklenemedi: {e}")
        df_sonuc['tahmin_lstm'], df_sonuc['skor_lstm'] = np.nan, np.nan

    # ==========================================
    # ENSEMBLE (ORTALAMA) VE TIE-BREAKER MANTIĞI
    # ==========================================
    print("\nMeta-Learner (Stacking) Eğitiliyor...")
    
    # 1. Modellerin tahminlerini log scale olarak al (Çünkü modeller log_bsr eğitiyor)
    # NaN değerleri o satırın diğer tahminlerinin ortalaması ile doldur.
    
    stack_df = pd.DataFrame()
    if 'tahmin_xgb' in df_sonuc.columns: stack_df['xgb'] = np.log1p(df_sonuc['tahmin_xgb'])
    if 'tahmin_cb' in df_sonuc.columns:  stack_df['cb'] = np.log1p(df_sonuc['tahmin_cb'])
    if 'tahmin_lgb' in df_sonuc.columns: stack_df['lgb'] = np.log1p(df_sonuc['tahmin_lgb'])
    if 'tahmin_lstm' in df_sonuc.columns: stack_df['lstm'] = np.log1p(df_sonuc['tahmin_lstm'])

    # Eksik tahminleri (örneğin bir model çöküşü) satır ortalamasıyla doldur
    stack_df = stack_df.apply(lambda row: row.fillna(row.mean()), axis=1)

    # 2. Gerçek BSR (Hedef Değişken)
    y_gercek_log = np.log1p(df_sonuc['bsr_gercek'].values)

    # 3. Ridge Regression Modelini Eğit (Meta-Learner)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_df, y_gercek_log)
    
    # Modelin verdiği ağırlıkları yazdıralım (Hangi modele ne kadar güvenmiş?)
    print("\n[Stacking Ağırlıkları]")
    for col, coef in zip(stack_df.columns, meta_model.coef_):
        print(f"  {col}: {coef:.4f}")
    
    # 4. Meta-Learner Tahmini (Stacking)
    meta_pred_log = meta_model.predict(stack_df)
    meta_pred_bsr = np.round(np.expm1(meta_pred_log)).astype(int)
    
    # 5. Sonuçlara Ekle (Artık düz ortalama yerine Meta-Learner var)
    df_sonuc['bsr_tahmin_ortalama'] = meta_pred_bsr # İsim 'ortalama' kalsın, arayüz bozulmasın
    preds_dict['Ensemble (Stacking)'] = df_sonuc['bsr_tahmin_ortalama'].values

    # Stacking Fırsat Skorunu da bu yeni daha güvenilir tahminle hesapla
    df_sonuc['bsr_tahmin_ortalama'], df_sonuc['firsat_skoru_ortalama'] = firsat_skoru_hesapla(df_sonuc, meta_pred_bsr, 'ortalama')

    # Meta-Modeli canlı sistem (canli_tahmin.py) için kaydet
    joblib.dump(meta_model, os.path.join(OUTPUT_DIR, 'meta_learner_ridge.pkl'))
    print(f"\n[OK] Meta-Learner kaydedildi: meta_learner_ridge.pkl")

    # Karşılaştırma tablosunu ekrana bas
    karsilastirma(df_sonuc, preds_dict)

    # Tie-Breaker (Eşitlik Bozucu) ile Adaletli Sıralama
    df_sonuc = df_sonuc.sort_values(
        by=['firsat_skoru_ortalama', 'bestseller_rank', 'review_count'],
        ascending=[False, True, False]
    )
    
    df_sonuc['oneri_yuzdelik'] = df_sonuc.groupby('category')['firsat_skoru_ortalama'].rank(pct=True) * 100

    print("\n" + "="*70)
    print("  ENSEMBLE - TOP 10 FIRSAT ÜRÜNÜ")
    print("="*70)
    
    top10 = df_sonuc.head(10)[['asin', 'category', 'price', 'bestseller_rank', 
                               'bsr_tahmin_ortalama', 'firsat_skoru_ortalama']]
    print(top10.to_string(index=False))

    # CSV Kaydet
    csv_yolu = os.path.join(OUTPUT_DIR, 'oneri_ensemble.csv')
    df_sonuc.to_csv(csv_yolu, index=False)
    print(f"\n[OK] Frontend arayüzü için ortak akıl verileri kaydedildi: {csv_yolu}")
    print("="*70)

if __name__ == "__main__":
    main()