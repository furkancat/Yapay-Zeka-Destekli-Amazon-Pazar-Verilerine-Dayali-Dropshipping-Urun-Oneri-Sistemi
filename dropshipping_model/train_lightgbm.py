"""
BSR Tahmin - LightGBM Eğitimi (v6.0 - 7 Günlük Gelecek Tahmini)
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from common import (veri_cek, veri_ayir, on_isleme, kategori_grupla,
                    ozellik_mühendisliği, model_hazirlik, OUTPUT_DIR, RANDOM_STATE)

def lightgbm_egit(X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, ozellikler, df_test):
    X_tr = np.hstack([X_tr_s, X_tr_k])
    X_ts = np.hstack([X_ts_s, X_ts_k])

    print("\n[LightGBM] 5-Fold CV Başlıyor...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = []

    for fold, (tr_i, val_i) in enumerate(kf.split(X_tr, X_tr_k[:, 0])):
        m = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        m.fit(X_tr[tr_i], y_tr[tr_i], eval_set=[(X_tr[val_i], y_tr[val_i])], eval_metric='mae')
        pred = m.predict(X_tr[val_i])
        cv_mae.append(mean_absolute_error(y_tr[val_i], pred))
        print(f"  Fold {fold+1} Log-MAE: {cv_mae[-1]:.4f}")

    print(f"  CV Ortalaması: {np.mean(cv_mae):.4f}")

    # Final Modeli Eğitimi
    final = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.02, random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(X_tr, y_tr)

    y_p_ts = final.predict(X_ts)
    bsr_pred = np.round(np.expm1(y_p_ts)).astype(int)
    bsr_g = np.expm1(y_ts)

    print("\n=== LightGBM SONUÇLARI (7 Gün Sonrası İçin) ===")
    print(f"Test Log-MAE: {mean_absolute_error(y_ts, y_p_ts):.4f}")
    print(f"Test R²     : {r2_score(y_ts, y_p_ts):.4f}")
    print(f"Test BSR-MAE: {mean_absolute_error(bsr_g, bsr_pred):.2f}")

    # Feature Importance (Özellik Önemleri)
    tum_isim = ozellikler + ['kategori_kod']
    onem = pd.DataFrame({'ozellik': tum_isim, 'onem': final.feature_importances_})
    onem = onem.sort_values('onem', ascending=False)
    print("\n--- LightGBM Özellik Önemleri (Top 15) ---")
    print(onem.head(15).to_string(index=False))

    # --- GRAFİK ÇİZİMİ VE KAYDEDİLMESİ ---
    plt.figure(figsize=(10, 8))
    sns.barplot(x='onem', y='ozellik', data=onem.head(20), palette='viridis')
    plt.title('LightGBM Özellik Önemleri (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lgbm_importance_v6.png'), dpi=150)
    plt.close()

    # Kategori bazlı Hata Hesaplaması
    df_test = df_test.copy()
    df_test['bsr_tahmin_lgbm'] = bsr_pred
    df_test['bsr_gercek'] = bsr_g

    print("\n--- Kategori Bazlı LightGBM BSR-MAE ---")
    for kat in sorted(df_test['category'].unique()):
        mask = df_test['category'] == kat
        if mask.sum() >= 5:
            mae = mean_absolute_error(df_test.loc[mask, 'bsr_gercek'],
                                      df_test.loc[mask, 'bsr_tahmin_lgbm'])
            print(f"{kat:35s}: MAE={mae:8.2f} (n={mask.sum()})")

    # Modeli Kaydet
    final.booster_.save_model(os.path.join(OUTPUT_DIR, 'bsr_lgbm_v6.txt'))
    print("\n[OK] LightGBM modeli ve grafiği kaydedildi.")

def main():
    print("="*70 + "\nBSR TAHMİN v6.0 - LightGBM Eğitimi (7 Gün Hedefli)\n" + "="*70)
    
    # 1. Veri Çekme ve Ayırma
    df_ana, df_zaman = veri_cek()
    df_train_raw, df_test_raw, le_cat, cutoff, train_asin, test_asin = veri_ayir(df_ana)
    
    # 2. Ön İşleme
    df_train = on_isleme(df_train_raw, df_zaman, train_asin, set(), cutoff)
    df_test = on_isleme(df_test_raw, df_zaman, set(), test_asin, cutoff)

    # 3. Kategori Gruplama
    df_all = pd.concat([df_train, df_test])
    df_all, le_cat = kategori_grupla(df_all)
    df_train = df_all[df_all['asin'].isin(train_asin)].copy()
    df_test = df_all[df_all['asin'].isin(test_asin)].copy()
    
    # 4. Özellik Mühendisliği ve Model Hazırlık
    df_train, df_test = ozellik_mühendisliği(df_train, df_test)
    (X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, scaler, ozellikler, mi_df) = model_hazirlik(df_train, df_test)

    # 5. Eğitimi Başlat
    lightgbm_egit(X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, ozellikler, df_test)

if __name__ == "__main__": 
    main()