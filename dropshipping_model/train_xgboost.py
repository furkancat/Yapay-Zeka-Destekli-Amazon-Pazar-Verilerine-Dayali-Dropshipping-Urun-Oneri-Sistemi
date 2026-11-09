"""
BSR Tahmin - XGBoost Eğitimi (v5.0)
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    veri_cek, veri_ayir, on_isleme, kategori_grupla,
    ozellik_mühendisliği, model_hazirlik, OUTPUT_DIR, RANDOM_STATE
)


def xgboost_egit(X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, ozellikler, df_test):
    X_tr = np.hstack([X_tr_s, X_tr_k])
    X_ts = np.hstack([X_ts_s, X_ts_k])

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

    final = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    final.fit(X_tr, y_tr, verbose=False)

    y_p_tr = final.predict(X_tr)
    y_p_ts = final.predict(X_ts)
    bsr_pred = np.round(np.expm1(y_p_ts)).astype(int)
    bsr_g = np.expm1(y_ts)

    print("\n=== XGBoost SONUÇLARI ===")
    print(f"Train Log-MAE: {mean_absolute_error(y_tr, y_p_tr):.4f}")
    print(f"Test  Log-MAE: {mean_absolute_error(y_ts, y_p_ts):.4f}")
    print(f"Test  R²     : {r2_score(y_ts, y_p_ts):.4f}")
    print(f"Test  BSR-MAE: {mean_absolute_error(bsr_g, bsr_pred):.2f}")

    tum_isim = ozellikler + ['kategori_kod']
    onem = pd.DataFrame({'ozellik': tum_isim, 'onem': final.feature_importances_})
    onem = onem.sort_values('onem', ascending=False)
    print("\n--- XGBoost Özellik Önemleri (Top 15) ---")
    print(onem.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(final, ax=ax, max_num_features=20, importance_type='gain')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_importance_v5.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_test[ozellikler].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix (v5.0)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_v5.png'), dpi=150)
    plt.close()

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

    final.save_model(os.path.join(OUTPUT_DIR, 'bsr_xgb_v5.json'))
    np.save(os.path.join(OUTPUT_DIR, 'xgb_features_v5.npy'), ozellikler)
    print("\n[OK] XGBoost modeli kaydedildi.")
    return final, bsr_pred, onem


def main():
    print("="*70)
    print("BSR TAHMİN v5.0 - XGBoost Eğitimi")
    print("="*70)

    df_ana, df_zaman = veri_cek()
    df_train_raw, df_test_raw, le_cat, cutoff, train_asin, test_asin = veri_ayir(df_ana)

    df_train = on_isleme(df_train_raw, df_zaman, train_asin, set(), cutoff)
    df_test = on_isleme(df_test_raw, df_zaman, set(), test_asin, cutoff)

    df_all = pd.concat([df_train, df_test])
    df_all, le_cat = kategori_grupla(df_all)
    df_train = df_all[df_all['asin'].isin(train_asin)].copy()
    df_test = df_all[df_all['asin'].isin(test_asin)].copy()

    df_train, df_test = ozellik_mühendisliği(df_train, df_test)

    (X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts,
     scaler, ozellikler, mi_df) = model_hazirlik(df_train, df_test)

    print("\n" + "="*70)
    print("MAKİNE ÖĞRENMESİ: XGBOOST")
    print("="*70)
    xgb_m, xgb_pred, xgb_onem = xgboost_egit(
        X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts, ozellikler, df_test
    )

    np.save(os.path.join(OUTPUT_DIR, 'encoder_v5.npy'), le_cat.classes_)
    np.save(os.path.join(OUTPUT_DIR, 'scaler_mean_v5.npy'), scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, 'scaler_scale_v5.npy'), scaler.scale_)
    mi_df.to_csv(os.path.join(OUTPUT_DIR, 'mi_scores_v5.csv'), index=False)
    print("\n[OK] Tüm dosyalar kaydedildi.")
    print("="*70)


if __name__ == "__main__":
    main()