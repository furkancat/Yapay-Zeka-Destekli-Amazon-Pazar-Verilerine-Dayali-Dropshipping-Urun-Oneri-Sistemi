"""
BSR Tahmin - LSTM Eğitimi (v5.0)
Ham zaman serisi (BSR, price, review) + Tabular feature + Kategori embedding
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import matplotlib.pyplot as plt

from common import (
    veri_cek, veri_ayir, on_isleme, kategori_grupla,
    ozellik_mühendisliği, model_hazirlik_lstm, zaman_serisi_sequence,
    OUTPUT_DIR, RANDOM_STATE, LSTM_WINDOW
)


def lstm_egit(X_tr_seq, X_tr_s, X_tr_k, y_tr, X_ts_seq, X_ts_s, X_ts_k, y_ts,
              kategori_sayisi, df_test):
    emb_dim = min(50, kategori_sayisi // 2)
    print(f"[LSTM] Window: {LSTM_WINDOW}, Kategori: {kategori_sayisi}, Embedding: {emb_dim}")

    inp_seq = layers.Input(shape=(LSTM_WINDOW, 3), name='zaman_serisi')
    inp_tab = layers.Input(shape=(X_tr_s.shape[1],), name='tabular')
    inp_cat = layers.Input(shape=(1,), name='kategori')

    # Orijinal, kusursuz çalışan mimariye geri dönüyoruz
    x_lstm = layers.LSTM(64, kernel_regularizer=regularizers.l2(0.01), return_sequences=False)(inp_seq)
    x_lstm = layers.Dropout(0.3)(x_lstm)

    x_tab = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inp_tab)
    x_tab = layers.Dropout(0.2)(x_tab)

    emb = layers.Embedding(kategori_sayisi, emb_dim, embeddings_regularizer=regularizers.l2(0.01))(inp_cat)
    emb = layers.Flatten()(emb)

    x = layers.concatenate([x_lstm, x_tab, emb])
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='linear')(x)

    model = Model([inp_seq, inp_tab, inp_cat], out)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])

    cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=5, min_lr=1e-6, verbose=1)
    ]

    print("\n[LSTM] Eğitim başlıyor (Batch Normalization ile)...")
    hist = model.fit([X_tr_seq, X_tr_s, X_tr_k], y_tr,
                     validation_split=0.2,
                     epochs=100, batch_size=32, callbacks=cb, verbose=1)

    y_p_ts = model.predict([X_ts_seq, X_ts_s, X_ts_k], verbose=0).flatten()
    bsr_pred = np.round(np.expm1(y_p_ts)).astype(int)
    bsr_g = np.expm1(y_ts)

    print("\n=== LSTM (Ham Zaman Serisi) SONUÇLARI ===")
    print(f"Test Log-MAE: {mean_absolute_error(y_ts, y_p_ts):.4f}")
    print(f"Test R²     : {r2_score(y_ts, y_p_ts):.4f}")
    print(f"Test BSR-MAE: {mean_absolute_error(bsr_g, bsr_pred):.2f}")

    df_test = df_test.copy()
    df_test['bsr_tahmin_lstm'] = bsr_pred
    if 'bsr_gercek' not in df_test.columns:
        df_test['bsr_gercek'] = bsr_g

    print("\n--- Kategori Bazlı LSTM BSR-MAE ---")
    for kat in sorted(df_test['category'].unique()):
        mask = df_test['category'] == kat
        if mask.sum() >= 5:
            mae = mean_absolute_error(df_test.loc[mask, 'bsr_gercek'],
                                      df_test.loc[mask, 'bsr_tahmin_lstm'])
            print(f"{kat:35s}: MAE={mae:8.2f} (n={mask.sum()})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist.history['loss'], label='Train')
    ax.plot(hist.history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('LSTM Eğitim Geçmişi')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_training_v5.png'), dpi=150)
    plt.close()

    model.save(os.path.join(OUTPUT_DIR, 'bsr_lstm_v5.keras'))
    print("\n[OK] LSTM modeli kaydedildi.")
    return model, bsr_pred


def main():
    print("="*70)
    print("BSR TAHMİN v5.0 - LSTM Eğitimi (Ham Zaman Serisi)")
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

    # LSTM sequence
    seq_dict = zaman_serisi_sequence(df_zaman, df_ana, train_asin, test_asin, cutoff, window=LSTM_WINDOW)

    (X_tr_s, X_tr_k, y_tr, X_ts_s, X_ts_k, y_ts,
     scaler, ozellikler, mi_df, X_tr_seq, X_ts_seq, seq_scaler) = model_hazirlik_lstm(df_train, df_test, seq_dict)

    print("\n" + "="*70)
    print("DERİN ÖĞRENME: LSTM (Ham Zaman Serisi)")
    print("="*70)
    lstm_m, lstm_pred = lstm_egit(
        X_tr_seq, X_tr_s, X_tr_k, y_tr,
        X_ts_seq, X_ts_s, X_ts_k, y_ts,
        len(le_cat.classes_), df_test
    )

    np.save(os.path.join(OUTPUT_DIR, 'seq_scaler_mean_v5.npy'), seq_scaler.mean_)
    np.save(os.path.join(OUTPUT_DIR, 'seq_scaler_scale_v5.npy'), seq_scaler.scale_)
    np.save(os.path.join(OUTPUT_DIR, 'encoder_v5_lstm.npy'), le_cat.classes_)
    print("\n[OK] Tüm dosyalar kaydedildi.")
    print("="*70)


if __name__ == "__main__":
    main()