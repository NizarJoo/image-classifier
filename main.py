from preprocessing.preprocess_and_save import save_preprocessed_data
from features.color import extract_color_features
from features.texture import extract_texture_features
from features.frequency import extract_frequency_features_from_npy
import numpy as np
import pickle

if __name__ == "__main__":
    # Optional: uncomment kalau mau preprocess ulang
    # save_preprocessed_data()

    # ----------------------------
    # 1. Ekstraksi Fitur Warna
    # ----------------------------
    real_color, real_label = extract_color_features(
        "data/processed/real_images.npy", "data/processed/real_labels.npy"
    )
    ai_color, ai_label = extract_color_features(
        "data/processed/ai_images.npy", "data/processed/ai_labels.npy"
    )

    X_color = np.vstack((real_color, ai_color))
    y_color = np.hstack((real_label, ai_label))

    np.save("data/processed/color_features.npy", X_color)
    np.save("data/processed/color_labels.npy", y_color)

    print("✔️ Fitur warna berhasil disimpan.")

    # ----------------------------
    # 2. Ekstraksi Fitur Tekstur
    # ----------------------------
    real_tex, real_label = extract_texture_features(
        "data/processed/real_images.npy", "data/processed/real_labels.npy"
    )
    ai_tex, ai_label = extract_texture_features(
        "data/processed/ai_images.npy", "data/processed/ai_labels.npy"
    )

    X_texture = np.vstack((real_tex, ai_tex))
    y_texture = np.hstack((real_label, ai_label))

    np.save("data/processed/texture_features.npy", X_texture)
    np.save("data/processed/texture_labels.npy", y_texture)

    print("✔️ Fitur tekstur berhasil disimpan.")

    # ----------------------------
    # 3. Ekstraksi Fitur Frekuensi
    # ----------------------------
    real_freq, real_label = extract_frequency_features_from_npy(
        "data/processed/real_images.npy", "data/processed/real_labels.npy"
    )
    ai_freq, ai_label = extract_frequency_features_from_npy(
        "data/processed/ai_images.npy", "data/processed/ai_labels.npy"
    )

    X_frequency = np.vstack((real_freq, ai_freq))
    y_frequency = np.hstack((real_label, ai_label))

    np.save("data/processed/frequency_features.npy", X_frequency)
    np.save("data/processed/frequency_labels.npy", y_frequency)

    print("✔️ Fitur frekuensi berhasil disimpan.")

    # ----------------------------
    # 4. Gabungkan semua fitur
    # ----------------------------
    X_combined = np.hstack((X_color, X_texture, X_frequency))
    y_combined = y_color  # label harus sama di semua

    with open("data/processed/features_labels.pkl", "wb") as f:
        pickle.dump((X_combined, y_combined), f)

    print("✔️ Fitur gabungan (warna + tekstur + frekuensi) berhasil disimpan ke features_labels.pkl.")
