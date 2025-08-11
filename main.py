from preprocessing.preprocess_and_save import save_preprocessed_data
from features.color import extract_color_features
import numpy as np
from features.texture import extract_texture_features


if __name__ == "__main__":
    # Optional: uncomment ini kalau pengen re-preprocess data mentah dari awal
    # save_preprocessed_data()

    # Load hasil preprocess dan ekstrak fitur warna
    real_feat, real_label = extract_color_features(
        "data/processed/real_images.npy", "data/processed/real_labels.npy"
    )
    ai_feat, ai_label = extract_color_features(
        "data/processed/ai_images.npy", "data/processed/ai_labels.npy"
    )

    X_color = np.vstack((real_feat, ai_feat))
    y_color = np.hstack((real_label, ai_label))

    print("Fitur shape:", X_color.shape)
    print("Label shape:", y_color.shape)

    np.save("data/processed/color_features.npy", X_color)
    np.save("data/processed/color_labels.npy", y_color)

    print("✔️ Fitur warna berhasil disimpan.")

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

from feature_extraction.extract_features import extract_combined_features
import pickle

# Ekstraksi fitur gabungan warna dan tekstur
real_comb, real_label = extract_combined_features(
    "data/processed/real_images.npy",
    "data/processed/real_labels.npy"
)

ai_comb, ai_label = extract_combined_features(
    "data/processed/ai_images.npy", "data/processed/ai_labels.npy"
)

X_combined = np.vstack((real_comb, ai_comb))
y_combined = np.hstack((real_label, ai_label))

# Simpan ke file pickle
with open("data/processed/features_labels.pkl", "wb") as f:
    pickle.dump((X_combined, y_combined), f)

print("✔️ Fitur gabungan warna + tekstur berhasil disimpan ke features_labels.pkl.")
