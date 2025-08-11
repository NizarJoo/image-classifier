import pickle
import numpy as np
from features.combined import extract_combined_features

# Ekstrak dari gambar asli
real_comb, real_label = extract_combined_features(
    "data/processed/real_images.npy", 
    "data/processed/real_labels.npy"
)

# Ekstrak dari gambar AI
ai_comb, ai_label = extract_combined_features(
    "data/processed/ai_images.npy", 
    "data/processed/ai_labels.npy"
)

# Gabungkan semua fitur dan label
X = np.concatenate([real_comb, ai_comb])
y = np.concatenate([real_label, ai_label])

# Simpan ke pickle
with open("data/processed/final_features_labels.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("✔️ Fitur dari gambar asli dan AI berhasil digabungkan dan disimpan ke final_features_labels.pkl")
