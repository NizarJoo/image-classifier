import numpy as np
from features.color import extract_color_histogram
from features.texture import extract_lbp_features, extract_haralick_features
from features.frequency import extract_frequency_features

def extract_combined_features(image_path, label_path):
    images = np.load(image_path)
    labels = np.load(label_path)

    features = []

    for img in images:
        # Ekstrak fitur
        color_feat = extract_color_histogram(img)
        lbp_feat = extract_lbp_features(img)
        haralick_feat = extract_haralick_features(img)
        freq_feat = extract_frequency_features(img)

        # Gabungkan semuanya
        combined = np.concatenate([color_feat, lbp_feat, haralick_feat, freq_feat])
        features.append(combined)

    return np.array(features), labels
