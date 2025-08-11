import numpy as np
import os

def extract_color_histogram(image, bins=(8, 8, 8)):
    # Histogram RGB 3D
    hist = np.histogramdd(
        image.reshape(-1, 3),
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)]
    )[0]

    # Normalisasi dan flatten
    hist = hist / np.sum(hist)
    return hist.flatten()

def extract_color_features(image_path, label_path):
    # Load gambar & label
    images = np.load(image_path)
    labels = np.load(label_path)

    features = []

    for img in images:
        hist = extract_color_histogram(img)
        features.append(hist)

    return np.array(features), labels
