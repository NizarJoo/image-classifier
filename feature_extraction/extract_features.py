# feature_extraction/extract_features.py

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from PIL import Image
import os
# import pandas as pd

def extract_color_features(image):
    image = np.array(image)
    mean = image.mean(axis=(0, 1))
    std = image.std(axis=(0, 1))
    return mean.tolist() + std.tolist()

def extract_texture_features(image):
    gray = rgb2gray(np.array(image))
    gray = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, homogeneity, energy, correlation]

def extract_frequency_features(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Statistik dasar dari spektrum frekuensi
    mean_freq = np.mean(magnitude_spectrum)
    std_freq = np.std(magnitude_spectrum)
    energy_freq = np.sum(magnitude_spectrum ** 2) / magnitude_spectrum.size

    return [mean_freq, std_freq, energy_freq]

def extract_features(image):
    color = extract_color_features(image)
    texture = extract_texture_features(image)
    frequency = extract_frequency_features(image)
    return color + texture + frequency

# ✅ Tambahkan ini:
# feature_extraction/extract_features.py

def extract_combined_features(image_array_path, label_array_path):
    X = np.load(image_array_path, allow_pickle=True)
    y = np.load(label_array_path, allow_pickle=True)

    all_features = []
    all_labels = []

    for image, label in zip(X, y):
        try:
            # Konversi ndarray ke Image untuk kompatibilitas fungsi fitur
            image_pil = Image.fromarray(image.astype('uint8')).convert('RGB')
            features = extract_features(image_pil)
            all_features.append(features)
            all_labels.append(label)
        except Exception as e:
            print(f"Error processing image: {e}")

    return np.array(all_features), np.array(all_labels)
    all_features = []
    all_labels = []

    for file_name in os.listdir(image_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(image_dir, file_name)
            try:
                npy_image = np.load(file_path)
                image = Image.fromarray(npy_image).convert('RGB')
                features = extract_features(image)
                all_features.append(features)

                # Label dari nama file, contoh: 'fake_001.npy' -> label: 'fake'
                label = file_name.split('_')[0].lower()
                all_labels.append(label)
            except Exception as e:
                print(f"Gagal proses {file_path}: {e}")

    return np.array(all_features), np.array(all_labels)


