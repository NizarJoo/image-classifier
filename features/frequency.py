import numpy as np
import cv2

def extract_frequency_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Statistik dasar dari spektrum frekuensi
    mean_freq = np.mean(magnitude_spectrum)
    std_freq = np.std(magnitude_spectrum)
    energy_freq = np.sum(magnitude_spectrum ** 2) / magnitude_spectrum.size

    return np.array([mean_freq, std_freq, energy_freq])

def extract_frequency_features_from_npy(image_path, label_path):
    images = np.load(image_path)
    labels = np.load(label_path)

    features = []
    for img in images:
        freq_feat = extract_frequency_features(img)
        features.append(freq_feat)

    return np.array(features), labels
