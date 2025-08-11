import os
import cv2
import numpy as np

def preprocess_images(input_dir, output_dir, label, image_size=(128, 128)):
    images = []
    labels = []

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(input_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)

            # Simpan ke list untuk disatukan
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def save_preprocessed_data():
    real_input = "data/raw/RealArt/RealArt"
    ai_input = "data/raw/AiArtData/AiArtData"
    real_output = "data/processed/real"
    ai_output = "data/processed/ai"

    # Proses real
    real_images, real_labels = preprocess_images(real_input, real_output, label=0)
    np.save("data/processed/real_images.npy", real_images)
    np.save("data/processed/real_labels.npy", real_labels)

    # Proses AI
    ai_images, ai_labels = preprocess_images(ai_input, ai_output, label=1)
    np.save("data/processed/ai_images.npy", ai_images)
    np.save("data/processed/ai_labels.npy", ai_labels)

    print(f"Saved: {real_images.shape[0]} real images, {ai_images.shape[0]} AI images")
