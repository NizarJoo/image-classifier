import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

def extract_lbp_features(image, P=8, R=1):
    gray = rgb2gray(image)
    gray = img_as_ubyte(gray)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_haralick_features(image):
    gray = rgb2gray(image)
    gray = img_as_ubyte(gray)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0]
    ]
    return np.array(features)

def extract_texture_features(image_path, label_path):
    images = np.load(image_path)
    labels = np.load(label_path)

    features = []

    for img in images:
        lbp_feat = extract_lbp_features(img)
        haralick_feat = extract_haralick_features(img)
        combined = np.concatenate([lbp_feat, haralick_feat])
        features.append(combined)

    return np.array(features), labels
