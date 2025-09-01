import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Load data ===
print("Loading features and labels...")
with open("data/processed/features_labels.pkl", "rb") as f:
    data = pickle.load(f)

X, y = data  

print("Shape fitur:", X.shape)
print("Shape label:", y.shape)


# === Normalisasi ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Definisikan parameter grid untuk tuning ===
param_grid = {
    "C": [0.1, 1, 10, 100],       # Regularisasi
    "gamma": [1, 0.1, 0.01, 0.001],  # Skala kernel RBF
    "kernel": ["linear", "rbf"]   # Kernel dasar
}

# === Model + GridSearch ===
print("\nTraining model SVM dengan GridSearchCV...")
grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("\n=== HASIL TUNING ===")
print(f"Best Params: {grid.best_params_}")
print(f"Best Cross-Validation Score: {grid.best_score_:.4f}")

# === Evaluasi di test set ===
y_pred = grid.predict(X_test)

print("\n=== HASIL EVALUASI TEST SET ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
