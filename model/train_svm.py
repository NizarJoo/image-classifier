import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

if __name__ == "__main__":
    with open("data/processed/features_labels.pkl", "rb") as f:
        X, y = pickle.load(f)

    print("Shape fitur:", X.shape)
    print("Shape label:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(
        kernel="rbf",  
        C=1.0,          
        gamma="scale",  
        random_state=42
    )

    print("Training model SVM...")
    model.fit(X_train, y_train)

    # 5. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== HASIL EVALUASI ===")
    print("Akurasi:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
