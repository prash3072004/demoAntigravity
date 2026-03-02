"""
Step 2: Train the Gesture Classifier
======================================
Run this script after collect_data.py has finished.
It loads the saved landmarks, trains a Random Forest classifier,
prints a full accuracy report, and saves the model.

Usage:
    python train_model.py
"""

import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def train():
    # ── Load data ────────────────────────────────────────────────────────────
    X_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("ERROR: Training data not found.")
        print("       Please run  python collect_data.py  first.")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Loaded data  →  X: {X.shape},  y: {y.shape}")
    print(f"Classes found: {sorted(set(y))}\n")

    # ── Encode labels ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train samples: {len(X_train)}   Test samples: {len(X_test)}\n")

    # ── Train model ──────────────────────────────────────────────────────────
    print("Training Random Forest classifier …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,       # use all CPU cores
    )
    clf.fit(X_train, y_train)
    print("Training complete!\n")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in le.classes_]
    ))

    # ── Save model & encoder ─────────────────────────────────────────────────
    with open(MODEL_PATH,   "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"✅ Model saved  →  {MODEL_PATH}")
    print(f"✅ Encoder saved→  {ENCODER_PATH}")
    print(f"\nNext step: run  python recognize.py")


if __name__ == "__main__":
    train()
