import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import warnings
import os

warnings.filterwarnings('ignore')

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=10)
args = parser.parse_args()

print("Loading data...")

# === Load dataset ===
if os.path.exists("diabetes_preprocessing.csv"):
    data_path = "diabetes_preprocessing.csv"
elif os.path.exists("../diabetes_preprocessing.csv"):
    data_path = "../diabetes_preprocessing.csv"
else:
    raise FileNotFoundError("diabetes_preprocessing.csv not found")

df = pd.read_csv(data_path)
print(f"Data loaded. Shape: {df.shape}")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=y
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# === MLflow tracking ===
with mlflow.start_run():
    mlflow.sklearn.autolog()

    # Log metadata (gunakan nama unik agar tidak bentrok dengan autolog)
    mlflow.log_param("dataset_name", "diabetes")
    mlflow.log_param("train_records", len(X_train))
    mlflow.log_param("test_records", len(X_test))
    mlflow.log_param("n_features", X.shape[1])

    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # === Evaluate ===
    y_pred_test = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    # Log metrics manually (autolog juga bisa, tapi ini eksplisit)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)

    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

# === Simpan model secara eksplisit ===
os.makedirs("models", exist_ok=True)
model_path = "models/model.pkl"

import joblib
joblib.dump(model, model_path)

# Log model ke MLflow
mlflow.sklearn.log_model(model, artifact_path="model")

print(f"Model saved locally at {model_path}")

print("Training complete")
