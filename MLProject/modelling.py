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

parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=10)
args = parser.parse_args()

print("Loading data...")

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

mlflow.sklearn.autolog()

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

y_pred_test = model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

print("Training complete")