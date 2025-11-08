import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARSE ARGUMENTS (untuk MLflow Project)
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=10)
args = parser.parse_args()

print("=" * 60)
print("MEMULAI TRAINING MODEL")
print("=" * 60)
print(f"Parameters:")
print(f"  - test_size: {args.test_size}")
print(f"  - random_state: {args.random_state}")
print(f"  - n_estimators: {args.n_estimators}")
print(f"  - max_depth: {args.max_depth}")

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nMemuat data preprocessing...")

# Auto-detect file path
import os
if os.path.exists("diabetes_preprocessing.csv"):
    data_path = "diabetes_preprocessing.csv"
elif os.path.exists("../diabetes_preprocessing.csv"):
    data_path = "../diabetes_preprocessing.csv"
else:
    raise FileNotFoundError("diabetes_preprocessing.csv tidak ditemukan!")

df = pd.read_csv(data_path)
print(f"Data berhasil dimuat! Shape: {df.shape}")

# Split features dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=args.test_size, 
    random_state=args.random_state, 
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================
# 2. TRAINING MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# PENTING: Jangan buat nested run jika sudah di dalam MLflow Project run
# Cek apakah sudah ada active run
active_run = mlflow.active_run()

if active_run is not None:
    # Sudah ada run dari MLflow Project, langsung log saja
    print(f"Using existing run: {active_run.info.run_id}")
    
    # Enable autologging
    mlflow.sklearn.autolog()
    
    # Log additional parameters
    mlflow.log_param("dataset_name", "diabetes")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("n_features", X.shape[1])
    
    # Create and train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model berhasil dilatih!")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL")
    print("=" * 60)
    print(f"Training Accuracy  : {train_acc:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
    print(f"Test Precision     : {test_precision:.4f}")
    print(f"Test Recall        : {test_recall:.4f}")
    print(f"Test F1-Score      : {test_f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    print("\nModel dan metrics berhasil disimpan ke MLflow!")
    
else:
    # Tidak ada run, buat run baru (untuk testing standalone)
    print("Creating new run...")
    
    # Set tracking URI jika belum diset
    if not mlflow.get_tracking_uri() or mlflow.get_tracking_uri() == "file:///mlruns":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Set experiment
    mlflow.set_experiment("Diabetes_Prediction_Experiment")
    
    # Enable autologging
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Basic"):
        # Log additional parameters
        mlflow.log_param("dataset_name", "diabetes")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X.shape[1])
        
        # Create and train model
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("Model berhasil dilatih!")
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        print("\n" + "=" * 60)
        print("HASIL EVALUASI MODEL")
        print("=" * 60)
        print(f"Training Accuracy  : {train_acc:.4f}")
        print(f"Test Accuracy      : {test_acc:.4f}")
        print(f"Test Precision     : {test_precision:.4f}")
        print(f"Test Recall        : {test_recall:.4f}")
        print(f"Test F1-Score      : {test_f1:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        
        print("\nModel dan metrics berhasil disimpan ke MLflow!")

print("\n" + "=" * 60)
print("TRAINING SELESAI!")
print("=" * 60)