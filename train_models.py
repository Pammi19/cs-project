import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name):
    """Trains a model and computes comprehensive classification metrics."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate ROC-AUC if probabilities are supported
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = 0.0

    cm = confusion_matrix(y_test, y_pred)
    
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc,
        "Confusion Matrix": cm,
        "fitted_model": model
    }

def main():
    print("--- Starting Days 9–11: Machine Learning Model Development ---")

    # 1. Load preprocessed arrays created in Days 6–8
    data_path = "datasets/processed_data.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Processed dataset not found. Please run feature_engineering.py first.")

    data = np.load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    print(f"Dataset Loaded Successfully: Train Shape {X_train.shape}, Test Shape {X_test.shape}")

    # 2. Define Algorithms to Compare
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }

    # 3. Train and Evaluate Each Model
    results = []
    trained_models = {}

    for name, clf in classifiers.items():
        metrics = evaluate_classifier(clf, X_train, X_test, y_train, y_test, name)
        results.append({
            "Model": metrics["Model"],
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"]
        })
        trained_models[name] = metrics

    # 4. Model Comparison Table
    df_results = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n=======================================================")
    print("             MODEL COMPARISON SUMMARY                 ")
    print("=======================================================")
    print(df_results.to_string(index=False))

    # 5. Save the Best Performing Model
    best_model_name = df_results.iloc[0]["Model"]
    best_model_obj = trained_models[best_model_name]["fitted_model"]
    
    print(f"\nSaving Best Model ('{best_model_name}') to models/ids_model.pkl...")

    # Load existing feature pipeline artifacts to merge
    pipeline_artifacts = joblib.load("models/feature_pipeline.pkl")
    
    model_bundle = {
        "model": best_model_obj,
        "model_name": best_model_name,
        "scaler": pipeline_artifacts["scaler"],
        "encoders": pipeline_artifacts["encoders"],
        "selected_features": pipeline_artifacts["selected_features"],
        "metrics": trained_models[best_model_name]
    }

    joblib.dump(model_bundle, "models/ids_model.pkl")
    print("Best model bundle saved successfully as 'models/ids_model.pkl'.")
    print("\n--- Model Training & Selection Complete! ---")

if __name__ == "__main__":
    main()