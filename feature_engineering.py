import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure directory exists for model artifacts
os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
DROP_COLS = ["label", "attack_category", "binary_label", "binary_class"]


def encode_categorical_features(train_df, test_df):
    """Encodes string categorical variables using LabelEncoder."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Combine unique values across train and test to handle unseen labels
        combined_classes = np.unique(
            np.concatenate([train_df[col].astype(str), test_df[col].astype(str)])
        )
        le.fit(combined_classes)

        train_encoded[col] = le.transform(train_df[col].astype(str))
        test_encoded[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le

    return train_encoded, test_encoded, encoders


def select_top_features(X_train, y_train, feature_names, top_n=25):
    """Identifies top N most influential features using Random Forest importance."""
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    selected_indices = sorted_indices[:top_n]
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"\nSelected Top {top_n} Features:")
    for idx, f_idx in enumerate(selected_indices[:10]):
        print(f"  {idx+1}. {feature_names[f_idx]} (Importance: {importances[f_idx]:.4f})")

    return selected_features, selected_indices


def main():
    print("--- Starting Days 6–8: Feature Engineering ---")

    # 1. Load cleaned datasets
    if not os.path.exists("datasets/cleaned_train.csv"):
        raise FileNotFoundError("Run eda_and_cleaning.py first to generate cleaned CSVs.")

    df_train = pd.read_csv("datasets/cleaned_train.csv")
    df_test = pd.read_csv("datasets/cleaned_test.csv")

    # 2. Encode categorical columns
    print("\n1. Encoding Categorical Features...")
    df_train_enc, df_test_enc, encoders = encode_categorical_features(df_train, df_test)

    # 3. Separate features (X) and target (y)
    feature_cols = [c for c in df_train_enc.columns if c not in DROP_COLS]
    
    X_train_raw = df_train_enc[feature_cols].values
    y_train_binary = df_train_enc["binary_label"].values

    X_test_raw = df_test_enc[feature_cols].values
    y_test_binary = df_test_enc["binary_label"].values

    # 4. Feature Selection
    print("\n2. Performing Feature Selection (Tree-Based)...")
    selected_features, selected_indices = select_top_features(
        X_train_raw, y_train_binary, feature_cols, top_n=25
    )

    X_train_sub = X_train_raw[:, selected_indices]
    X_test_sub = X_test_raw[:, selected_indices]

    # 5. Feature Scaling
    print("\n3. Scaling Features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled = scaler.transform(X_test_sub)

    # 6. Save Arrays and Transformer Artifacts
    print("\n4. Saving Feature Arrays and Pipelines to models/...")
    
    np.savez_compressed(
        "datasets/processed_data.npz",
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train_binary,
        y_test=y_test_binary,
        y_train_multi=df_train_enc["attack_category"].values,
        y_test_multi=df_test_enc["attack_category"].values,
    )

    artifacts = {
        "scaler": scaler,
        "encoders": encoders,
        "selected_features": selected_features,
        "all_feature_cols": feature_cols,
    }
    joblib.dump(artifacts, "models/feature_pipeline.pkl")

    print("\n--- Feature Engineering Complete! ---")
    print("Processed matrix shape (Train):", X_train_scaled.shape)
    print("Processed matrix shape (Test):", X_test_scaled.shape)


if __name__ == "__main__":
    main()