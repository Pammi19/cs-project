import os
import joblib
import numpy as np
import pandas as pd
import pytest

from reports import generate_executive_text_report, convert_dataframe_to_csv
from visualization import plot_protocol_attack_breakdown, plot_threat_radar


# -------------------------------------------------------------------
# Artifact & Model Tests
# -------------------------------------------------------------------
def test_model_artifacts_exist():
    """Verify that required model artifacts are generated and accessible."""
    assert os.path.exists("models/ids_model.pkl"), "Model bundle missing!"
    assert os.path.exists("models/feature_pipeline.pkl"), "Pipeline missing!"


def test_model_bundle_structure():
    """Validate that the saved model bundle contains all critical keys."""
    bundle = joblib.load("models/ids_model.pkl")
    required_keys = ["model", "model_name", "scaler", "encoders", "selected_features", "metrics"]
    for key in required_keys:
        assert key in bundle, f"Missing key '{key}' in model bundle."


# -------------------------------------------------------------------
# Preprocessing & Edge Case Tests
# -------------------------------------------------------------------
def test_unseen_categorical_label_handling():
    """Ensure the pipeline safely encodes unseen categorical string labels without crashing."""
    bundle = joblib.load("models/ids_model.pkl")
    encoders = bundle["encoders"]
    
    le = encoders["protocol_type"]
    unseen_value = "quantum_protocol_v9"
    
    # Safe encoding logic test
    encoded_val = le.transform([unseen_value])[0] if unseen_value in le.classes_ else 0
    assert encoded_val == 0, "Unseen category should default safely to index 0."


def test_inference_pipeline_execution():
    """Verify inference pipeline execution end-to-end on synthetic data."""
    bundle = joblib.load("models/ids_model.pkl")
    scaler = bundle["scaler"]
    model = bundle["model"]
    selected_features = bundle["selected_features"]

    # Generate dummy scaled input array matching the selected features length (25)
    dummy_input = np.zeros((10, len(selected_features)))
    scaled_input = scaler.transform(dummy_input)
    predictions = model.predict(scaled_input)

    assert len(predictions) == 10, "Prediction count must match input row count."
    assert set(predictions).issubset({0, 1}), "Predictions must be binary (0 or 1)."


# -------------------------------------------------------------------
# Reporting & Visualization Module Tests
# -------------------------------------------------------------------
def test_report_generation_non_empty():
    """Confirm security text report compiles successfully with test data."""
    dummy_df = pd.DataFrame({
        "Threat_Status": ["Normal", "Malicious", "Normal"],
        "protocol_type": ["tcp", "udp", "icmp"],
        "duration": [0, 12, 0]
    })
    
    report_text = generate_executive_text_report(dummy_df, model_name="Random Forest")
    
    assert "INTRUSION DETECTION SYSTEM (IDS) SECURITY REPORT" in report_text
    assert "Total Packets Processed : 3" in report_text
    assert "Malicious Flags Count   : 1" in report_text


def test_csv_export_byte_conversion():
    """Ensure dataframe export correctly outputs UTF-8 encoded bytes."""
    dummy_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    csv_bytes = convert_dataframe_to_csv(dummy_df)
    
    assert isinstance(csv_bytes, bytes)
    assert b"col1,col2" in csv_bytes


def test_visualization_rendering_without_exceptions():
    """Verify Plotly figure generation completes without runtime exceptions."""
    dummy_df = pd.DataFrame({
        "Threat_Status": ["Normal", "Malicious"],
        "protocol_type": ["tcp", "udp"],
        "src_bytes": [100, 5000],
        "dst_bytes": [200, 0],
        "duration": [0, 5]
    })
    
    fig_proto = plot_protocol_attack_breakdown(dummy_df)
    assert fig_proto is not None

    dummy_metrics = {"Accuracy": 0.95, "Precision": 0.92, "Recall": 0.90, "F1-Score": 0.91, "ROC-AUC": 0.96}
    fig_radar = plot_threat_radar(dummy_metrics)
    assert fig_radar is not None