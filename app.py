import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Custom imports from your project files
from visualization import (
    plot_threat_radar,
    plot_protocol_attack_breakdown,
    plot_byte_distribution,
    plot_confidence_histogram,
)
from reports import generate_executive_text_report, convert_dataframe_to_csv

# Page Configuration
st.set_page_config(
    page_title="Intrusion Detection System (IDS)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Trained Model Artifacts
@st.cache_resource
def load_system_artifacts():
    model_path = "models/ids_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model_bundle = load_system_artifacts()

# Sidebar Navigation - All 6 Modules Integrated
st.sidebar.title(" IDS Control Center")
st.sidebar.caption("Bleep Education LLP - Cybersecurity Project")

nav_choice = st.sidebar.radio(
    "Navigation Modules:",
    [
        "1. Home & Overview",
        "2. Dataset Explorer",
        "3. Model Performance",
        "4. Live Threat Prediction",
        "5. Executive Analytics & Charts",
        "6. Incident Reporting & Export"
    ]
)

# -------------------------------------------------------------------
# Module 1: Home & Overview
# -------------------------------------------------------------------
if nav_choice == "1. Home & Overview":
    # --- Modern AI Operations Center CSS Injection ---
    st.markdown("""
        <style>
        /* Hero Banner Styling */
        .hero-container {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 16px;
            padding: 2.2rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ffffff 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
            line-height: 1.2;
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            max-width: 820px;
            line-height: 1.6;
        }

        /* Feature Card Styling */
        .feature-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.3rem;
            height: 100%;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .feature-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }
        .feature-icon {
            font-size: 1.6rem;
            margin-bottom: 0.5rem;
        }
        .feature-title {
            color: #f8fafc;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .feature-desc {
            color: #94a3b8;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- 1. Hero Header Banner ---
    st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">⚡ Real-Time Cyber Threat Intelligence</span>
            <div class="hero-title">🛡️ Network Intrusion Detection System (IDS)</div>
            <div class="hero-subtitle">
                An end-to-end Machine Learning platform designed to detect, classify, and report malicious network traffic signatures in real time. Powered by standard Scikit-Learn pipelines, Random Forest classifiers, and Plotly executive analytics.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 2. Live System Telemetry Metrics ---
    st.subheader("📊 System Telemetry & ML Status")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(label="Primary Dataset", value="NSL-KDD", delta="Benchmark")
    col2.metric(label="Selected Features", value="25 Engineered", delta="Top Gini")
    col3.metric(
        label="Primary Algorithm", 
        value=model_bundle["model_name"] if model_bundle else "Not Trained",
        delta="Random Forest" if model_bundle else "Offline",
        delta_color="normal" if model_bundle else "off"
    )
    col4.metric(
        label="System Status", 
        value="ONLINE 🟢" if model_bundle else "OFFLINE 🔴",
        delta="Active" if model_bundle else "Requires Training",
        delta_color="normal" if model_bundle else "inverse"
    )

    st.markdown("---")

    # --- 3. Key Capabilities Grid ---
    st.subheader("🚀 Core Capabilities")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Real-Time Threat Detection</div>
                <div class="feature-desc">
                    Infers live packet file uploads against trained decision boundary thresholds to instantly flag <b>Normal</b> vs <b>Malicious</b> network traffic.
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")  # Spacer

        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Executive Analytics</div>
                <div class="feature-desc">
                    Generates interactive Plotly radar charts, packet scatter distributions, and ROC-AUC benchmarks for deep network traffic profiling.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚙️</div>
                <div class="feature-title">Automated Feature Pipeline</div>
                <div class="feature-desc">
                    Handles categorical LabelEncoding, StandardScaler normalization, and automatic fallback handling for unseen protocol types.
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")  # Spacer

        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Incident Reporting & Audit</div>
                <div class="feature-desc">
                    Exports SOC-ready executive security summaries and tagged traffic logs in <code>.TXT</code> and <code>.CSV</code> formats for incident response.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- 4. Interactive Tabs for Architecture, Objectives & Tech Stack ---
    st.subheader("🧩 System Architecture & Strategy")

    tab_arch, tab_obj, tab_tech = st.tabs([
        "🔄 Architecture Workflow", 
        "🎯 Primary Objectives", 
        "🛠️ Tech Stack"
    ])

    with tab_arch:
        st.code("""
+-----------------------+     +--------------------------+     +---------------------------+
| Network Packet Data   | --> | Preprocessing & Scaling  | --> | Machine Learning Model    |
| (CSV / Log Capture)   |     | (LabelEnc + StandardScaler)|    | (Random Forest Classifier)|
+-----------------------+     +--------------------------+     +---------------------------+
                                                                             |
                                                                             v
+-----------------------+     +--------------------------+     +---------------------------+
| Security Reports      | <-- | Analytics & Charts       | <-- | Threat Detection Result   |
| (.TXT / .CSV Export)  |     | (Plotly Visualizations)  |     | (Normal vs Malicious)     |
+-----------------------+     +--------------------------+     +---------------------------+
        """, language="text")

    with tab_obj:
        st.markdown("""
        * **Automated Threat Classification**: Instantly identify signature anomalies and malicious behavior across HTTP, TCP, UDP, and ICMP protocols.
        * **Feature Optimization**: Reduce raw 41-feature inputs down to **25 high-impact statistical features** (focusing on duration, byte counts, and error flags).
        * **Low False Positive Rate**: Maintain precise decision boundaries so benign administrative traffic is not misflagged as security breaches.
        * **Operational Deployment**: Ensure end-to-end container readiness via Docker and continuous unit testing through Pytest.
        """)

    with tab_tech:
        st.markdown("""
        | Component | Technologies Used |
        | :--- | :--- |
        | **Frontend / UI** | Streamlit, HTML5, Custom CSS3 |
        | **Machine Learning** | Scikit-Learn (Random Forest, Decision Tree, KNN, Logistic Regression) |
        | **Data Engineering** | Pandas, NumPy, StandardScaler, LabelEncoder |
        | **Visualizations** | Plotly Express, Plotly Graph Objects |
        | **Testing & DevOps** | Pytest, Docker, Joblib |
        """)

    st.info("💡 **Quickstart**: Select **'4. Live Threat Prediction'** from the sidebar to upload network packet logs and run inference.")

# -------------------------------------------------------------------
# Module 2: Dataset Explorer
# -------------------------------------------------------------------
elif nav_choice == "2. Dataset Explorer":
    st.title("📊 NSL-KDD Dataset Analytics")

    train_path = "datasets/cleaned_train.csv"
    if os.path.exists(train_path):
        df_clean = pd.read_csv(train_path)

        st.subheader("Dataset Summary Statistics")
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Total Records:** {df_clean.shape[0]:,}")
        c2.write(f"**Total Columns:** {df_clean.shape[1]}")
        c3.write(f"**Malicious Ratio:** {(df_clean['binary_label'].mean() * 100):.2f}%")

        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(df_clean.head(100), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Traffic Class Distribution")
            fig_pie = px.pie(
                df_clean, 
                names='binary_class', 
                color='binary_class',
                color_discrete_map={'Normal': '#2ecc71', 'Malicious': '#e74c3c'},
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Attack Families Breakdown")
            attack_counts = df_clean['attack_category'].value_counts().reset_index()
            attack_counts.columns = ['Category', 'Count']
            fig_bar = px.bar(
                attack_counts, 
                x='Category', 
                y='Count', 
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Cleaned dataset not found in `datasets/cleaned_train.csv`. Run `eda_and_cleaning.py` first.")

# -------------------------------------------------------------------
# Module 3: Model Performance
# -------------------------------------------------------------------
elif nav_choice == "3. Model Performance":
    st.title("📈 Machine Learning Performance Metrics")

    if model_bundle is None:
        st.warning("⚠️ Saved model bundle not found in `models/ids_model.pkl`. Please run `train_models.py` first.")
    else:
        metrics = model_bundle["metrics"]
        st.subheader(f"Active Model: {model_bundle['model_name']}")

        # Key Metrics Displays
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        m2.metric("Precision", f"{metrics['Precision']:.4f}")
        m3.metric("Recall", f"{metrics['Recall']:.4f}")
        m4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        m5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = metrics["Confusion Matrix"]
            fig_cm = px.imshow(
                cm,
                x=["Normal", "Malicious"],
                y=["Normal", "Malicious"],
                text_auto=True,
                color_continuous_scale="Reds",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            st.subheader("Top Feature Importances")
            model = model_bundle["model"]
            selected_features = model_bundle["selected_features"]

            if hasattr(model, "feature_importances_"):
                imp_df = pd.DataFrame({
                    "Feature": selected_features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=True).tail(10)

                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance display is available for tree-based models.")

# -------------------------------------------------------------------
# Module 4: Live Threat Prediction
# -------------------------------------------------------------------
elif nav_choice == "4. Live Threat Prediction":
    st.title("🔍 Live Packet Analysis & Threat Prediction")

    if model_bundle is None:
        st.warning("⚠️ Saved model bundle not found. Please run `train_models.py` before performing inference.")
    else:
        st.subheader("Upload Network Traffic Logs")
        uploaded_file = st.file_uploader("Upload CSV File (NSL-KDD Format)", type=["csv"])

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Loaded {input_df.shape[0]} packet records.")
            
            if st.button(" Run Threat Detection Analysis"):
                with st.spinner("Processing network packets and applying classifier..."):
                    encoders = model_bundle["encoders"]
                    scaler = model_bundle["scaler"]
                    selected_features = model_bundle["selected_features"]
                    model = model_bundle["model"]

                    df_proc = input_df.copy()

                    # Apply saved encoders safely
                    for col, le in encoders.items():
                        if col in df_proc.columns:
                            df_proc[col] = df_proc[col].astype(str).map(
                                lambda s: le.transform([s])[0] if s in le.classes_ else 0
                            )

                    # --- ADDED VALIDATION CHECK ---
                    # Verify that all required model features exist in the uploaded CSV
                    missing_features = [col for col in selected_features if col not in df_proc.columns]

                    if missing_features:
                        st.warning(
                            f"⚠️ **Invalid Dataset Schema:** The uploaded CSV file is missing {len(missing_features)} expected feature columns "
                            f"(such as: `{', '.join(missing_features[:3])}`). Please upload a valid NSL-KDD dataset file (e.g., `cleaned_test.csv`)."
                        )
                        st.stop()  # Safely halts execution without crashing the dashboard
                    # ------------------------------

                    X_input = df_proc[selected_features].values
                    X_scaled = scaler.transform(X_input)

                    predictions = model.predict(X_scaled)
                    probas = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(predictions)

                    results_df = input_df.copy()
                    results_df["Threat_Status"] = ["Malicious" if p == 1 else "Normal" for p in predictions]
                    results_df["Confidence_Score"] = [f"{p:.2%}" for p in probas]

                    # Store results in session_state for Module 5 and 6
                    st.session_state["inference_results"] = results_df

                    st.markdown("---")
                    st.subheader("Inference Summary")
                    malicious_count = (results_df["Threat_Status"] == "Malicious").sum()
                    normal_count = len(results_df) - malicious_count

                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Total Analyzed", len(results_df))
                    res_col2.metric("Normal Packets", normal_count)
                    res_col3.metric("Malicious Flagged", malicious_count, delta=f"{malicious_count}", delta_color="inverse")

                    st.subheader("Detailed Traffic Classification")
                    st.dataframe(results_df, use_container_width=True)

# -------------------------------------------------------------------
# Module 5: Executive Analytics & Charts
# -------------------------------------------------------------------
elif nav_choice == "5. Executive Analytics & Charts":
    st.title("📊 Advanced Security Analytics Dashboard")

    df_display = st.session_state.get("inference_results")
    if df_display is None:
        if os.path.exists("datasets/cleaned_train.csv"):
            st.info("💡 Displaying baseline training dataset analytics. Upload a live log in Module 4 to view real-time session analytics.")
            df_display = pd.read_csv("datasets/cleaned_train.csv").head(2000)
            df_display["Threat_Status"] = df_display["binary_class"]
        else:
            st.warning("⚠️ No traffic data available.")

    if df_display is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_protocol_attack_breakdown(df_display), use_container_width=True)
        with col2:
            if model_bundle and "metrics" in model_bundle:
                st.plotly_chart(plot_threat_radar(model_bundle["metrics"]), use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_byte_distribution(df_display), use_container_width=True)
        with col4:
            fig_conf = plot_confidence_histogram(df_display)
            if fig_conf:
                st.plotly_chart(fig_conf, use_container_width=True)
            else:
                st.info("Run live prediction in Module 4 to view prediction confidence scores.")

# -------------------------------------------------------------------
# Module 6: Incident Reporting & Export
# -------------------------------------------------------------------
elif nav_choice == "6. Incident Reporting & Export":
    st.title("📄 Incident Reporting & File Export")

    df_report = st.session_state.get("inference_results")
    if df_report is None:
        st.info("💡 Generating sample report preview based on training data. Run predictions in Module 4 for custom session reports.")
        if os.path.exists("datasets/cleaned_train.csv"):
            df_report = pd.read_csv("datasets/cleaned_train.csv").head(1000)
            df_report["Threat_Status"] = df_report["binary_class"]
        else:
            df_report = pd.DataFrame()

    if not df_report.empty:
        metrics = model_bundle["metrics"] if model_bundle else None
        model_name = model_bundle["model_name"] if model_bundle else "Default ML Model"

        report_text = generate_executive_text_report(df_report, model_name=model_name, metrics=metrics)

        st.subheader("Executive Security Summary Preview")
        st.text_area("Security Incident Report", report_text, height=380)

        st.markdown("---")
        st.subheader("📥 Export Deliverables")

        c1, c2 = st.columns(2)

        c1.download_button(
            label="💾 Download Security Report (.TXT)",
            data=report_text,
            file_name="IDS_Executive_Security_Report.txt",
            mime="text/plain",
        )

        csv_bytes = convert_dataframe_to_csv(df_report)
        c2.download_button(
            label="📊 Download Classified Records (.CSV)",
            data=csv_bytes,
            file_name="IDS_Classified_Traffic_Logs.csv",
            mime="text/csv",
        )
