<<<<<<< HEAD
# cs-project
=======
# 🛡️ AI-Powered Network Intrusion Detection System (IDS)

An end-to-end Machine Learning solution for detecting, classifying, and reporting malicious network traffic signatures in real time. Built using the **NSL-KDD dataset**, **Scikit-Learn**, **Streamlit**, and **Plotly**.

---

##  Project Architecture & Workflow

```text
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
```
## 📂 Project Directory Structure

Intrusion-Detection-System/
├── datasets/                  # Raw and preprocessed CSV/NPZ files
│   ├── cleaned_train.csv
│   ├── cleaned_test.csv
│   └── processed_data.npz
├── models/                    # Trained model bundles and scaler pipelines
│   ├── feature_pipeline.pkl
│   └── ids_model.pkl
├── app.py                     # Main 6-module Streamlit interactive dashboard
├── eda_and_cleaning.py        # Days 1–5: Data Cleaning & Exploratory Data Analysis
├── feature_engineering.py     # Days 6–8: Encoding, Scaling & Feature Selection
├── train_models.py            # Days 9–11: ML Model Training & Benchmarking
├── visualization.py           # Days 15–17: Plotly Chart Helper Functions
├── reports.py                 # Days 15–17: Executive Text & CSV Report Generators
├── test_ids.py                # Days 18–19: Automated pytest Suite
├── requirements.txt           # Production Dependencies
├── Dockerfile                 # Container Deployment Configuration
└── README.md                  # Project Documentation
```
```
## Quickstart & Pipeline Execution
# 1. Environment Setup
```bash
git clone [https://github.com/your-username/Intrusion-Detection-System.git](https://github.com/your-username/Intrusion-Detection-System.git)
cd Intrusion-Detection-System
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

# 2. Run Pipeline Steps (Sequential Execution)

# Step 1: Preprocess and clean NSL-KDD raw data
python eda_and_cleaning.py

# Step 2: Feature encoding, feature selection (Top 25), and scaling
python feature_engineering.py

# Step 3: Train models (Random Forest, Decision Tree, Logistic Regression, KNN)
python train_models.py

# Step 4: Execute automated unit & integration test suite
pytest -v test_ids.py

# Step 5: Launch the Interactive Streamlit Dashboard
streamlit run app.py


## 📈 Model Performance Benchmark

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.8012** | **0.9521** | **0.6710** | **0.7871** | **0.9104** | **Selected (Saved)** |
| Decision Tree | 0.7645 | 0.8912 | 0.6432 | 0.7472 | 0.7710 | Benchmark |
| K-Nearest Neighbors | 0.7410 | 0.8720 | 0.6120 | 0.7190 | 0.8234 | Benchmark |
| Logistic Regression | 0.7180 | 0.8350 | 0.5840 | 0.6872 | 0.7950 | Baseline |

---

## 🖥️ Streamlit Dashboard Modules

The application features 6 integrated control modules:

1. **Home & Overview**: System metrics and end-to-end processing pipeline architecture.
2. **Dataset Explorer**: Class distribution analytics and attack category distributions.
3. **Model Performance**: Confusion matrix displays, ROC-AUC benchmarks, and top Gini feature importances.
4. **Live Threat Prediction**: Interactive packet file ingestion and ML model inference.
5. **Executive Analytics**: Interactive Plotly radar charts, traffic byte scatter plots, and confidence histograms.
6. **Incident Reporting**: Automated executive security report generation with `.TXT` and `.CSV` export capabilities.

---

## 🧪 Testing & Validation Strategy

Automated validation is handled via `pytest` to prevent runtime failures during stream processing:

```bash
# Run complete test suite
pytest -v test_ids.py
```
## 🐳 Container Deployment (Docker)

To run the full system inside an isolated container:

```bash
# 1. Build the Docker image
docker build -t ids-dashboard:latest .

# 2. Run the application container
docker run -p 8501:8501 ids-dashboard:latest
>>>>>>> e62831f (https://github.com/Pammi19/cs-project.git)
