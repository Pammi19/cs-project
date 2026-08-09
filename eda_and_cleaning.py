import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure outputs folder exists for saving charts
os.makedirs("outputs", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Attack Mapping Dictionaries for NSL-KDD
ATTACK_MAPPING = {
    'normal': 'Normal',
    # DoS (Denial of Service)
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 'mailbomb': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L (Remote to Local)
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    # U2R (User to Root)
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'ps': 'U2R', 'xterm': 'U2R', 'sqlattack': 'U2R'
}

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, names=COL_NAMES)
    
    # 1. Drop unnecessary columns
    if 'difficulty_level' in df.columns:
        df.drop(columns=['difficulty_level'], inplace=True)
        
    # 2. Check for duplicate records
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"   Dropped {duplicates} duplicate rows.")

    # 3. Add Multi-Class Attack Category
    df['attack_category'] = df['label'].map(ATTACK_MAPPING).fillna('Other')

    # 4. Add Binary Label (0 = Normal, 1 = Malicious)
    df['binary_label'] = df['attack_category'].apply(lambda x: 0 if x == 'Normal' else 1)
    df['binary_class'] = df['attack_category'].apply(lambda x: 'Normal' if x == 'Normal' else 'Malicious')

    return df

def generate_eda_plots(df):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Chart 1: Binary Label Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x='binary_class', palette=['#2ecc71', '#e74c3c'])
    plt.title('Binary Target Distribution (Normal vs Malicious)', fontsize=14, fontweight='bold')
    plt.xlabel('Traffic Type')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('outputs/binary_distribution.png')
    plt.close()

    # Chart 2: Multi-Class Attack Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='attack_category', order=df['attack_category'].value_counts().index, palette='magma')
    plt.title('Attack Category Breakdown (NSL-KDD)', fontsize=14, fontweight='bold')
    plt.xlabel('Category')
    plt.ylabel('Count')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('outputs/attack_category_distribution.png')
    plt.close()

    # Chart 3: Protocol Type by Target Category
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x='protocol_type', hue='binary_class', palette=['#2ecc71', '#e74c3c'])
    plt.title('Protocol Types vs Traffic Class', fontsize=14, fontweight='bold')
    plt.xlabel('Protocol')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/protocol_vs_class.png')
    plt.close()

    # Chart 4: Feature Correlation Heatmap (Numeric)
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    top_corr_features = corr['binary_label'].abs().sort_values(ascending=False).head(12).index
    sns.heatmap(df[top_corr_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Top Correlated Numeric Features Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

def main():
    print("--- Starting Days 3–5: Data Cleaning & EDA ---")
    
    print("\n1. Cleaning KDDTrain+.txt...")
    df_train = load_and_clean_data("datasets/KDDTrain+.txt")
    print(f"   Cleaned Train Set Shape: {df_train.shape}")
    
    print("\n2. Cleaning KDDTest+.txt...")
    df_test = load_and_clean_data("datasets/KDDTest+.txt")
    print(f"   Cleaned Test Set Shape: {df_test.shape}")

    print("\n3. Generating EDA Visualizations in outputs/...")
    generate_eda_plots(df_train)
    print("   Charts saved: binary_distribution.png, attack_category_distribution.png, protocol_vs_class.png, correlation_heatmap.png")

    print("\n4. Saving Cleaned CSV Datasets...")
    df_train.to_csv("datasets/cleaned_train.csv", index=False)
    df_test.to_csv("datasets/cleaned_test.csv", index=False)
    print("   Saved: datasets/cleaned_train.csv and datasets/cleaned_test.csv")

    print("\n--- EDA & Data Cleaning Complete! ---")

if __name__ == "__main__":
    main()