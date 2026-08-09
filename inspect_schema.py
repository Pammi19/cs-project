import pandas as pd

# Official NSL-KDD Feature Column Names (41 Features + Label + Difficulty)
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

def inspect_dataset():
    print("--- Loading NSL-KDD Dataset ---")
    df = pd.read_csv("datasets/KDDTrain+.txt", names=COL_NAMES)

    print(f"\n1. Shape of Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n2. Column Data Types:")
    print(df.dtypes.value_counts())

    print("\n3. Categorical Columns Summary:")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    for col in cat_cols:
        print(f"\nUnique values in '{col}': {df[col].nunique()}")
        print(df[col].value_counts().head(5))

    print("\n4. Missing Values Count:")
    null_count = df.isnull().sum().sum()
    print(f"Total Null Values: {null_count}")

    print("\n5. Top 10 Attack Labels Distribution:")
    print(df['label'].value_counts().head(10))

if __name__ == "__main__":
    inspect_dataset()