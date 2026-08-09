from datetime import datetime
import io
import pandas as pd


def generate_executive_text_report(df_results, model_name="Random Forest", metrics=None):
    """Generates a text report for SOC managers and security compliance."""
    total_packets = len(df_results)
    
    status_col = "Threat_Status" if "Threat_Status" in df_results.columns else "binary_class"
    malicious_count = (df_results[status_col] == "Malicious").sum()
    normal_count = total_packets - malicious_count
    threat_percentage = (malicious_count / total_packets * 100) if total_packets > 0 else 0.0

    # Risk Assessment Level
    if threat_percentage > 30:
        risk_level = "CRITICAL 🔴"
    elif threat_percentage > 10:
        risk_level = "ELEVATED 🟡"
    else:
        risk_level = "LOW / NORMAL 🟢"

    report_lines = [
        "=" * 68,
        "             INTRUSION DETECTION SYSTEM (IDS) SECURITY REPORT",
        "=" * 68,
        f"Generated Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Organization        : Bleep Education LLP - Cyber Security Project",
        f"Active Classifier   : {model_name}",
        f"Threat Level Status : {risk_level}",
        "-" * 68,
        "1. EXECUTIVE TRAFFIC SUMMARY",
        "-" * 68,
        f"Total Packets Processed : {total_packets:,}",
        f"Normal Traffic Count    : {normal_count:,} ({((normal_count/total_packets)*100):.2f}%)",
        f"Malicious Flags Count   : {malicious_count:,} ({threat_percentage:.2f}%)",
        "",
        "-" * 68,
        "2. PROTOCOL BREAKDOWN",
        "-" * 68,
    ]

    if "protocol_type" in df_results.columns:
        proto_summary = df_results.groupby(["protocol_type", status_col]).size().unstack(fill_value=0)
        for proto in proto_summary.index:
            n_count = proto_summary.loc[proto, "Normal"] if "Normal" in proto_summary.columns else 0
            m_count = proto_summary.loc[proto, "Malicious"] if "Malicious" in proto_summary.columns else 0
            report_lines.append(f"- Protocol [{proto.upper():<4}] -> Normal: {n_count:<6} | Malicious: {m_count:<6}")

    if metrics:
        report_lines.extend([
            "",
            "-" * 68,
            "3. ACTIVE MODEL EVALUATION BENCHMARK",
            "-" * 68,
            f"Accuracy  : {metrics.get('Accuracy', 0):.4f}",
            f"Precision : {metrics.get('Precision', 0):.4f}",
            f"Recall    : {metrics.get('Recall', 0):.4f}",
            f"F1-Score  : {metrics.get('F1-Score', 0):.4f}",
            f"ROC-AUC   : {metrics.get('ROC-AUC', 0):.4f}",
        ])

    report_lines.extend([
        "",
        "-" * 68,
        "4. RECOMMENDED ACTION ITEMS",
        "-" * 68,
        "1. Isolate IP sources exhibiting continuous malicious signature hits." if malicious_count > 0 else "1. No immediate network containment needed.",
        "2. Review firewall block rules for flagged UDP/TCP ports.",
        "3. Export flagged incident CSV logs to SIEM for event correlation.",
        "=" * 68,
        "                      END OF SECURITY REPORT",
        "=" * 68,
    ])

    return "\n".join(report_lines)


def convert_dataframe_to_csv(df):
    """Converts prediction results dataframe into downloadable CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")