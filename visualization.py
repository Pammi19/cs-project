import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_threat_radar(metrics):
    """Renders a radar chart showing model evaluation metrics across 5 dimensions."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [metrics.get(cat, 0.0) for cat in categories]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Model Performance",
            line_color="#3498db",
            fillcolor="rgba(52, 152, 219, 0.4)",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="<b>Model Metrics Profile</b>",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_protocol_attack_breakdown(df):
    """Renders a stacked bar chart of protocol types split by threat classification."""
    target_col = "Threat_Status" if "Threat_Status" in df.columns else "binary_class"

    fig = px.histogram(
        df,
        x="protocol_type",
        color=target_col,
        barmode="group",
        title="<b>Protocol Type vs Threat Status</b>",
        color_discrete_map={"Normal": "#2ecc71", "Malicious": "#e74c3c"},
        labels={"protocol_type": "Protocol", "count": "Packet Count"},
    )
    fig.update_layout(legend_title_text="Classification")
    return fig


def plot_byte_distribution(df):
    """Renders a scatter plot showing source vs destination bytes by traffic type."""
    target_col = "Threat_Status" if "Threat_Status" in df.columns else "binary_class"

    fig = px.scatter(
        df,
        x="src_bytes",
        y="dst_bytes",
        color=target_col,
        size="duration",
        hover_data=["service", "flag"],
        title="<b>Network Traffic Distribution (Src Bytes vs Dst Bytes)</b>",
        color_discrete_map={"Normal": "#2ecc71", "Malicious": "#e74c3c"},
        log_x=True,
        log_y=True,
    )
    fig.update_layout(
        xaxis_title="Source Bytes (Log Scale)",
        yaxis_title="Destination Bytes (Log Scale)",
    )
    return fig


def plot_confidence_histogram(df):
    """Renders confidence distribution histogram for live predictions."""
    if "Confidence_Score" not in df.columns:
        return None

    # Parse confidence percentage string to float
    df_plot = df.copy()
    if df_plot["Confidence_Score"].dtype == object:
        df_plot["Confidence_Val"] = (
            df_plot["Confidence_Score"].str.rstrip("%").astype(float)
        )
    else:
        df_plot["Confidence_Val"] = df_plot["Confidence_Score"] * 100

    fig = px.histogram(
        df_plot,
        x="Confidence_Val",
        color="Threat_Status",
        nbins=20,
        title="<b>Threat Confidence Score Distribution</b>",
        color_discrete_map={"Normal": "#2ecc71", "Malicious": "#e74c3c"},
        labels={"Confidence_Val": "Confidence Percentage (%)"},
    )
    return fig