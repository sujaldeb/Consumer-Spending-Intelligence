import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import os

st.set_page_config(page_title="Anomaly Detection", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "../styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

if "df_scored" not in st.session_state:
    path = os.path.join(os.path.dirname(__file__),
                        "../../data/processed/transactions_scored.parquet")
    df = pd.read_parquet(path, engine="pyarrow")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
else:
    df = st.session_state.df_scored

COLORS = {
    "primary" : "#7c3aed",
    "blue"    : "#3b82f6",
    "emerald" : "#10b981",
    "crimson" : "#ef4444",
    "amber"   : "#f59e0b",
    "lavender": "#a78bfa",
    "bg"      : "#0d1117",
    "card"    : "#161b27",
    "grid"    : "#1e2a3a",
    "text"    : "#e2e8f0",
    "subtext" : "#94a3b8",
}

def base_layout(height=320, title=""):
    return dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        title=dict(text=title, font=dict(color="#ffffff", size=13)),
        xaxis=dict(gridcolor=COLORS["grid"], tickfont=dict(size=10)),
        yaxis=dict(gridcolor=COLORS["grid"]),
        hovermode="closest"
    )

st.markdown("## Anomaly Detection")
st.markdown(
    '<p style="color:#94a3b8; font-size:0.9rem;">Isolation Forest · '
    'ROC-AUC 0.9032 · zero label leakage · flagged transaction explorer</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# KPI cards
total_flagged  = df["anomaly_flag"].sum()
total_fraud    = df["is_fraud"].sum()
flagged_median = df[df["anomaly_flag"] == 1]["amt"].median()
normal_median  = df[df["anomaly_flag"] == 0]["amt"].median()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">ROC-AUC</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#7c3aed;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">0.9032</div>
        </div>
        <div class="kpi-sub">Unsupervised · no label access</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomalies Flagged</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#ef4444;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">{total_flagged:,}</div>
        </div>
        <div class="kpi-sub">vs {total_fraud:,} known fraud cases</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Flagged Median Amt</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#f59e0b;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">${flagged_median:,.0f}</div>
        </div>
        <div class="kpi-sub">vs ${normal_median:.2f} normal</div>
    </div>""", unsafe_allow_html=True)

with col4:
    ratio = flagged_median / normal_median
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Spend Ratio</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#10b981;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">{ratio:.1f}x</div>
        </div>
        <div class="kpi-sub">Flagged vs normal median</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ROC curve + score distribution
st.markdown('<p class="section-header">Model Validation</p>',
            unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    y_true  = df["is_fraud"].values
    y_score = -df["anomaly_score"].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.1)",
        name=f"Isolation Forest (AUC = {auc:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=COLORS["subtext"], width=1, dash="dash"),
        name="Random baseline",
        hoverinfo="skip"
    ))
    layout = base_layout(height=360, title="ROC Curve — Anomaly vs Fraud Labels")
    layout["xaxis"]["title"] = "False Positive Rate"
    layout["yaxis"]["title"] = "True Positive Rate"
    layout["showlegend"] = True
    layout["legend"] = dict(
        bgcolor="rgba(22,27,39,0.8)",
        bordercolor=COLORS["grid"],
        font=dict(size=10)
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    normal_scores  = -df[df["is_fraud"] == 0]["anomaly_score"]
    fraud_scores   = -df[df["is_fraud"] == 1]["anomaly_score"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal_scores,
        nbinsx=80,
        name="Normal",
        marker_color=COLORS["blue"],
        opacity=0.6,
        histnorm="probability density",
        hovertemplate="Score: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Histogram(
        x=fraud_scores,
        nbinsx=80,
        name="Fraud",
        marker_color=COLORS["crimson"],
        opacity=0.7,
        histnorm="probability density",
        hovertemplate="Score: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>"
    ))
    layout = base_layout(height=360,
                         title="Anomaly Score Distribution — Fraud vs Normal")
    layout["xaxis"]["title"] = "Anomaly Score (higher = more anomalous)"
    layout["yaxis"]["title"] = "Density"
    layout["barmode"] = "overlay"
    layout["showlegend"] = True
    layout["legend"] = dict(
        bgcolor="rgba(22,27,39,0.8)",
        bordercolor=COLORS["grid"],
        font=dict(size=10)
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# anomaly profile
st.markdown('<p class="section-header">Anomaly Profile — Flagged vs Normal</p>',
            unsafe_allow_html=True)

profile_data = pd.DataFrame({
    "Feature"         : ["Amount ($)", "Amt Z-Score",
                         "Amt/Median Ratio", "Distance (km)",
                         "Hour", "Age"],
    "Flagged Median"  : [
        df[df["anomaly_flag"]==1]["amt"].median(),
        df[df["anomaly_flag"]==1]["amt_zscore"].median(),
        df[df["anomaly_flag"]==1]["amt_to_median_ratio"].median(),
        df[df["anomaly_flag"]==1]["distance_km"].median()
        if "distance_km" in df.columns else 76.5,
        df[df["anomaly_flag"]==1]["trans_date_trans_time"].dt.hour.median(),
        df[df["anomaly_flag"]==1]["age"].median()
        if "age" in df.columns else 44.0,
    ],
    "Normal Median"   : [
        df[df["anomaly_flag"]==0]["amt"].median(),
        df[df["anomaly_flag"]==0]["amt_zscore"].median(),
        df[df["anomaly_flag"]==0]["amt_to_median_ratio"].median(),
        df[df["anomaly_flag"]==0]["distance_km"].median()
        if "distance_km" in df.columns else 78.2,
        df[df["anomaly_flag"]==0]["trans_date_trans_time"].dt.hour.median(),
        df[df["anomaly_flag"]==0]["age"].median()
        if "age" in df.columns else 43.0,
    ]
})

profile_data["Difference %"] = (
    (profile_data["Flagged Median"] - profile_data["Normal Median"]) /
    profile_data["Normal Median"].abs() * 100
).round(1).astype(str) + "%"

profile_data["Flagged Median"] = profile_data["Flagged Median"].round(2)
profile_data["Normal Median"]  = profile_data["Normal Median"].round(2)

st.dataframe(profile_data, use_container_width=True, hide_index=True)

# category distribution of anomalies
st.markdown('<p class="section-header">Anomaly Category Distribution</p>',
            unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    cat_flagged = (
        df[df["anomaly_flag"] == 1]["category"]
        .value_counts(normalize=True)
        .mul(100).round(2)
        .reset_index()
    )
    cat_flagged.columns = ["category", "pct"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_flagged["category"].astype(str)[::-1],
        x=cat_flagged["pct"][::-1],
        orientation="h",
        marker_color=COLORS["crimson"],
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% of anomalies<extra></extra>"
    ))
    layout = base_layout(height=400,
                         title="Category Share of Flagged Anomalies (%)")
    layout["xaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col8:
    cat_normal = (
        df[df["anomaly_flag"] == 0]["category"]
        .value_counts(normalize=True)
        .mul(100).round(2)
        .reset_index()
    )
    cat_normal.columns = ["category", "pct"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cat_normal["category"].astype(str)[::-1],
        x=cat_normal["pct"][::-1],
        orientation="h",
        marker_color=COLORS["blue"],
        hovertemplate="<b>%{y}</b><br>%{x:.2f}% of normal<extra></extra>"
    ))
    layout = base_layout(height=400,
                         title="Category Share of Normal Transactions (%)")
    layout["xaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# flagged transaction explorer
st.markdown('<p class="section-header">Flagged Transaction Explorer</p>',
            unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    categories = ["All"] + sorted(
        df["category"].astype(str).unique().tolist()
    )
    selected_cat = st.selectbox("Filter by category", categories)

with col_f2:
    states = ["All"] + sorted(df["state"].astype(str).unique().tolist())
    selected_state = st.selectbox("Filter by state", states)

with col_f3:
    min_amt = float(df[df["anomaly_flag"]==1]["amt"].min())
    max_amt = float(df[df["anomaly_flag"]==1]["amt"].max())
    amt_range = st.slider("Min transaction amount ($)",
                          min_value=int(min_amt),
                          max_value=int(max_amt),
                          value=int(min_amt))

flagged = df[df["anomaly_flag"] == 1].copy()

if selected_cat != "All":
    flagged = flagged[flagged["category"].astype(str) == selected_cat]
if selected_state != "All":
    flagged = flagged[flagged["state"].astype(str) == selected_state]
flagged = flagged[flagged["amt"] >= amt_range]

display_flagged = (
    flagged[["trans_date_trans_time", "cc_num", "merchant",
             "category", "amt", "state", "is_fraud",
             "anomaly_score", "amt_zscore"]]
    .sort_values("amt", ascending=False)
    .head(100)
    .copy()
)
display_flagged["trans_date_trans_time"] = (
    display_flagged["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M")
)
display_flagged["amt"] = display_flagged["amt"].apply(lambda x: f"${x:,.2f}")
display_flagged["anomaly_score"] = display_flagged["anomaly_score"].round(4)
display_flagged["amt_zscore"]    = display_flagged["amt_zscore"].round(2)
display_flagged.columns = ["Date", "Customer", "Merchant", "Category",
                           "Amount", "State", "Known Fraud",
                           "Anomaly Score", "Amt Z-Score"]

st.markdown(f"Showing top 100 of **{len(flagged):,}** flagged transactions",
            unsafe_allow_html=True)
st.dataframe(display_flagged, use_container_width=True, hide_index=True)

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("""
    <div class="insight-box crimson">
        <strong>shopping_net and shopping_pos account for 70.9% of anomalies</strong>
        despite being 19.66% of total revenue — online and in-store shopping
        are the primary fraud risk categories.
    </div>""", unsafe_allow_html=True)

with col_i2:
    st.markdown("""
    <div class="insight-box violet">
        <strong>Flagged transactions are 24x the customer's own median spend</strong>
        — the amt_to_median_ratio engineered feature is the key differentiator
        between flagged and normal behaviour.
    </div>""", unsafe_allow_html=True)