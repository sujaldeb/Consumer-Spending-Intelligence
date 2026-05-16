import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Consumer Spending Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_data
def load_main_data():
    path = os.path.join(os.path.dirname(__file__),
                        "../data/processed/transactions_clean.parquet")
    df = pd.read_parquet(path, engine="pyarrow")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]                   = pd.to_datetime(df["dob"])
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.day_name()
    df["month"]       = df["trans_date_trans_time"].dt.to_period("M")
    df["year"]        = df["trans_date_trans_time"].dt.year
    df["age"]         = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["is_weekend"]  = df["day_of_week"].isin(["Saturday", "Sunday"])
    return df

@st.cache_data
def load_scored_data():
    path = os.path.join(os.path.dirname(__file__),
                        "../data/processed/transactions_scored.parquet")
    df = pd.read_parquet(path, engine="pyarrow")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df

if "df" not in st.session_state:
    with st.spinner("Loading transaction data..."):
        st.session_state.df        = load_main_data()
        st.session_state.df_scored = load_scored_data()

with st.sidebar:
    st.markdown('<p class="sidebar-title">Consumer Spending Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">1.296M transactions · Jan 2019 – Jun 2020</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Navigate**")
    st.page_link("pages/01_overview.py",
                 label="Spending Trends", icon="📈")
    st.page_link("pages/02_spending_trends.py",
                 label="Customer Intelligence", icon="👤")
    st.page_link("pages/03_customer_intelligence.py",
                 label="Geo & Merchant", icon="🗺️")
    st.page_link("pages/04_geo_merchant.py",
                 label="Anomaly Detection", icon="🔍")
    st.page_link("pages/05_anomaly_detection.py",
                 label="Behaviour Shift", icon="📉")
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem; color:#64748b;">Sujal Deb · github.com/sujaldeb</p>',
        unsafe_allow_html=True
    )

df = st.session_state.df

st.markdown("## Consumer Spending Intelligence")
st.markdown(
    '<p style="color:#94a3b8; font-size:0.95rem;">End-to-end financial analytics pipeline · '
    '1.296M transactions · 7 hypothesis tests · ROC-AUC 0.9032</p>',
    unsafe_allow_html=True
)
st.markdown("---")

total_revenue = df["amt"].sum()
total_txns    = len(df)
unique_custs  = df["cc_num"].nunique()
avg_spend     = df["amt"].mean()
fraud_rate    = df["is_fraud"].mean() * 100
fraud_count   = df["is_fraud"].sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Revenue</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;background:#7c3aed;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">${total_revenue/1e6:.1f}M</div>
        </div>
        <div class="kpi-sub">Jan 2019 – Jun 2020</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Transactions</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;background:#3b82f6;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">{total_txns:,}</div>
        </div>
        <div class="kpi-sub">18 months of data</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Unique Customers</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;background:#10b981;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">{unique_custs:,}</div>
        </div>
        <div class="kpi-sub">983 high-frequency buyers</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Transaction</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;background:#f59e0b;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">${avg_spend:.2f}</div>
        </div>
        <div class="kpi-sub">Median $47.52</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Fraud Rate</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;background:#ef4444;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">{fraud_rate:.2f}%</div>
        </div>
        <div class="kpi-sub">{fraud_count:,} fraudulent transactions</div>
    </div>""", unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Anomaly Detection AUC</div>
        <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;background:#a78bfa;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">0.9032</div>
        </div>
        <div class="kpi-sub">Unsupervised · zero label leakage</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Monthly Revenue Trend</p>',
            unsafe_allow_html=True)

monthly = (
    df.groupby("month")["amt"]
    .sum().reset_index()
)
monthly["month_str"] = monthly["month"].astype(str)
monthly["revenue_m"] = monthly["amt"] / 1e6

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly["month_str"],
    y=monthly["revenue_m"],
    mode="lines+markers",
    line=dict(color="#7c3aed", width=2.5),
    marker=dict(size=6, color="#7c3aed"),
    fill="tozeroy",
    fillcolor="rgba(124,58,237,0.1)",
    hovertemplate="<b>%{x}</b><br>Revenue: $%{y:.2f}M<extra></extra>"
))
fig.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b27",
    font=dict(color="#e2e8f0", family="sans-serif"),
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(gridcolor="#1e2a3a", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2a3a", tickprefix="$", ticksuffix="M"),
    showlegend=False,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<p class="section-header">Key Findings</p>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="insight-box violet">
        <strong>grocery_pos dominates revenue at 15.85%</strong> — everyday essential
        spending is the single largest category. Combined shopping categories
        (shopping_pos + shopping_net) total 19.66%.
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Age group 31-45 drives 36.36% of revenue</strong> with a median
        lifetime value of $118,121 — the highest value demographic confirmed
        by Kruskal-Wallis test (p &lt; 0.001).
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box amber">
        <strong>February 2020 confirmed as COVID-19 changepoint</strong> — all 14
        categories declined simultaneously. Travel most impacted at -24.1%
        (Cohen's d = 0.928).
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="insight-box crimson">
        <strong>Anomaly detection ROC-AUC of 0.9032</strong> — fully unsupervised
        Isolation Forest with zero access to fraud labels during training.
        Flagged transactions average 24x the customer's own median spend.
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box violet">
        <strong>Female customers drive 54.6% of revenue</strong> — statistically
        confirmed spend premium of $91.5K vs $77.6K male median lifetime value
        (Mann-Whitney U, p = 0.004, Cohen's d = -0.256).
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <strong>70.9% of fraud anomalies in shopping categories</strong> —
        shopping_net (36.7%) and shopping_pos (34.2%) dominate flagged
        transactions despite being 19.66% of total revenue.
    </div>""", unsafe_allow_html=True)