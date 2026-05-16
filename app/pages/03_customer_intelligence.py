import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(page_title="Geo & Merchant", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "../styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

if "df" not in st.session_state:
    path = os.path.join(os.path.dirname(__file__),
                        "../../data/processed/transactions_clean.parquet")
    df = pd.read_parquet(path, engine="pyarrow")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
else:
    df = st.session_state.df

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

st.markdown("## Geographic & Merchant Intelligence")
st.markdown(
    '<p style="color:#94a3b8; font-size:0.9rem;">State revenue heatmap · '
    'merchant Pareto · correlation hypothesis tests</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# state aggregation
state_summary = (
    df.groupby("state", observed=True)["amt"]
    .agg(
        transactions  = "count",
        total_revenue = "sum",
        median_amt    = "median"
    )
    .reset_index()
)
state_summary["revenue_share"] = (
    state_summary["total_revenue"] /
    state_summary["total_revenue"].sum() * 100
).round(2)
state_summary = state_summary.sort_values("total_revenue", ascending=False)

# KPI row
st.markdown('<p class="section-header">Geographic Overview</p>',
            unsafe_allow_html=True)

top_state     = state_summary.iloc[0]
top3_share    = state_summary.head(3)["revenue_share"].sum()
median_median = state_summary["median_amt"].median()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Top State</div>
        <div class="kpi-value violet">{top_state['state']}</div>
        <div class="kpi-sub">{top_state['revenue_share']:.1f}% of revenue</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Top 3 States Share</div>
        <div class="kpi-value blue">{top3_share:.1f}%</div>
        <div class="kpi-sub">TX · NY · PA</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">States Analysed</div>
        <div class="kpi-value emerald">51</div>
        <div class="kpi-sub">Including DC</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Median Txn (typical state)</div>
        <div class="kpi-value amber">${median_median:.2f}</div>
        <div class="kpi-sub">Uniform across states</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# choropleth map
st.markdown('<p class="section-header">State Revenue Heatmap</p>',
            unsafe_allow_html=True)

fig = go.Figure(data=go.Choropleth(
    locations=state_summary["state"].astype(str),
    z=state_summary["total_revenue"] / 1e6,
    locationmode="USA-states",
    colorscale=[
        [0.0, "#0d1117"],
        [0.2, "#2d1b69"],
        [0.5, "#5b21b6"],
        [1.0, "#7c3aed"]
    ],
    colorbar=dict(
        title=dict(text="Revenue ($M)",
                   font=dict(color=COLORS["subtext"])),
        tickfont=dict(color=COLORS["subtext"])
    ),
    hovertemplate="<b>%{location}</b><br>Revenue: $%{z:.2f}M<extra></extra>"
))

fig.update_layout(
    paper_bgcolor=COLORS["bg"],
    geo=dict(
        scope="usa",
        bgcolor=COLORS["bg"],
        lakecolor=COLORS["bg"],
        landcolor="#161b27",
        showlakes=True,
        showframe=False,
        coastlinecolor=COLORS["grid"],
        projection_type="albers usa"
    ),
    font=dict(color=COLORS["text"]),
    margin=dict(l=0, r=0, t=40, b=0),
    height=420,
    title=dict(text="Total Revenue by State ($M)",
               font=dict(color="#ffffff", size=13))
)
st.plotly_chart(fig, use_container_width=True)

# top 20 states bar
col5, col6 = st.columns(2)

with col5:
    top20 = state_summary.head(20)
    bar_colors = ([COLORS["primary"]] * 5 +
                  [COLORS["blue"]] * 5 +
                  [COLORS["emerald"]] * 10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top20["state"].astype(str)[::-1],
        x=top20["total_revenue"][::-1] / 1e6,
        orientation="h",
        marker_color=bar_colors[::-1],
        hovertemplate="<b>%{y}</b><br>Revenue: $%{x:.2f}M<extra></extra>"
    ))
    layout = base_layout(height=500, title="Top 20 States by Revenue ($M)")
    layout["xaxis"]["ticksuffix"] = "M"
    layout["yaxis"]["tickfont"] = dict(size=10)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top20["state"].astype(str)[::-1],
        x=top20["median_amt"][::-1],
        orientation="h",
        marker_color=COLORS["amber"],
        hovertemplate="<b>%{y}</b><br>Median Txn: $%{x:.2f}<extra></extra>"
    ))
    layout = base_layout(height=500,
                         title="Median Transaction Value — Top 20 States ($)")
    layout["xaxis"]["tickprefix"] = "$"
    layout["yaxis"]["tickfont"] = dict(size=10)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# merchant analysis
st.markdown('<p class="section-header">Merchant Performance & Pareto</p>',
            unsafe_allow_html=True)

merchant_summary = (
    df.groupby("merchant", observed=True)
    .agg(
        transactions  = ("amt", "count"),
        total_revenue = ("amt", "sum"),
        median_amt    = ("amt", "median"),
        category      = ("category", "first"),
        fraud_count   = ("is_fraud", "sum"),
    )
    .reset_index()
)
merchant_summary["fraud_rate"] = (
    merchant_summary["fraud_count"] /
    merchant_summary["transactions"] * 100
).round(2)
merchant_summary["revenue_share"] = (
    merchant_summary["total_revenue"] /
    merchant_summary["total_revenue"].sum() * 100
).round(3)
merchant_summary = merchant_summary.sort_values(
    "total_revenue", ascending=False
).reset_index(drop=True)
merchant_summary["cumulative_share"] = merchant_summary["revenue_share"].cumsum()

merchants_80 = merchant_summary[merchant_summary["cumulative_share"] <= 80]

col7, col8 = st.columns(2)

with col7:
    top20_merch = merchant_summary.head(20)
    # clean merchant names for display
    top20_merch = top20_merch.copy()
    top20_merch["merchant_clean"] = (
        top20_merch["merchant"].astype(str)
        .str.replace("fraud_", "", regex=False)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top20_merch["merchant_clean"][::-1],
        x=top20_merch["total_revenue"][::-1] / 1000,
        orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate="<b>%{y}</b><br>Revenue: $%{x:.0f}K<extra></extra>"
    ))
    layout = base_layout(height=500, title="Top 20 Merchants by Revenue ($K)")
    layout["xaxis"]["ticksuffix"] = "K"
    layout["yaxis"]["tickfont"] = dict(size=9)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col8:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(merchant_summary) + 1)),
        y=merchant_summary["cumulative_share"],
        mode="lines",
        line=dict(color=COLORS["emerald"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.1)",
        hovertemplate="Merchants: %{x}<br>Cumulative Share: %{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(y=80, line_color=COLORS["amber"],
                  line_dash="dash", line_width=1.5,
                  annotation_text="80% threshold",
                  annotation_font_color=COLORS["amber"])
    fig.add_vline(x=len(merchants_80), line_color=COLORS["crimson"],
                  line_dash="dash", line_width=1.5,
                  annotation_text=f"{len(merchants_80)} merchants",
                  annotation_font_color=COLORS["crimson"])
    layout = base_layout(height=500, title="Merchant Revenue Pareto Curve")
    layout["xaxis"]["title"] = "Number of Merchants"
    layout["yaxis"]["title"] = "Cumulative Revenue Share (%)"
    layout["yaxis"]["range"] = [0, 105]
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# hypothesis test results
st.markdown('<p class="section-header">Hypothesis Test Results</p>',
            unsafe_allow_html=True)

test_data = pd.DataFrame({
    "Test"          : ["Spearman Correlation", "Spearman Correlation"],
    "Question"      : [
        "Does city population predict transaction amount?",
        "Does transaction distance predict fraud rate?"
    ],
    "Result"        : ["Reject H0", "Fail to reject H0"],
    "Spearman rho"  : ["0.137", "0.034"],
    "p-value"       : ["< 0.001", "0.370"],
    "Interpretation": [
        "Weak positive — larger cities spend slightly more",
        "No relationship — distance is not a fraud signal"
    ]
})
st.dataframe(test_data, use_container_width=True, hide_index=True)

# top merchants table
st.markdown('<p class="section-header">Top 15 Merchants — Revenue & Fraud Rate</p>',
            unsafe_allow_html=True)

display_df = merchant_summary.head(15)[
    ["merchant", "category", "transactions",
     "total_revenue", "revenue_share", "fraud_rate"]
].copy()
display_df["merchant"] = (display_df["merchant"].astype(str)
                          .str.replace("fraud_", "", regex=False))
display_df["total_revenue"] = display_df["total_revenue"].apply(
    lambda x: f"${x:,.0f}")
display_df["revenue_share"] = display_df["revenue_share"].apply(
    lambda x: f"{x:.3f}%")
display_df["fraud_rate"] = display_df["fraud_rate"].apply(
    lambda x: f"{x:.2f}%")
display_df.columns = ["Merchant", "Category", "Transactions",
                      "Revenue", "Share", "Fraud Rate"]

st.dataframe(display_df, use_container_width=True, hide_index=True)

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("""
    <div class="insight-box violet">
        <strong>42 of 51 states drive 80% of revenue</strong> — far more distributed
        than classic 80/20. Texas leads at 7.46%, but no single state dominates.
        Top 3 states combined: 20.37%.
    </div>""", unsafe_allow_html=True)

with col_i2:
    st.markdown("""
    <div class="insight-box crimson">
        <strong>Transaction distance has zero predictive power for fraud</strong> —
        Spearman rho = 0.034, p = 0.370. Fraud rate is uniform across all
        distance quartiles (0.57%–0.59%).
    </div>""", unsafe_allow_html=True)