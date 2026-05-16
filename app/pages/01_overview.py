import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
import os

st.set_page_config(page_title="Spending Trends", layout="wide")

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

df["month"]       = df["trans_date_trans_time"].dt.to_period("M")
df["quarter"]     = df["trans_date_trans_time"].dt.to_period("Q")
df["year"]        = df["trans_date_trans_time"].dt.year
df["hour"]        = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.day_name()

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

FILL_COLORS = {
    "#7c3aed": "rgba(124,58,237,0.1)",
    "#3b82f6": "rgba(59,130,246,0.1)",
    "#10b981": "rgba(16,185,129,0.1)",
    "#ef4444": "rgba(239,68,68,0.1)",
    "#f59e0b": "rgba(245,158,11,0.1)",
    "#a78bfa": "rgba(167,139,250,0.1)",
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
        hovermode="x unified"
    )

st.markdown("## Spending Trend Analysis")
st.markdown(
    '<p style="color:#94a3b8; font-size:0.9rem;">Monthly & quarterly revenue · '
    'STL decomposition · 3 hypothesis tests</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# aggregations
monthly = (
    df.groupby("month")["amt"]
    .agg(transactions="count", revenue="sum")
    .reset_index()
)
monthly["month_str"] = monthly["month"].astype(str)
monthly["revenue_m"] = monthly["revenue"] / 1e6

quarterly = (
    df.groupby("quarter")["amt"]
    .agg(transactions="count", revenue="sum")
    .reset_index()
)
quarterly["quarter_str"] = quarterly["quarter"].astype(str)
quarterly["revenue_m"]   = quarterly["revenue"] / 1e6

# monthly + quarterly charts
st.markdown('<p class="section-header">Monthly & Quarterly Revenue</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month_str"],
        y=monthly["revenue_m"],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=6, color=COLORS["primary"]),
        fill="tozeroy",
        fillcolor=FILL_COLORS[COLORS["primary"]],
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:.2f}M<extra></extra>"
    ))
    layout = base_layout(height=320, title="Monthly Revenue ($M)")
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    q_colors = [COLORS["primary"], COLORS["blue"], COLORS["emerald"],
                COLORS["amber"], COLORS["crimson"], COLORS["lavender"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quarterly["quarter_str"],
        y=quarterly["revenue_m"],
        marker_color=q_colors[:len(quarterly)],
        text=quarterly["revenue_m"].apply(lambda x: f"${x:.1f}M"),
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:.2f}M<extra></extra>"
    ))
    layout = base_layout(height=320, title="Quarterly Revenue ($M)")
    layout["yaxis"]["range"] = [0, quarterly["revenue_m"].max() * 1.2]
    layout["hovermode"] = "x"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# STL decomposition
st.markdown('<p class="section-header">STL Decomposition — Trend · Seasonality · Residual</p>',
            unsafe_allow_html=True)

monthly_stl = (
    df.groupby("month")["amt"].sum().reset_index()
)
monthly_stl["date"] = monthly_stl["month"].dt.to_timestamp()
monthly_stl = monthly_stl.set_index("date")["amt"]

stl    = STL(monthly_stl, period=6, robust=True)
result = stl.fit()

components = [
    ("Observed ($M)",  result.observed  / 1e6, COLORS["primary"]),
    ("Trend ($M)",     result.trend     / 1e6, COLORS["blue"]),
    ("Seasonal ($M)",  result.seasonal  / 1e6, COLORS["emerald"]),
    ("Residual ($M)",  result.resid     / 1e6, COLORS["crimson"]),
]

col3, col4 = st.columns(2)
cols_cycle = [col3, col4, col3, col4]

for i, (title, series, color) in enumerate(components):
    with cols_cycle[i]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index.strftime("%Y-%m"),
            y=series.values,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            fill="tozeroy",
            fillcolor=FILL_COLORS.get(color, "rgba(124,58,237,0.1)"),
            hovertemplate=f"<b>%{{x}}</b><br>{title}: $%{{y:.3f}}M<extra></extra>"
        ))
        fig.add_hline(y=0, line_color=COLORS["subtext"],
                      line_dash="dash", line_width=0.8)
        layout = base_layout(height=240, title=title)
        layout["xaxis"]["tickangle"] = 45
        layout["xaxis"]["tickfont"]  = dict(size=8)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

# category trend lines
st.markdown('<p class="section-header">Category Revenue Trends</p>',
            unsafe_allow_html=True)

cat_monthly = (
    df.groupby(["month", "category"], observed=True)["amt"]
    .sum().reset_index()
)
cat_monthly["month_str"] = cat_monthly["month"].astype(str)
cat_monthly["revenue_m"] = cat_monthly["amt"] / 1e6

all_cats = [str(c) for c in df["category"].cat.categories.tolist()]
selected_cats = st.multiselect(
    "Select categories",
    options=all_cats,
    default=all_cats[:6]
)

line_colors = [
    COLORS["primary"], COLORS["blue"], COLORS["emerald"],
    COLORS["amber"], COLORS["crimson"], COLORS["lavender"],
    "#f472b6", "#34d399", "#60a5fa", "#fbbf24",
    "#a78bfa", "#fb7185", "#86efac", "#93c5fd"
]

fig = go.Figure()
for i, cat in enumerate(selected_cats):
    subset = (cat_monthly[cat_monthly["category"].astype(str) == cat]
              .sort_values("month_str"))
    fig.add_trace(go.Scatter(
        x=subset["month_str"],
        y=subset["revenue_m"],
        mode="lines+markers",
        name=cat,
        line=dict(color=line_colors[i % len(line_colors)], width=2),
        marker=dict(size=4),
        hovertemplate=f"<b>{cat}</b><br>%{{x}}: $%{{y:.3f}}M<extra></extra>"
    ))

layout = base_layout(height=380, title="Monthly Revenue by Category ($M)")
layout["legend"] = dict(
    bgcolor="rgba(22,27,39,0.8)",
    bordercolor=COLORS["grid"],
    font=dict(size=10)
)
layout["xaxis"]["tickangle"] = 45
layout["xaxis"]["tickfont"]  = dict(size=9)
fig.update_layout(**layout)
st.plotly_chart(fig, use_container_width=True)

# hypothesis test results
st.markdown('<p class="section-header">Hypothesis Test Results</p>',
            unsafe_allow_html=True)

test_data = pd.DataFrame({
    "Test"       : ["Kruskal-Wallis", "Mann-Whitney U", "Chi-Square"],
    "Question"   : [
        "Do amounts differ across categories?",
        "Do weekend amounts differ from weekday?",
        "Is category associated with gender?"
    ],
    "Result"     : ["Reject H0", "Reject H0", "Reject H0"],
    "Statistic"  : ["H = 262,024", "Significant", "χ² significant"],
    "p-value"    : ["< 0.001", "< 0.05", "< 0.001"],
    "Effect Size": [
        "grocery_pos $105 vs travel $6",
        "Cohen's d = small",
        "Cramer's V = weak"
    ]
})

st.dataframe(test_data, use_container_width=True, hide_index=True)

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("""
    <div class="insight-box violet">
        <strong>December 2019 STL residual: $4.9M</strong> — the holiday spike was
        a structural anomaly, not seasonal. Seasonal component explains only
        ±$1.4M of variation across the full 18 months.
    </div>""", unsafe_allow_html=True)

with col_i2:
    st.markdown("""
    <div class="insight-box amber">
        <strong>YoY revenue declined 7.1%</strong> — Jan–Jun 2020 vs 2019.
        Q1 2020 dropped to $12.2M from $15.9M — COVID-19 demand shock
        beginning March 2020.
    </div>""", unsafe_allow_html=True)