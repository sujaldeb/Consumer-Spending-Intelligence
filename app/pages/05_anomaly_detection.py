import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import STL
import os

st.set_page_config(page_title="Behaviour Shift", layout="wide")

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

df["month"] = df["trans_date_trans_time"].dt.to_period("M")

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
        hovermode="x unified"
    )

st.markdown("## Behaviour Shift Detection")
st.markdown(
    '<p style="color:#94a3b8; font-size:0.9rem;">CUSUM control chart · '
    'Pettitt permutation test · category-level Cohen\'s d · February 2020 changepoint</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# monthly revenue series
monthly = (
    df.groupby("month")["amt"]
    .sum().reset_index()
)
monthly["month_str"] = monthly["month"].astype(str)
monthly["revenue_m"] = monthly["amt"] / 1e6
revenue = monthly["revenue_m"].values

# CUSUM
mean          = revenue.mean()
std           = revenue.std()
cusum         = np.cumsum(revenue - mean)
control_limit = 5 * std

# Pettitt
def pettitt_statistic(x):
    n = len(x)
    U = np.zeros(n)
    for t in range(1, n):
        for i in range(t):
            U[t] += np.sign(x[t] - x[i])
    return np.max(np.abs(U)), np.argmax(np.abs(U))

K_obs, cp_idx = pettitt_statistic(revenue)

rng    = np.random.default_rng(42)
K_null = np.array([
    pettitt_statistic(rng.permutation(revenue))[0]
    for _ in range(5000)
])
p_val = (K_null >= K_obs).mean()

pre  = revenue[:cp_idx]
post = revenue[cp_idx:]

def cohens_d(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled_std

d = cohens_d(pre, post)
cp_label = monthly["month_str"].iloc[cp_idx]

# KPI cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Changepoint Month</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#ef4444;flex-shrink:0;"></div>
            <div style="font-size:1.4rem;font-weight:700;color:#ffffff;">
                {cp_label}
            </div>
        </div>
        <div class="kpi-sub">Pettitt test identified</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Cohen's d Effect Size</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#7c3aed;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">
                {d:.3f}
            </div>
        </div>
        <div class="kpi-sub">Medium practical effect</div>
    </div>""", unsafe_allow_html=True)

with col3:
    monthly_decline = post.mean() - pre.mean()
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Monthly Revenue Change</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#f59e0b;flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">
                ${monthly_decline:.2f}M
            </div>
        </div>
        <div class="kpi-sub">Pre ${pre.mean():.2f}M to Post ${post.mean():.2f}M</div>
    </div>""", unsafe_allow_html=True)

with col4:
    breach       = "Yes" if np.any(np.abs(cusum) > control_limit) else "No"
    breach_color = "#ef4444" if breach == "Yes" else "#10b981"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">CUSUM Breach</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:8px;margin:8px 0;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{breach_color};flex-shrink:0;"></div>
            <div style="font-size:1.75rem;font-weight:700;color:#ffffff;">
                {breach}
            </div>
        </div>
        <div class="kpi-sub">Control limit: +/-${control_limit:.2f}M</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# CUSUM + changepoint charts
st.markdown('<p class="section-header">CUSUM Control Chart & Changepoint</p>',
            unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month_str"],
        y=monthly["revenue_m"],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=5, color=COLORS["primary"]),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.1)",
        name="Revenue",
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:.2f}M<extra></extra>"
    ))
    fig.add_shape(
        type="line",
        x0=cp_label, x1=cp_label,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=COLORS["crimson"], dash="dash", width=2)
    )
    fig.add_annotation(
        x=cp_label, y=0.98,
        xref="x", yref="paper",
        text=f"Changepoint: {cp_label}",
        showarrow=False,
        font=dict(color=COLORS["crimson"], size=10),
        yanchor="top", xanchor="left"
    )
    fig.add_shape(
        type="line",
        x0=monthly["month_str"].iloc[0],
        x1=monthly["month_str"].iloc[len(pre)-1],
        y0=pre.mean(), y1=pre.mean(),
        xref="x", yref="y",
        line=dict(color=COLORS["amber"], dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=monthly["month_str"].iloc[0], y=pre.mean(),
        xref="x", yref="y",
        text=f"Pre mean: ${pre.mean():.2f}M",
        showarrow=False,
        font=dict(color=COLORS["amber"], size=9),
        yanchor="bottom", xanchor="left"
    )
    fig.add_shape(
        type="line",
        x0=cp_label,
        x1=monthly["month_str"].iloc[-1],
        y0=post.mean(), y1=post.mean(),
        xref="x", yref="y",
        line=dict(color=COLORS["emerald"], dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=cp_label, y=post.mean(),
        xref="x", yref="y",
        text=f"Post mean: ${post.mean():.2f}M",
        showarrow=False,
        font=dict(color=COLORS["emerald"], size=9),
        yanchor="bottom", xanchor="left"
    )
    layout = base_layout(height=360,
                         title="Monthly Revenue with Pettitt Changepoint")
    layout["xaxis"]["tickangle"] = 45
    layout["xaxis"]["tickfont"]  = dict(size=8)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month_str"],
        y=cusum,
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2.5),
        marker=dict(size=5, color=COLORS["blue"]),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.1)",
        name="CUSUM",
        hovertemplate="<b>%{x}</b><br>CUSUM: $%{y:.3f}M<extra></extra>"
    ))
    fig.add_shape(
        type="line",
        x0=monthly["month_str"].iloc[0],
        x1=monthly["month_str"].iloc[-1],
        y0=control_limit, y1=control_limit,
        xref="x", yref="y",
        line=dict(color=COLORS["crimson"], dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=monthly["month_str"].iloc[0], y=control_limit,
        xref="x", yref="y",
        text=f"UCL: ${control_limit:.2f}M",
        showarrow=False,
        font=dict(color=COLORS["crimson"], size=9),
        yanchor="bottom", xanchor="left"
    )
    fig.add_shape(
        type="line",
        x0=monthly["month_str"].iloc[0],
        x1=monthly["month_str"].iloc[-1],
        y0=-control_limit, y1=-control_limit,
        xref="x", yref="y",
        line=dict(color=COLORS["crimson"], dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=monthly["month_str"].iloc[0], y=-control_limit,
        xref="x", yref="y",
        text=f"LCL: -${control_limit:.2f}M",
        showarrow=False,
        font=dict(color=COLORS["crimson"], size=9),
        yanchor="top", xanchor="left"
    )
    fig.add_shape(
        type="line",
        x0=monthly["month_str"].iloc[0],
        x1=monthly["month_str"].iloc[-1],
        y0=0, y1=0,
        xref="x", yref="y",
        line=dict(color=COLORS["subtext"], width=0.8, dash="dot")
    )
    layout = base_layout(height=360,
                         title="CUSUM Control Chart — Cumulative Deviation")
    layout["xaxis"]["tickangle"] = 45
    layout["xaxis"]["tickfont"]  = dict(size=8)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# category shift analysis
st.markdown('<p class="section-header">Category-Level Shift — Cohen\'s d & Revenue Change</p>',
            unsafe_allow_html=True)

cat_monthly = (
    df.groupby(["month", "category"], observed=True)["amt"]
    .sum().reset_index()
)
cat_monthly["month_str"] = cat_monthly["month"].astype(str)
cat_monthly["revenue_m"] = cat_monthly["amt"] / 1e6

categories = df["category"].cat.categories.tolist()

results = []
for cat in categories:
    subset = (cat_monthly[cat_monthly["category"] == cat]
              .sort_values("month_str")["revenue_m"].values)
    if len(subset) < 6:
        continue
    _, cp = pettitt_statistic(subset)
    pre_c  = subset[:cp]
    post_c = subset[cp:]
    if len(pre_c) < 2 or len(post_c) < 2:
        continue
    d_cat   = cohens_d(pre_c, post_c)
    pct_chg = (post_c.mean() - pre_c.mean()) / pre_c.mean() * 100
    results.append({
        "category"  : str(cat),
        "cohens_d"  : round(d_cat, 4),
        "pct_change": round(pct_chg, 2),
    })

results_df = pd.DataFrame(results).sort_values("cohens_d")

col7, col8 = st.columns(2)

with col7:
    bar_colors = [COLORS["crimson"] if d < 0 else COLORS["emerald"]
                  for d in results_df["cohens_d"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=results_df["category"],
        x=results_df["cohens_d"],
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>Cohen's d: %{x:.4f}<extra></extra>"
    ))
    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=COLORS["amber"], dash="dash", width=1)
    )
    fig.add_annotation(
        x=0.5, y=1,
        xref="x", yref="paper",
        text="Medium (0.5)",
        showarrow=False,
        font=dict(color=COLORS["amber"], size=9),
        yanchor="bottom"
    )
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=COLORS["subtext"], width=0.8, dash="dot")
    )
    layout = base_layout(height=420,
                         title="Cohen's d by Category — Pre vs Post Changepoint")
    layout["xaxis"]["title"] = "Cohen's d"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

with col8:
    pct_colors = [COLORS["crimson"] if p < 0 else COLORS["emerald"]
                  for p in results_df["pct_change"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=results_df["category"],
        x=results_df["pct_change"],
        orientation="h",
        marker_color=pct_colors,
        hovertemplate="<b>%{y}</b><br>Revenue change: %{x:.2f}%<extra></extra>"
    ))
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=COLORS["subtext"], width=0.8, dash="dot")
    )
    layout = base_layout(height=420,
                         title="Revenue % Change — Pre vs Post Changepoint")
    layout["xaxis"]["title"]      = "Revenue Change (%)"
    layout["xaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

# results table
st.markdown('<p class="section-header">Category Shift Summary</p>',
            unsafe_allow_html=True)

display_results = results_df.sort_values("cohens_d", ascending=False).copy()
display_results["Effect Size"] = display_results["cohens_d"].apply(
    lambda x: "Large" if abs(x) >= 0.8
    else "Medium" if abs(x) >= 0.5
    else "Small"
)
display_results["pct_change"] = display_results["pct_change"].apply(
    lambda x: f"{x:.2f}%"
)
display_results.columns = ["Category", "Cohen's d", "Revenue Change", "Effect Size"]
st.dataframe(display_results, use_container_width=True, hide_index=True)

# hypothesis test summary
st.markdown('<p class="section-header">Statistical Test Summary</p>',
            unsafe_allow_html=True)

test_data = pd.DataFrame({
    "Test"          : ["CUSUM", "Pettitt (permutation)"],
    "Question"      : [
        "Did revenue breach control limits?",
        "Does a structural changepoint exist?"
    ],
    "Result"        : ["No breach", "Fail to reject H0"],
    "Key Stat"      : [
        f"Max CUSUM: ${max(abs(cusum)):.3f}M vs limit ${control_limit:.3f}M",
        f"K = {K_obs:.2f}, p = {p_val:.4f}"
    ],
    "Effect Size"   : [
        "CUSUM range within bounds",
        f"Cohen's d = {d:.4f} (medium)"
    ],
    "Interpretation": [
        "Series volatile but never structurally out of control",
        "18 months insufficient power — effect real, not significant"
    ]
})
st.dataframe(test_data, use_container_width=True, hide_index=True)

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("""
    <div class="insight-box crimson">
        <strong>All 14 categories declined post-February 2020</strong> — the COVID
        shock was universal. Travel most impacted at Cohen's d = 0.928 and
        -24.1% revenue. shopping_pos most resilient at -11.9%.
    </div>""", unsafe_allow_html=True)

with col_i2:
    st.markdown("""
    <div class="insight-box amber">
        <strong>Cohen's d = 0.632 — medium practical effect</strong> despite failing
        statistical significance. Pre-changepoint mean $5.28M vs post $4.51M —
        a $0.77M monthly revenue decline beginning February 2020.
    </div>""", unsafe_allow_html=True)