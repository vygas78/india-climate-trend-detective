from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="EarthTrend Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent

VAR_LABELS = {
    "T2M": "Temperature (T2M)",
    "PRECTOTCORR": "Precipitation (PRECTOTCORR)",
}
VAR_COLORS = {"T2M": "#E8590C", "PRECTOTCORR": "#1C7ED6"}
VAR_SOFT = {"T2M": "#F7C6A6", "PRECTOTCORR": "#A9D0F5"}
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#E5EAF1"
UNITS_NOTE = "units not given in CSV"

REQUIRED_YEARLY = ["state", "variable", "year", "value"]
REQUIRED_TRENDS = ["state", "variable", "change_per_year", "p_value", "significant"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.hero {
  background: linear-gradient(120deg, #0B1F3A 0%, #12497A 55%, #1B7F8C 100%);
  border-radius: 20px; padding: 30px 34px; color: #fff; margin-bottom: 22px;
  box-shadow: 0 10px 30px rgba(11,31,58,.18); position: relative; overflow: hidden;
}
.hero:after {
  content: ""; position: absolute; right: -60px; top: -80px; width: 280px; height: 280px;
  border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.16), transparent 70%);
}
.eyebrow { font-size: 11.5px; letter-spacing: .14em; font-weight: 600; opacity: .75; margin-bottom: 8px; }
.hero h1 { margin: 0 0 6px 0; font-size: 34px; font-weight: 800; letter-spacing: -.02em; color: #fff; padding: 0; }
.hero p { margin: 0; opacity: .85; font-size: 15px; max-width: 760px; line-height: 1.5; }
.pills { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
  padding: 5px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 500;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #DDE4EE; }
.stTabs [data-baseweb="tab"] {
  height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0; font-weight: 600; color: #475569;
}
.stTabs [aria-selected="true"] { color: #12497A; }

.card {
  background: #fff; border: 1px solid #E3E9F2; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.kpi { background:#fff; border:1px solid #E3E9F2; border-radius:16px; padding:16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); border-top: 4px solid var(--accent, #12497A); height: 100%; }
.kpi .lbl { font-size: 12px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: .05em; }
.kpi .val { font-size: 30px; font-weight: 800; color: #0F172A; letter-spacing: -.02em; margin: 4px 0 2px 0; line-height: 1.1; }
.kpi .sub { font-size: 12.5px; color: #64748B; }

.section-title { font-size: 17px; font-weight: 700; color: #0F172A; margin: 4px 0 2px 0; }
.section-sub { font-size: 13px; color: #64748B; margin-bottom: 10px; }

table.stats { width:100%; border-collapse: collapse; font-size: 13.5px; }
table.stats th { text-align:left; color:#64748B; font-weight:600; font-size:11.5px; text-transform:uppercase;
  letter-spacing:.05em; padding: 6px 8px; border-bottom: 1px solid #E3E9F2; }
table.stats td { padding: 8px; border-bottom: 1px solid #F0F3F8; color:#0F172A; }
table.stats td.grp { font-weight:700; color:#12497A; white-space: nowrap; }
table.stats td.num { text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }

.verdict { display:flex; align-items:center; gap:16px; border-radius:16px; padding:16px 20px;
  border:1px solid var(--vb); background: var(--vbg); }
.verdict .vicon { width:46px; height:46px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:20px; color:#fff; background: var(--vc); flex-shrink:0; }
.verdict .vtitle { font-size:20px; font-weight:800; color: var(--vc); letter-spacing:-.01em; }
.verdict .vsub { font-size:13px; color:#475569; margin-top:2px; line-height:1.45; }
.verdict.up { --vc:#C2410C; --vbg:#FFF4EC; --vb:#F9D3B8; }
.verdict.down { --vc:#1D5FA8; --vbg:#EEF5FF; --vb:#C9DDF7; }
.verdict.weak { --vc:#92400E; --vbg:#FEF8E3; --vb:#F5DFA0; }
.verdict.none { --vc:#475569; --vbg:#F1F4F8; --vb:#DCE3EC; }

.notice {
  background: #FFF8E6; border: 1px solid #F5DFA0; border-radius: 12px; padding: 12px 14px;
  font-size: 13px; color: #6B4E00; line-height: 1.5;
}
.notice.blue { background:#EEF5FF; border-color:#C9DDF7; color:#1E3A5F; }

section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #E3E9F2; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def fmt_signed(v: float, d: int = 4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def fmt_num(v: float, d: int = 4) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def fmt_p(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return "<0.001" if v < 0.001 else f"{v:.4f}"


def var_label(v: str) -> str:
    return VAR_LABELS.get(v, v)


def kpi(label: str, value: str, sub: str = "", accent: str = "#12497A") -> str:
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="lbl">{label}</div><div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    truthy = {"true", "yes", "y", "1", "significant", "sig", "t"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def read_source(upload, filename: str):
    if upload is not None:
        return pd.read_csv(upload), f"uploaded {upload.name}"
    path = APP_DIR / filename
    if path.exists():
        return pd.read_csv(path), f"bundled {filename}"
    return None, f"{filename} not found"


def check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"**{name}** is missing column(s): {', '.join(missing)}. "
            f"Found: {', '.join(df.columns)}."
        )
        st.stop()


def analyse(years, values):
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return None
    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, n - 2)
    tau, tau_p = stats.kendalltau(x, y)
    sen = stats.theilslopes(y, x, 0.95)
    return {
        "n": n,
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "ci_low": float(lr.slope - tcrit * lr.stderr),
        "ci_high": float(lr.slope + tcrit * lr.stderr),
        "p": float(lr.pvalue),
        "r2": float(lr.rvalue**2),
        "tau": float(tau),
        "tau_p": float(tau_p),
        "sen": float(sen.slope),
        "sen_intercept": float(sen.intercept),
        "sen_low": float(sen.low_slope),
        "sen_high": float(sen.high_slope),
    }


def trend_verdict(res: dict, alpha: float = 0.05):
    p_ok = res["p"] < alpha
    t_ok = res["tau_p"] < alpha
    same_dir = np.sign(res["slope"]) == np.sign(res["tau"])
    if p_ok and t_ok and same_dir:
        if res["slope"] > 0:
            return "Increasing trend", "up", "▲"
        return "Decreasing trend", "down", "▼"
    if p_ok or t_ok:
        return "Weak or uncertain trend", "weak", "◆"
    return "No clear trend", "none", "●"


def base_layout(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


with st.sidebar:
    st.markdown("### 📂 Data")
    up_yearly = st.file_uploader("yearly_series.csv (optional)", type="csv", key="up_y")
    up_trends = st.file_uploader("state_trends.csv (optional)", type="csv", key="up_t")
    st.caption("Leave empty to use the CSV files bundled with the app.")
    st.markdown("---")
    st.markdown("### ⚠️ Read this first")
    st.markdown(
        '<div class="notice">Regression, Kendall tau and Sen\'s slope treat each yearly '
        "value as independent. They do not account for serial correlation or seasonality, "
        "so p-values can look more convincing than they are.<br><br>"
        "The CSVs give no units and do not say how values were aggregated over each state, "
        "so they are <b>not</b> described as state-wide averages.</div>",
        unsafe_allow_html=True,
    )

yearly_raw, yearly_src = read_source(up_yearly, "yearly_series.csv")
trends_raw, trends_src = read_source(up_trends, "state_trends.csv")

if yearly_raw is None:
    st.error("yearly_series.csv was not found next to the app file. Upload it in the sidebar.")
    st.stop()
check_columns(yearly_raw, REQUIRED_YEARLY, "yearly_series.csv")

yearly = yearly_raw.copy()
yearly["state"] = yearly["state"].astype(str).str.strip()
yearly["variable"] = yearly["variable"].astype(str).str.strip()
yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
yearly["value"] = pd.to_numeric(yearly["value"], errors="coerce")
yearly = yearly.dropna(subset=["year"]).astype({"year": int})

trends = None
if trends_raw is not None:
    check_columns(trends_raw, REQUIRED_TRENDS, "state_trends.csv")
    trends = trends_raw.copy()
    trends["state"] = trends["state"].astype(str).str.strip()
    trends["variable"] = trends["variable"].astype(str).str.strip()
    trends["change_per_year"] = pd.to_numeric(trends["change_per_year"], errors="coerce")
    trends["p_value"] = pd.to_numeric(trends["p_value"], errors="coerce")
    trends["significant"] = to_bool(trends["significant"])
    if "trend" in trends.columns:
        trends["trend"] = trends["trend"].astype(str).str.strip().str.capitalize()
    else:
        trends["trend"] = np.where(
            ~trends["significant"],
            "No significant trend",
            np.where(trends["change_per_year"] > 0, "Increasing", "Decreasing"),
        )

with st.sidebar:
    st.markdown("---")
    st.caption(f"Yearly data: {yearly_src}")
    st.caption(f"Summary data: {trends_src}")

states = sorted(yearly["state"].unique())
variables = [v for v in VAR_LABELS if v in set(yearly["variable"])] or sorted(yearly["variable"].unique())
y_min, y_max = int(yearly["year"].min()), int(yearly["year"].max())

st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">NASA HACKATHON &nbsp;·&nbsp; EARTH SCIENCE DATA</div>
  <h1>EarthTrend Explorer</h1>
  <p>Explore how temperature and precipitation have changed across Indian states,
  from yearly observations to a state-by-state comparison.</p>
  <div class="pills">
    <span class="pill">🗺️ {len(states)} states</span>
    <span class="pill">📈 {len(variables)} variables</span>
    <span class="pill">📅 {y_min}–{y_max}</span>
    <span class="pill">🧮 {len(yearly):,} yearly records</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📈 Year-wise State Analysis", "🗺️ All-State Trend Overview"])

with tab1:
    c1, c2, c3 = st.columns([1.4, 1.6, 2.2])
    with c1:
        state = st.selectbox("State", states, index=states.index("Goa") if "Goa" in states else 0)
    with c2:
        variable = st.radio("Variable", variables, format_func=var_label, horizontal=True)
    with c3:
        year_range = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

    start, end = year_range
    sub = yearly[
        (yearly["state"] == state)
        & (yearly["variable"] == variable)
        & (yearly["year"].between(start, end))
    ].sort_values("year")

    accent = VAR_COLORS.get(variable, "#12497A")
    res = analyse(sub["year"], sub["value"])

    if res is None:
        st.warning(
            f"Only {len(sub)} usable yearly value(s) for {state} in {start}-{end}. "
            "Select a range of at least 4 years to calculate trends."
        )
    else:
        span = end - start
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(
            kpi("Regression slope / yr", fmt_signed(res["slope"]), f"R² = {fmt_num(res['r2'], 3)}", accent),
            unsafe_allow_html=True,
        )
        k2.markdown(
            kpi(
                "Sen's slope / yr",
                fmt_signed(res["sen"]),
                f"95% CI {fmt_signed(res['sen_low'])} to {fmt_signed(res['sen_high'])}",
                accent,
            ),
            unsafe_allow_html=True,
        )
        k3.markdown(
            kpi("Kendall tau", fmt_signed(res["tau"], 3), f"p = {fmt_p(res['tau_p'])}", accent),
            unsafe_allow_html=True,
        )
        k4.markdown(
            kpi("Regression p-value", fmt_p(res["p"]), f"n = {res['n']} years", accent),
            unsafe_allow_html=True,
        )
        k5.markdown(
            kpi("Fitted change over range", fmt_signed(res["slope"] * span, 3), f"{start} → {end}", accent),
            unsafe_allow_html=True,
        )

        st.write("")
        v_title, v_kind, v_icon = trend_verdict(res)
        v_detail = {
            "up": "Both the regression and Kendall tests are below p = 0.05 and the slope is positive.",
            "down": "Both the regression and Kendall tests are below p = 0.05 and the slope is negative.",
            "weak": "Only one of the two tests is below p = 0.05, or they disagree in direction, so the evidence is mixed.",
            "none": "Neither test is below p = 0.05, so the yearly values do not show a statistically clear trend.",
        }[v_kind]
        st.markdown(
            f'<div class="verdict {v_kind}"><div class="vicon">{v_icon}</div><div>'
            f'<div class="vtitle">{v_title} · {state}, {var_label(variable)}, {start}–{end}</div>'
            f'<div class="vsub">{v_detail} Regression p = {fmt_p(res["p"])}, Kendall p = {fmt_p(res["tau_p"])}. '
            "Rule: both p &lt; 0.05 and same direction. Assumes independent years, so treat it as a guide.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        left, right = st.columns([1.75, 1])

        with left:
            st.markdown(
                f'<div class="section-title">{state} · {var_label(variable)}, {start}–{end}</div>'
                f'<div class="section-sub">Observed yearly values with fitted trend lines '
                f"(values: {UNITS_NOTE})</div>",
                unsafe_allow_html=True,
            )
            xs = np.array([sub["year"].min(), sub["year"].max()], dtype=float)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sub["year"],
                    y=sub["value"],
                    mode="lines+markers",
                    name="Observed",
                    line=dict(color=accent, width=2.5, shape="linear"),
                    marker=dict(size=8, color="white", line=dict(color=accent, width=2.2)),
                    hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=res["intercept"] + res["slope"] * xs,
                    mode="lines",
                    name=f"Linear regression ({fmt_signed(res['slope'])}/yr)",
                    line=dict(color=INK, width=2, dash="dash"),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=res["sen_intercept"] + res["sen"] * xs,
                    mode="lines",
                    name=f"Sen's slope ({fmt_signed(res['sen'])}/yr)",
                    line=dict(color=MUTED, width=2, dash="dot"),
                    hoverinfo="skip",
                )
            )
            base_layout(fig, 450)
            fig.update_xaxes(tickformat="d", dtick=2 if (end - start) > 8 else 1, title="Year")
            fig.update_yaxes(title=f"Value ({UNITS_NOTE})")
            st.plotly_chart(fig, config={"displayModeBar": False})

        with right:
            st.markdown('<div class="section-title">Calculated from the yearly values</div>', unsafe_allow_html=True)
            rows = [
                ("Linear regression", "Slope (per year)", fmt_signed(res["slope"])),
                ("", "95% CI of slope", f"{fmt_signed(res['ci_low'])} to {fmt_signed(res['ci_high'])}"),
                ("", "P-value", fmt_p(res["p"])),
                ("", "R-squared", fmt_num(res["r2"], 3)),
                ("Kendall tau", "Tau", fmt_signed(res["tau"], 3)),
                ("", "P-value", fmt_p(res["tau_p"])),
                ("Sen's slope", "Slope (per year)", fmt_signed(res["sen"])),
                ("", "95% CI", f"{fmt_signed(res['sen_low'])} to {fmt_signed(res['sen_high'])}"),
            ]
            body = "".join(
                f'<tr><td class="grp">{g}</td><td>{s}</td><td class="num">{v}</td></tr>' for g, s, v in rows
            )
            st.markdown(
                '<div class="card" style="margin-top:10px"><table class="stats">'
                "<tr><th>Method</th><th>Statistic</th><th style='text-align:right'>Value</th></tr>"
                f"{body}</table></div>",
                unsafe_allow_html=True,
            )
            st.caption("Slopes are in the CSV's (unspecified) units per year. p-values assume independent years.")

        with st.expander("📋 View / download the yearly values use
