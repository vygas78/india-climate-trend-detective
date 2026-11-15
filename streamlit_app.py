"""EarthTrend Explorer - redesigned Streamlit UI.

Reads the real datasets (no sample data):
  yearly_series.csv : state, variable, year, value
  state_trends.csv  : state, variable, trend, change_per_year, p_value, significant
Both are loaded from the same folder as this file, or uploaded in the sidebar.
"""
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

# --------------------------------------------------------------------------
# Constants / palette
# --------------------------------------------------------------------------
VAR_LABELS = {
    "T2M": "Temperature (T2M)",
    "PRECTOTCORR": "Precipitation (PRECTOTCORR)",
}
VAR_COLORS = {"T2M": "#E8590C", "PRECTOTCORR": "#1C7ED6"}
VAR_SOFT = {"T2M": "#F7C6A6", "PRECTOTCORR": "#A9D0F5"}
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#E5EAF1"
VAR_UNITS={
  "T2M": "°C",
  "PRECTOTCORR" : "mm/day"
}
REQUIRED_YEARLY = ["state", "variable", "year", "value"]
REQUIRED_TRENDS = ["state", "variable", "change_per_year", "p_value", "significant"]

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Hero */
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

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #DDE4EE; }
.stTabs [data-baseweb="tab"] {
  height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0; font-weight: 600; color: #475569;
}
.stTabs [aria-selected="true"] { color: #12497A; }

/* Cards */
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

/* Badges */
.badge { display:inline-block; padding: 4px 11px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }
.badge.good { background:#E6F6EC; color:#15803D; }
.badge.warn { background:#FEF3C7; color:#92400E; }
.badge.neutral { background:#EEF2F7; color:#475569; }

/* Stats table */
table.stats { width:100%; border-collapse: collapse; font-size: 13.5px; }
table.stats th { text-align:left; color:#64748B; font-weight:600; font-size:11.5px; text-transform:uppercase;
  letter-spacing:.05em; padding: 6px 8px; border-bottom: 1px solid #E3E9F2; }
table.stats td { padding: 8px; border-bottom: 1px solid #F0F3F8; color:#0F172A; }
table.stats td.grp { font-weight:700; color:#12497A; white-space: nowrap; }
table.stats td.num { text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }

/* Trend verdict banner */
.verdict { display:flex; align-items:center; gap:16px; border-radius:16px; padding:16px 20px;
  border:1px solid var(--vb); background: var(--vbg); }
.verdict .vicon { width:46px; height:46px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:20px; color:#fff; background: var(--vc); flex-shrink:0; }
.verdict .vtitle { font-size:20px; font-weight:800; color: var(--vc); letter-spacing:-.01em; }
.verdict .vsub { font-size:13px; color:#475569; margin-top:2px; line-height:1.45; }
.verdict.up   { --vc:#C2410C; --vbg:#FFF4EC; --vb:#F9D3B8; }
.verdict.down { --vc:#1D5FA8; --vbg:#EEF5FF; --vb:#C9DDF7; }
.verdict.weak { --vc:#92400E; --vbg:#FEF8E3; --vb:#F5DFA0; }
.verdict.none { --vc:#475569; --vbg:#F1F4F8; --vb:#DCE3EC; }

/* Notice */
.notice {
  background: #FFF8E6; border: 1px solid #F5DFA0; border-radius: 12px; padding: 12px 14px;
  font-size: 13px; color: #6B4E00; line-height: 1.5;
}
.notice.blue { background:#EEF5FF; border-color:#C9DDF7; color:#1E3A5F; }

section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #E3E9F2; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
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
    """Uploaded file wins; otherwise the file next to app.py."""
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
    """Regression, Kendall tau and Sen's slope on the selected yearly values."""
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
    """Standard rule: a trend is reported only if BOTH the regression and the
    Kendall tau p-values are below alpha and both point the same way."""
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


# --------------------------------------------------------------------------
# Sidebar: data sources + reading notes
# --------------------------------------------------------------------------
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
    st.error("yearly_series.csv was not found next to app.py. Upload it in the sidebar.")
    st.stop()
check_columns(yearly_raw, REQUIRED_YEARLY, "yearly_series.csv")

yearly = yearly_raw.copy()
yearly["state"] = yearly["state"].astype(str).str.strip()
yearly["variable"] = yearly["variable"].astype(str).str.strip()
yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
yearly["value"] = pd.to_numeric(yearly["value"], errors="coerce")
yearly = yearly.dropna(subset=["year", "value"]).copy()
yearly["year"] = yearly["year"].astype(int)

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
    else:  # derive a label from the supplied columns if the file has none
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

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
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

tab1, tab2 = st.tabs(["📈  Year-wise State Analysis", "🗺️  All-State Trend Overview"])

# ==========================================================================
# PAGE 1 - Year-wise State Analysis
# ==========================================================================
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
                f"(values: {VAR_UNITS.get(variable,'units not verified')})</div>",
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
            fig.update_yaxes(title=f"Value ({VAR_UNITS.get(ov_var,"units not verified")})")
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

        with st.expander("📋 View / download the yearly values used"):
            show = sub[["year", "value"]].rename(columns={"year": "Year", "value": "Value"})
            st.dataframe(show, hide_index=True)
            st.download_button(
                "Download as CSV",
                show.to_csv(index=False).encode("utf-8"),
                file_name=f"{state}_{variable}_{start}-{end}.csv".replace(" ", "_"),
                mime="text/csv",
            )

# ==========================================================================
# PAGE 2 - All-State Trend Overview
# ==========================================================================
with tab2:
    if trends is None:
        st.error("state_trends.csv was not found next to app.py. Upload it in the sidebar.")
    else:
        st.markdown(
            '<div class="notice blue"><b>About this page:</b> trend, rate of change, p-value and '
            "significance are the <b>pre-calculated values supplied in state_trends.csv</b>. "
            "They are not recalculated here, so they can differ from the statistics on the "
            "Year-wise page.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        o1, o2 = st.columns([2, 1.4])
        with o1:
            ov_var = st.radio(
                "Variable", variables, format_func=var_label, horizontal=True, key="ov_var"
            )
        with o2:
            only_sig = st.toggle("Show only significant states", value=False)

        d = trends[trends["variable"] == ov_var].dropna(subset=["change_per_year"]).copy()
        if only_sig:
            d = d[d["significant"]]

        if d.empty:
            st.warning("No states to show with the current filters.")
        else:
            acc = VAR_COLORS.get(ov_var, "#12497A")
            soft = VAR_SOFT.get(ov_var, "#CBD5E1")
            n_sig = int(d["significant"].sum())
            top = d.loc[d["change_per_year"].abs().idxmax()]

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(kpi("States shown", f"{len(d)}", var_label(ov_var), acc), unsafe_allow_html=True)
            k2.markdown(
                kpi("Marked significant", f"{n_sig}", "per supplied summary", acc), unsafe_allow_html=True
            )
            k3.markdown(
                kpi(
                    "Rising / falling",
                    f"{int((d['change_per_year'] > 0).sum())} / {int((d['change_per_year'] < 0).sum())}",
                    "by sign of change per year",
                    acc,
                ),
                unsafe_allow_html=True,
            )
            k4.markdown(
                kpi(
                    "Largest change",
                    fmt_signed(top["change_per_year"]),
                    f"{top['state']} (per year)",
                    acc,
                ),
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown(
                f'<div class="section-title">Rate of change per year by state · {var_label(ov_var)}</div>'
                f'<div class="section-sub">Solid bars are marked significant in the supplied summary; '
                f"faded bars are not. Values: {VAR_UNITS.get(ov_var,"units not verified")}.</div>",
                unsafe_allow_html=True,
            )
            ds = d.sort_values("change_per_year")
            fig2 = go.Figure()
            for flag, name, colr in [(True, "Significant (supplied)", acc), (False, "Not significant", soft)]:
                part = ds[ds["significant"] == flag]
                if part.empty:
                    continue
                fig2.add_trace(
                    go.Bar(
                        x=part["change_per_year"],
                        y=part["state"],
                        orientation="h",
                        name=name,
                        marker=dict(color=colr),
                        customdata=np.column_stack([part["p_value"].to_numpy(), part["trend"].to_numpy()]),
                        hovertemplate="<b>%{y}</b><br>Trend: %{customdata[1]}<br>Change/yr: %{x:+.4f}<br>p-value: %{customdata[0]:.4f}<extra></extra>",
                    )
                )
            base_layout(fig2, max(440, 24 * len(ds) + 90))
            fig2.update_layout(barmode="relative", bargap=0.25)
            fig2.update_yaxes(categoryorder="array", categoryarray=list(ds["state"]), gridcolor="rgba(0,0,0,0)")
            fig2.update_xaxes(title=f"Change per year ({VAR_UNITS.get(ov_var,"units not verified")})", gridcolor=GRID, zeroline=True,
                              zerolinecolor="#94A3B8", showgrid=True)
            st.plotly_chart(fig2, config={"displayModeBar": False})

            a, b = st.columns([1.2, 1])
            with a:
                st.markdown(
                    '<div class="section-title">Effect size vs p-value</div>'
                    '<div class="section-sub">Dashed line marks p = 0.05 (supplied p-values).</div>',
                    unsafe_allow_html=True,
                )
                fig3 = go.Figure()
                fig3.add_trace(
                    go.Scatter(
                        x=d["change_per_year"],
                        y=d["p_value"],
                        mode="markers",
                        text=d["state"],
                        marker=dict(
                            size=11,
                            color=[acc if s else soft for s in d["significant"]],
                            line=dict(color="white", width=1.2),
                        ),
                        hovertemplate="<b>%{text}</b><br>Change/yr: %{x:+.4f}<br>p-value: %{y:.4f}<extra></extra>",
                        showlegend=False,
                    )
                )
                fig3.add_hline(y=0.05, line=dict(color=MUTED, dash="dash", width=1.5))
                base_layout(fig3, 400)
                fig3.update_xaxes(title=f"Change per year ({VAR_UNITS.get(ov_var,"units not verified")})", showgrid=True, gridcolor=GRID)
                fig3.update_yaxes(title="p-value (supplied)")
                st.plotly_chart(fig3, config={"displayModeBar": False})

            with b:
                st.markdown(
                    '<div class="section-title">State table</div>'
                    '<div class="section-sub">Sorted by size of change.</div>',
                    unsafe_allow_html=True,
                )
                cols = ["state", "trend", "change_per_year", "p_value", "significant"]
                tbl = d.sort_values("change_per_year", ascending=False)[cols].rename(
                    columns={
                        "state": "State",
                        "trend": "Trend",
                        "change_per_year": "Change / yr",
                        "p_value": "p-value",
                        "significant": "Significant",
                    }
                )
                st.dataframe(
                    tbl,
                    hide_index=True,
                    height=400,
                    column_config={
                        "Change / yr": st.column_config.NumberColumn(format="%+.4f"),
                        "p-value": st.column_config.NumberColumn(format="%.4f"),
                    },
                )

        # ---- Temperature vs precipitation comparison -----------------------
        if {"T2M", "PRECTOTCORR"}.issubset(set(trends["variable"])):
            st.write("")
            st.markdown(
                '<div class="section-title">Temperature vs precipitation, state by state</div>'
                '<div class="section-sub">Each dot is a state. The two axes use different (unspecified) '
                "units, so compare direction and ranking, not magnitude.</div>",
                unsafe_allow_html=True,
            )
            piv = trends.pivot_table(
                index="state", columns="variable", values=["change_per_year", "significant"], aggfunc="first"
            )
            piv.columns = [f"{a}|{b}" for a, b in piv.columns]
            piv = piv.dropna(subset=["change_per_year|T2M", "change_per_year|PRECTOTCORR"]).reset_index()
            sig_t = piv["significant|T2M"].astype(bool)
            sig_p = piv["significant|PRECTOTCORR"].astype(bool)
            piv["sig_count"] = sig_t.astype(int) + sig_p.astype(int)
            palette = {0: "#CBD5E1", 1: "#12497A", 2: "#E8590C"}
            names = {0: "Neither significant", 1: "One significant", 2: "Both significant"}
            fig4 = go.Figure()
            for k in (0, 1, 2):
                part = piv[piv["sig_count"] == k]
                if part.empty:
                    continue
                fig4.add_trace(
                    go.Scatter(
                        x=part["change_per_year|T2M"],
                        y=part["change_per_year|PRECTOTCORR"],
                        mode="markers",
                        name=names[k],
                        text=part["state"],
                        marker=dict(size=12, color=palette[k], line=dict(color="white", width=1.2)),
                        hovertemplate="<b>%{text}</b><br>Temp change/yr: %{x:+.4f}<br>Precip change/yr: %{y:+.4f}<extra></extra>",
                    )
                )
            base_layout(fig4, 430)
            fig4.add_vline(x=0, line=dict(color="#94A3B8", width=1))
            fig4.add_hline(y=0, line=dict(color="#94A3B8", width=1))
            fig4.update_xaxes(title="Temperature change per year (T2M)", showgrid=True, gridcolor=GRID)
            fig4.update_yaxes(title="Precipitation change per year (PRECTOTCORR)")
            st.plotly_chart(fig4, config={"displayModeBar": False})
