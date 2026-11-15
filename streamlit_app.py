# -*- coding: utf-8 -*-
"""
EarthTrend Explorer - web version (Streamlit)

Same data, same statistics and same charts as the Windows desktop app:
this file reuses the functions in earthtrend_explorer.py, so the numbers match.

Run locally:   py -m streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

import earthtrend_explorer as core  # Tkinter is optional there, so this works on a server

APP_DIR = Path(__file__).resolve().parent
YEARLY_PATH = APP_DIR / "yearly_series.csv"
TRENDS_PATH = APP_DIR / "state_trends.csv"

st.set_page_config(page_title="EarthTrend Explorer", page_icon="\U0001F30D",
                   layout="wide")


# ---------------------------------------------------------------- data ----
@st.cache_data(show_spinner=False)
def default_yearly(path):
    return core.load_yearly_csv(path)


@st.cache_data(show_spinner=False)
def default_trends(path):
    return core.load_trends_csv(path)


def load_data():
    """Return (yearly, trends); uploaded files win over the bundled ones."""
    st.sidebar.header("Data")
    up_y = st.sidebar.file_uploader("Use your own yearly_series.csv (optional)",
                                    type="csv", key="up_yearly")
    up_t = st.sidebar.file_uploader("Use your own state_trends.csv (optional)",
                                    type="csv", key="up_trends")
    yearly = trends = None
    try:
        if up_y is not None:
            yearly = core.load_yearly_csv(up_y)
        elif YEARLY_PATH.exists():
            yearly = default_yearly(str(YEARLY_PATH))
    except core.DataError as exc:
        st.sidebar.error(str(exc))
    try:
        if up_t is not None:
            trends = core.load_trends_csv(up_t)
        elif TRENDS_PATH.exists():
            trends = default_trends(str(TRENDS_PATH))
    except core.DataError as exc:
        st.sidebar.error(str(exc))

    st.sidebar.divider()
    st.sidebar.subheader("Read this first")
    st.sidebar.caption(
        "Regression, Kendall tau and Sen's slope treat each yearly value as "
        "independent. They do not account for serial correlation or seasonality, "
        "so p-values can look more convincing than they are.")
    st.sidebar.caption(
        "The CSV files give no units and do not say how values were aggregated "
        "over each state, so they are not described as state-wide averages.")
    return yearly, trends


# -------------------------------------------------------------- page 1 ----
def verdict(res):
    lr_sig = res["lr_p"] < core.ALPHA
    mk_sig = res["tau_p"] < core.ALPHA
    sen_sig = res["sen_low"] > 0 or res["sen_high"] < 0
    count = int(lr_sig) + int(mk_sig) + int(sen_sig)
    direction = ("rising" if res["sen_slope"] > 0
                 else "falling" if res["sen_slope"] < 0 else "flat")
    if count == 3:
        st.success("All three methods flag a trend at the 5% level "
                   f"(fitted direction: {direction}). Worth investigating, not proof.")
    elif count == 0:
        st.info("None of the three methods shows a clear trend at the 5% level. "
                "The year-to-year changes may be natural variability.")
    else:
        st.warning(f"The methods disagree ({count} of 3 flag a trend), "
                   "so this trend is uncertain.")


def page_series(yearly, trends):
    if yearly is None:
        st.error("yearly_series.csv was not found. Upload it in the sidebar.")
        return
    states = sorted(yearly["state"].unique())
    variables = core.order_variables(yearly["variable"].unique())
    years = [int(y) for y in sorted(yearly["year"].unique())]

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        state = st.selectbox("State", states, key="s_state")
    with c2:
        var = st.selectbox("Variable", variables, key="s_var",
                           format_func=lambda v: core.VAR_LABELS.get(v, v))
    with c3:
        y0 = st.selectbox("From year", years, index=0, key="s_y0")
    with c4:
        y1 = st.selectbox("To year", years, index=len(years) - 1, key="s_y1")

    try:
        res = core.analyse_series(yearly, state, var, int(y0), int(y1))
    except core.DataError as exc:
        st.error(str(exc))
        return

    label = core.VAR_LABELS.get(var, var)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Regression slope / year", core.fmt_num(res["slope"], signed=True))
    with m2:
        st.metric("Sen's slope / year", core.fmt_num(res["sen_slope"], signed=True))
    with m3:
        st.metric("Kendall tau", core.fmt_num(res["tau"], 3, signed=True))
    with m4:
        st.metric("Regression p-value", core.fmt_p(res["lr_p"]))

    left, right = st.columns([3, 2])
    with left:
        fig = Figure(figsize=(9, 5.2), dpi=110)
        core.draw_series_plot(fig, res, label)
        st.pyplot(fig)
    with right:
        st.subheader("Calculated from the yearly values")
        rows = [
            ("Linear regression", "Slope (per year)", core.fmt_num(res["slope"], signed=True)),
            ("", "95% CI of slope",
             f"{core.fmt_num(res['slope_ci'][0], signed=True)} to "
             f"{core.fmt_num(res['slope_ci'][1], signed=True)}"),
            ("", "P-value", core.fmt_p(res["lr_p"])),
            ("", "R-squared", core.fmt_num(res["r2"], 3)),
            ("Kendall tau", "Tau", core.fmt_num(res["tau"], 3, signed=True)),
            ("", "P-value", core.fmt_p(res["tau_p"])),
            ("Sen's slope", "Slope (per year)", core.fmt_num(res["sen_slope"], signed=True)),
            ("", "95% confidence interval",
             f"{core.fmt_num(res['sen_low'], signed=True)} to "
             f"{core.fmt_num(res['sen_high'], signed=True)}"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Method", "Statistic", "Value"]),
                     hide_index=True)
        verdict(res)
        st.caption(f"{res['n']} yearly values, {res['start']}–{res['end']}. "
                   f"Residual lag-1 autocorrelation: "
                   f"{core.fmt_num(res['lag1'], 2, signed=True)} "
                   "(rough check of year-to-year correlation).")

    st.subheader("Supplied summary (state_trends.csv)")
    sup = None
    if trends is not None:
        m = trends[(trends["state"] == state) & (trends["variable"] == var)]
        if not m.empty:
            sup = m.iloc[0]
    if sup is None:
        st.caption("No matching row in the trends file.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Trend label", str(sup["trend"]))
        with s2:
            st.metric("Change per year", core.fmt_num(sup["change_per_year"], signed=True))
        with s3:
            st.metric("P-value", core.fmt_p(sup["p_value"]))
        with s4:
            st.metric("Marked significant", "Yes" if sup["significant"] else "No")
        st.caption("Pre-calculated values. The method and year range are not "
                   "documented in the CSV, so they can differ from the numbers "
                   "calculated above.")


# -------------------------------------------------------------- page 2 ----
def page_overview(trends):
    if trends is None:
        st.error("state_trends.csv was not found. Upload it in the sidebar.")
        return
    all_vars = core.order_variables(trends["variable"].unique())

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        choice = st.selectbox("Variable", [core.BOTH_LABEL] + all_vars, key="o_var",
                              format_func=lambda v: v if v == core.BOTH_LABEL
                              else core.VAR_LABELS.get(v, v))
    with c2:
        sort_mode = st.selectbox("Sort by", core.SORT_OPTIONS, key="o_sort")
    with c3:
        only_sig = st.checkbox("Show only significant (as supplied)", key="o_sig")

    variables = all_vars if choice == core.BOTH_LABEL else [choice]

    cols = st.columns(len(variables))
    for col, v in zip(cols, variables):
        d = trends[trends["variable"] == v]
        sig = d[d["significant"]]
        with col:
            st.metric(core.VAR_LABELS.get(v, v),
                      f"{len(sig)} of {len(d)} significant")
            st.caption(f"{int((sig['change_per_year'] > 0).sum())} increasing, "
                       f"{int((sig['change_per_year'] < 0).sum())} decreasing "
                       "(as supplied)")

    data = core.filter_trends(trends, variables, only_sig, sort_mode)
    fig = Figure(figsize=(13, 9) if len(variables) > 1 else (8, 9), dpi=110)
    core.draw_overview_plot(fig, data, variables)
    st.pyplot(fig)

    table = data.rename(columns={
        "state": "State", "variable": "Variable", "trend": "Trend",
        "change_per_year": "Change per year", "p_value": "P-value",
        "significant": "Significant"})
    st.dataframe(table, hide_index=True)
    st.caption(
        "These values come from the supplied state_trends.csv. They were "
        "pre-calculated by a method the file does not document and are NOT "
        "recomputed here, so they can differ from Page 1. The file holds 56 "
        "tests; at a 5% level a few 'significant' results are expected by "
        "chance alone.")


# ----------------------------------------------------------------- main ----
def main():
    st.title("EarthTrend Explorer")
    st.caption("Environmental trends across Indian states – T2M (temperature) "
               "and PRECTOTCORR (precipitation), 2003–2023")
    yearly, trends = load_data()
    tab1, tab2 = st.tabs(["1  Year-wise State Analysis", "2  All-State Trend Overview"])
    with tab1:
        page_series(yearly, trends)
    with tab2:
        page_overview(trends)


main()
