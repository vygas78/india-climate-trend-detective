# -*- coding: utf-8 -*-
"""
EarthTrend Explorer
===================
Windows desktop app (Python + Tkinter) for exploring environmental trends
across Indian states using two real CSV files:

  yearly_series.csv : state, variable, year, value
  state_trends.csv  : state, variable, trend, change_per_year, p_value, significant

Page 1 - Year-wise State Analysis: statistics are CALCULATED here from the
         yearly observations (regression, Kendall tau, Sen's slope).
Page 2 - All-State Trend Overview: shows the PRE-CALCULATED values supplied in
         state_trends.csv. They are not recomputed by this app.

IMPORTANT: The CSV files do not state units or how values were spatially
aggregated, so this app never labels the data as state-wide averages.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
    TK_ERROR = None
except ImportError as _exc:  # handled in main()
    TK_ERROR = _exc


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
APP_NAME = "EarthTrend Explorer"
APP_DIR = Path(__file__).resolve().parent

YEARLY_COLUMNS = ["state", "variable", "year", "value"]
TREND_COLUMNS = ["state", "variable", "trend", "change_per_year",
                 "p_value", "significant"]

VAR_LABELS = {
    "T2M": "T2M (temperature)",
    "PRECTOTCORR": "PRECTOTCORR (precipitation)",
}
PREFERRED_VARIABLES = ["T2M", "PRECTOTCORR"]

ALPHA = 0.05          # 5% level used for plain-language wording on Page 1
MIN_POINTS = 5        # minimum number of years needed for the statistics

SORT_RATE = "Rate of change (high to low)"
SORT_P = "P-value (low to high)"
SORT_NAME = "State name (A-Z)"
SORT_OPTIONS = [SORT_RATE, SORT_P, SORT_NAME]
BOTH_LABEL = "Both variables"

# Colours
COL_NAVY = "#12263f"
COL_NAVY_LIGHT = "#22405f"
COL_ACCENT = "#1b6ca8"
COL_ACCENT_DARK = "#14527f"
COL_BG = "#f2f5f9"
COL_CARD = "#ffffff"
COL_TEXT = "#1d2733"
COL_MUTED = "#5b6b7c"
COL_OBS = "#1f4e79"
COL_REG = "#d95f02"
COL_SEN = "#2a9d8f"
COL_INC = "#d95f02"
COL_DEC = "#1b6ca8"
COL_NS = "#b9c0c9"

FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")


class DataError(Exception):
    """Problem with the data or the user's selection (message is user-friendly)."""


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _read_csv(path, required, label):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        raise DataError(f"File not found:\n{path}")
    except pd.errors.EmptyDataError:
        raise DataError(f"The {label} file is empty:\n{path}")
    except Exception as exc:
        raise DataError(f"Could not read the {label} file:\n{path}\n\n{exc}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(
            f"This does not look like the {label} file.\n\n"
            f"Missing column(s): {', '.join(missing)}\n"
            f"Required columns: {', '.join(required)}\n"
            f"Columns found: {', '.join(df.columns)}"
        )
    return df[required].copy()


def load_yearly_csv(path):
    """Load yearly_series.csv and validate it."""
    df = _read_csv(path, YEARLY_COLUMNS, "yearly series (yearly_series.csv)")
    for col in ("state", "variable"):
        df[col] = df[col].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["year", "value"]).copy()
    dropped = before - len(df)
    if df.empty:
        raise DataError("The yearly series file has no usable rows "
                        "(year and value must be numbers).")
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["state", "variable", "year"]).reset_index(drop=True)
    df.attrs["dropped_rows"] = dropped
    return df


def load_trends_csv(path):
    """Load state_trends.csv and validate it."""
    df = _read_csv(path, TREND_COLUMNS, "state trends (state_trends.csv)")
    for col in ("state", "variable", "trend"):
        df[col] = df[col].astype(str).str.strip()
    df["change_per_year"] = pd.to_numeric(df["change_per_year"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["significant"] = (
        df["significant"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "y"])
    )
    df = df.dropna(subset=["change_per_year"]).copy()
    if df.empty:
        raise DataError("The state trends file has no usable rows "
                        "(change_per_year must be a number).")
    return df.reset_index(drop=True)


def order_variables(variables):
    variables = list(variables)
    first = [v for v in PREFERRED_VARIABLES if v in variables]
    rest = sorted(v for v in variables if v not in PREFERRED_VARIABLES)
    return first + rest


# --------------------------------------------------------------------------
# Statistics (Page 1) - calculated from the yearly observations
# --------------------------------------------------------------------------
def analyse_series(df, state, variable, start_year, end_year):
    """Return a dict of statistics for one state/variable/year range."""
    if start_year > end_year:
        raise DataError("The start year must not be later than the end year.")
    sub = df[(df["state"] == state) & (df["variable"] == variable)
             & (df["year"].between(start_year, end_year))]
    if sub.empty:
        raise DataError(f"No data found for {state} / {variable} "
                        f"between {start_year} and {end_year}.")
    sub = (sub.groupby("year", as_index=False)["value"].mean()
              .sort_values("year"))
    n = len(sub)
    if n < MIN_POINTS:
        raise DataError(
            f"Only {n} year(s) selected. Please choose at least {MIN_POINTS} "
            f"years (more is better) so the statistics are meaningful.")

    x = sub["year"].to_numpy(dtype=float)
    y = sub["value"].to_numpy(dtype=float)

    # Ordinary least squares
    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, n - 2)
    slope_ci = (lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr)

    # Kendall tau (rank correlation between year and value)
    tau, tau_p = stats.kendalltau(x, y)

    # Sen's (Theil-Sen) slope with 95% confidence interval
    sen_slope, sen_intercept, sen_low, sen_high = (
        float(v) for v in stats.theilslopes(y, x, alpha=0.95))

    # Rough serial-correlation check on regression residuals
    resid = y - (lr.intercept + lr.slope * x)
    lag1 = np.nan
    if n >= 4 and np.std(resid) > 0:
        lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])

    return {
        "state": state, "variable": variable,
        "start": int(x.min()), "end": int(x.max()), "n": n,
        "years": x, "values": y,
        "slope": float(lr.slope), "intercept": float(lr.intercept),
        "slope_ci": (float(slope_ci[0]), float(slope_ci[1])),
        "r2": float(lr.rvalue ** 2) if np.isfinite(lr.rvalue) else float("nan"),
        "lr_p": float(lr.pvalue),
        "tau": float(tau), "tau_p": float(tau_p),
        "sen_slope": sen_slope, "sen_intercept": sen_intercept,
        "sen_low": sen_low, "sen_high": sen_high,
        "lag1": lag1,
    }


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def fmt_num(v, digits=4, signed=False):
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def build_report(res, supplied, var_label):
    """Build the text shown beside the chart as a list of (text, tag)."""
    seg = []
    add = seg.append

    add((f"{res['state']}\n", "title"))
    add((f"{var_label}\n", "sub"))
    add((f"{res['start']}–{res['end']}  |  {res['n']} yearly values\n", "sub"))

    add(("\nLinear regression (calculated here)\n", "h"))
    add(("Rate of change (slope): ", "row"))
    add((f"{fmt_num(res['slope'], signed=True)} per year\n", "val"))
    add(("95% CI of slope: ", "row"))
    add((f"{fmt_num(res['slope_ci'][0], signed=True)} to "
         f"{fmt_num(res['slope_ci'][1], signed=True)}\n", "val"))
    add(("P-value: ", "row"))
    add((f"{fmt_p(res['lr_p'])}\n", "val"))
    add(("R-squared: ", "row"))
    add((f"{fmt_num(res['r2'], 3)}\n", "val"))

    add(("\nKendall tau (calculated here)\n", "h"))
    add(("Tau: ", "row"))
    add((f"{fmt_num(res['tau'], 3, signed=True)}\n", "val"))
    add(("P-value: ", "row"))
    add((f"{fmt_p(res['tau_p'])}\n", "val"))

    add(("\nSen's slope (calculated here)\n", "h"))
    add(("Sen's slope: ", "row"))
    add((f"{fmt_num(res['sen_slope'], signed=True)} per year\n", "val"))
    add(("95% confidence interval: ", "row"))
    add((f"{fmt_num(res['sen_low'], signed=True)} to "
         f"{fmt_num(res['sen_high'], signed=True)}\n", "val"))

    # Plain-language reading
    lr_sig = res["lr_p"] < ALPHA
    mk_sig = res["tau_p"] < ALPHA
    sen_sig = res["sen_low"] > 0 or res["sen_high"] < 0
    count = int(lr_sig) + int(mk_sig) + int(sen_sig)
    direction = ("rising" if res["sen_slope"] > 0
                 else "falling" if res["sen_slope"] < 0 else "flat")
    add(("\nHow to read this\n", "h"))
    if count == 3:
        add(("All three methods flag a trend at the 5% level "
             f"(fitted direction: {direction}). Treat this as a signal worth "
             "investigating, not proof.\n", "ok"))
    elif count == 0:
        add(("None of the three methods shows a clear trend at the 5% level. "
             "The year-to-year variation may simply be natural variability.\n",
             "row"))
    else:
        add((f"The methods disagree ({count} of 3 flag a trend), so this "
             "trend is uncertain.\n", "warn"))

    add(("\nDiagnostic\n", "h"))
    add(("Residual lag-1 autocorrelation: ", "row"))
    add((f"{fmt_num(res['lag1'], 2, signed=True)}\n", "val"))
    if np.isfinite(res["lag1"]) and res["lag1"] > 2 / np.sqrt(res["n"]):
        add(("Residuals look positively correlated from year to year, so the "
             "p-values above are probably too optimistic.\n", "warn"))
    else:
        add(("No strong sign of year-to-year correlation (rough check only).\n",
             "row"))

    add(("\nSupplied summary (state_trends.csv)\n", "h"))
    if supplied is None:
        add(("No matching row loaded (load state_trends.csv to compare).\n",
             "note"))
    else:
        add(("Trend label: ", "row"))
        add((f"{supplied['trend']}\n", "val"))
        add(("Change per year: ", "row"))
        add((f"{fmt_num(supplied['change_per_year'], signed=True)}\n", "val"))
        add(("P-value: ", "row"))
        add((f"{fmt_p(supplied['p_value'])}\n", "val"))
        add(("Marked significant: ", "row"))
        add(("Yes\n" if supplied["significant"] else "No\n", "val"))
        add(("Pre-calculated values; the method and year range are not "
             "documented in the CSV, so they can differ from the numbers "
             "calculated above.\n", "note"))

    add(("\nCaution\n", "h"))
    add(("These tests treat each yearly value as independent and do not model "
         "serial correlation or seasonality. With only a few years of data, "
         "treat p-values as a rough guide. The CSV gives no units and does "
         "not say how values were aggregated over each state.\n", "note"))
    return seg


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def draw_message(fig, message, color=COL_MUTED):
    fig.clear()
    fig.text(0.5, 0.5, message, ha="center", va="center", fontsize=12,
             color=color, wrap=True)


def draw_series_plot(fig, res, var_label):
    fig.clear()
    ax = fig.add_subplot(111)
    x, y = res["years"], res["values"]
    ax.plot(x, y, marker="o", markersize=5, linewidth=2, color=COL_OBS,
            label="Observed yearly values")
    ax.plot(x, res["intercept"] + res["slope"] * x, linestyle="--",
            linewidth=2, color=COL_REG,
            label=f"Linear regression ({fmt_num(res['slope'], signed=True)}/yr)")
    ax.plot(x, res["sen_intercept"] + res["sen_slope"] * x, linestyle=":",
            linewidth=2.2, color=COL_SEN,
            label=f"Sen's slope ({fmt_num(res['sen_slope'], signed=True)}/yr)")
    ax.set_title(f"{res['state']} – {var_label}, {res['start']}–{res['end']}",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{res['variable']} (units not given in CSV)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()


def filter_trends(trends, variables, only_significant, sort_mode):
    d = trends[trends["variable"].isin(variables)]
    if only_significant:
        d = d[d["significant"]]
    if sort_mode == SORT_P:
        d = d.sort_values(["variable", "p_value"], ascending=[True, True])
    elif sort_mode == SORT_NAME:
        d = d.sort_values(["variable", "state"], ascending=[True, True])
    else:
        d = d.sort_values(["variable", "change_per_year"],
                          ascending=[True, False])
    return d.reset_index(drop=True)


def draw_overview_plot(fig, data, variables):
    """Horizontal bar chart of supplied change_per_year, one panel per variable."""
    fig.clear()
    n_panels = len(variables)
    for i, var in enumerate(variables):
        ax = fig.add_subplot(1, n_panels, i + 1)
        d = data[data["variable"] == var]
        ax.set_title(VAR_LABELS.get(var, var), fontsize=11,
                     fontweight="bold", loc="left")
        if d.empty:
            ax.text(0.5, 0.5, "No rows to show", ha="center", va="center",
                    transform=ax.transAxes, color=COL_MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        colors = []
        for _, row in d.iterrows():
            if not row["significant"]:
                colors.append(COL_NS)
            elif row["change_per_year"] >= 0:
                colors.append(COL_INC)
            else:
                colors.append(COL_DEC)
        pos = np.arange(len(d))
        ax.barh(pos, d["change_per_year"].to_numpy(), color=colors, height=0.72)
        ax.set_yticks(pos)
        ax.set_yticklabels(d["state"].tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Change per year (supplied; units not given)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [
        Patch(color=COL_INC, label="Significant, increasing"),
        Patch(color=COL_DEC, label="Significant, decreasing"),
        Patch(color=COL_NS, label="Not significant (as supplied)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class EarthTrendApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_NAME)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1320, sw - 60), min(820, sh - 90)
        root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 2 - 20, 0)}")
        root.minsize(980, 640)
        root.configure(bg=COL_BG)

        self.yearly = None
        self.trends = None
        self.yearly_path = None
        self.trends_path = None
        self.var_label_to_key = {}
        self.pages = {}
        self.nav_buttons = {}

        self._setup_style()
        self._build_header()
        self._build_status_bar()
        self.container = tk.Frame(root, bg=COL_BG)
        self.container.pack(fill="both", expand=True)
        self._build_series_page()
        self._build_overview_page()

        self.auto_load()
        self.show_page("series")

    # ---- style -----------------------------------------------------------
    def _setup_style(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", font=FONT)
        st.configure("TFrame", background=COL_BG)
        st.configure("Card.TFrame", background=COL_CARD)
        st.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        st.configure("Card.TLabel", background=COL_CARD, foreground=COL_TEXT)
        st.configure("Muted.TLabel", background=COL_CARD, foreground=COL_MUTED,
                     font=("Segoe UI", 9))
        st.configure("TCombobox", padding=4)
        st.configure("TButton", padding=(10, 5))
        st.configure("Accent.TButton", background=COL_ACCENT,
                     foreground="white", padding=(16, 6), font=FONT_BOLD)
        st.map("Accent.TButton",
               background=[("active", COL_ACCENT_DARK)],
               foreground=[("active", "white")])
        st.configure("TCheckbutton", background=COL_CARD, foreground=COL_TEXT)
        st.configure("Treeview", rowheight=24, font=FONT)
        st.configure("Treeview.Heading", font=FONT_BOLD)

    # ---- header, status bar ---------------------------------------------
    def _build_header(self):
        bar = tk.Frame(self.root, bg=COL_NAVY)
        bar.pack(fill="x")
        tk.Label(bar, text=APP_NAME, bg=COL_NAVY, fg="white",
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=(18, 8), pady=10)
        tk.Label(bar, text="Environmental trends across Indian states",
                 bg=COL_NAVY, fg="#9fb3c8",
                 font=("Segoe UI", 9)).pack(side="left", padx=6)
        nav = tk.Frame(bar, bg=COL_NAVY)
        nav.pack(side="right", padx=12)
        for key, text in (("series", "1   Year-wise State Analysis"),
                          ("overview", "2   All-State Trend Overview")):
            btn = tk.Button(nav, text=text, relief="flat", bd=0, padx=16,
                            pady=8, font=FONT_BOLD, cursor="hand2",
                            command=lambda k=key: self.show_page(k))
            btn.pack(side="left", padx=4)
            self.nav_buttons[key] = btn

        data = tk.Frame(self.root, bg="#e3e9f0")
        data.pack(fill="x")
        ttk.Button(data, text="Load yearly CSV...",
                   command=self.pick_yearly).pack(side="left", padx=(14, 4), pady=6)
        ttk.Button(data, text="Load trends CSV...",
                   command=self.pick_trends).pack(side="left", padx=4)
        self.file_label = tk.Label(data, text="", bg="#e3e9f0", fg=COL_MUTED,
                                   font=("Segoe UI", 9), anchor="w")
        self.file_label.pack(side="left", padx=12, fill="x", expand=True)

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 bg="#d6dfe9", fg=COL_TEXT, font=("Segoe UI", 9),
                 padx=12, pady=4).pack(side="bottom", fill="x")

    def set_status(self, message):
        self.status_var.set(message)

    def show_page(self, key):
        for frame in self.pages.values():
            frame.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        for k, btn in self.nav_buttons.items():
            bg = COL_ACCENT if k == key else COL_NAVY_LIGHT
            btn.configure(bg=bg, fg="white", activebackground=COL_ACCENT_DARK,
                          activeforeground="white")
        if key == "overview":
            self.refresh_overview()
        else:
            self.canvas1.draw_idle()

    # ---- Page 1: year-wise analysis -------------------------------------
    def _build_series_page(self):
        page = tk.Frame(self.container, bg=COL_BG)
        self.pages["series"] = page

        ctl = ttk.Frame(page, style="Card.TFrame", padding=(14, 10))
        ctl.pack(fill="x", padx=14, pady=(12, 8))

        self.state_var = tk.StringVar()
        self.var_var = tk.StringVar()
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()

        ttk.Label(ctl, text="State", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(ctl, text="Variable", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=(14, 0))
        ttk.Label(ctl, text="From year", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(14, 0))
        ttk.Label(ctl, text="To year", style="Card.TLabel").grid(row=0, column=3, sticky="w", padx=(14, 0))

        self.state_cb = ttk.Combobox(ctl, textvariable=self.state_var, state="readonly", width=24)
        self.var_cb = ttk.Combobox(ctl, textvariable=self.var_var, state="readonly", width=28)
        self.start_cb = ttk.Combobox(ctl, textvariable=self.start_var, state="readonly", width=8)
        self.end_cb = ttk.Combobox(ctl, textvariable=self.end_var, state="readonly", width=8)
        self.state_cb.grid(row=1, column=0, sticky="w")
        self.var_cb.grid(row=1, column=1, sticky="w", padx=(14, 0))
        self.start_cb.grid(row=1, column=2, sticky="w", padx=(14, 0))
        self.end_cb.grid(row=1, column=3, sticky="w", padx=(14, 0))
        ttk.Button(ctl, text="Analyse", style="Accent.TButton",
                   command=lambda: self.run_analysis(from_button=True)
                   ).grid(row=1, column=4, padx=(20, 0))
        for cb in (self.state_cb, self.var_cb, self.start_cb, self.end_cb):
            cb.bind("<<ComboboxSelected>>", lambda _e: self.run_analysis())

        body = tk.Frame(page, bg=COL_BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=COL_CARD)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.fig1 = Figure(figsize=(6, 4.5), dpi=100)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=left)
        toolbar = NavigationToolbar2Tk(self.canvas1, left, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas1.get_tk_widget().pack(side="top", fill="both", expand=True)

        right = tk.Frame(body, bg=COL_CARD)
        right.grid(row=0, column=1, sticky="nsew")
        scroll = ttk.Scrollbar(right, orient="vertical")
        self.report = tk.Text(right, wrap="word", width=50, bd=0, padx=14,
                              pady=10, bg=COL_CARD, fg=COL_TEXT, font=FONT,
                              state="disabled", cursor="arrow",
                              yscrollcommand=scroll.set)
        scroll.configure(command=self.report.yview)
        scroll.pack(side="right", fill="y")
        self.report.pack(side="left", fill="both", expand=True)
        self.report.tag_configure("title", font=("Segoe UI", 14, "bold"), foreground=COL_NAVY)
        self.report.tag_configure("sub", font=("Segoe UI", 10), foreground=COL_MUTED)
        self.report.tag_configure("h", font=FONT_BOLD, foreground=COL_ACCENT, spacing1=8, spacing3=2)
        self.report.tag_configure("row", font=FONT)
        self.report.tag_configure("val", font=FONT_BOLD)
        self.report.tag_configure("note", font=("Segoe UI", 9, "italic"), foreground=COL_MUTED)
        self.report.tag_configure("warn", font=FONT, foreground="#b3261e")
        self.report.tag_configure("ok", font=FONT, foreground="#1b7f4b")

        draw_message(self.fig1, "Load yearly_series.csv to begin.")
        self._set_report([("Results will appear here.\n", "note")])

    def _set_report(self, segments):
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        for text, tag in segments:
            self.report.insert("end", text, tag)
        self.report.configure(state="disabled")
        self.report.yview_moveto(0)

    def var_key(self, label):
        return self.var_label_to_key.get(label, label)

    def populate_series_controls(self):
        df = self.yearly
        states = sorted(df["state"].unique())
        variables = order_variables(df["variable"].unique())
        years = [str(y) for y in sorted(df["year"].unique())]
        self.var_label_to_key = {VAR_LABELS.get(v, v): v for v in variables}

        self.state_cb["values"] = states
        self.var_cb["values"] = list(self.var_label_to_key.keys())
        self.start_cb["values"] = years
        self.end_cb["values"] = years

        if self.state_var.get() not in states:
            self.state_var.set(states[0])
        if self.var_var.get() not in self.var_label_to_key:
            self.var_var.set(list(self.var_label_to_key.keys())[0])
        if self.start_var.get() not in years:
            self.start_var.set(years[0])
        if self.end_var.get() not in years:
            self.end_var.set(years[-1])

    def run_analysis(self, from_button=False):
        if self.yearly is None:
            draw_message(self.fig1, "Load yearly_series.csv to begin.")
            self.canvas1.draw_idle()
            if from_button:
                messagebox.showinfo(
                    APP_NAME, "Please load yearly_series.csv first "
                              "(button at the top left).")
            return
        try:
            state = self.state_var.get()
            var = self.var_key(self.var_var.get())
            y0 = int(self.start_var.get())
            y1 = int(self.end_var.get())
        except ValueError:
            self._series_error("Please choose a state, a variable, and valid "
                               "start and end years.", from_button)
            return
        try:
            res = analyse_series(self.yearly, state, var, y0, y1)
        except DataError as exc:
            self._series_error(str(exc), from_button)
            return
        except Exception as exc:  # unexpected - still show something useful
            self._series_error(f"Unexpected problem while analysing:\n{exc}",
                               from_button)
            return

        supplied = None
        if self.trends is not None:
            m = self.trends[(self.trends["state"] == state)
                            & (self.trends["variable"] == var)]
            if not m.empty:
                supplied = m.iloc[0]
        label = VAR_LABELS.get(var, var)
        draw_series_plot(self.fig1, res, label)
        self.canvas1.draw_idle()
        self._set_report(build_report(res, supplied, label))
        self.set_status(f"Analysed {state}, {var}, {res['start']}-{res['end']} "
                        f"({res['n']} years).")

    def _series_error(self, message, show_dialog):
        draw_message(self.fig1, message, color="#b3261e")
        self.canvas1.draw_idle()
        self._set_report([("Cannot analyse this selection\n", "title"),
                          (message + "\n", "warn")])
        self.set_status(message.replace("\n", " "))
        if show_dialog:
            messagebox.showwarning("Cannot analyse", message)

    # ---- Page 2: all-state overview -------------------------------------
    def _build_overview_page(self):
        page = tk.Frame(self.container, bg=COL_BG)
        self.pages["overview"] = page

        ctl = ttk.Frame(page, style="Card.TFrame", padding=(14, 10))
        ctl.pack(fill="x", padx=14, pady=(12, 8))
        self.ov_var_var = tk.StringVar(value=BOTH_LABEL)
        self.ov_sort_var = tk.StringVar(value=SORT_RATE)
        self.ov_sig_var = tk.BooleanVar(value=False)

        ttk.Label(ctl, text="Variable", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(ctl, text="Sort by", style="Card.TLabel").grid(row=0, column=1, sticky="w", padx=(14, 0))
        self.ov_var_cb = ttk.Combobox(ctl, textvariable=self.ov_var_var,
                                      state="readonly", width=30, values=[BOTH_LABEL])
        self.ov_sort_cb = ttk.Combobox(ctl, textvariable=self.ov_sort_var,
                                       state="readonly", width=28, values=SORT_OPTIONS)
        self.ov_var_cb.grid(row=1, column=0, sticky="w")
        self.ov_sort_cb.grid(row=1, column=1, sticky="w", padx=(14, 0))
        ttk.Checkbutton(ctl, text="Show only significant (as supplied)",
                        variable=self.ov_sig_var,
                        command=self.refresh_overview).grid(row=1, column=2, padx=(20, 0))
        for cb in (self.ov_var_cb, self.ov_sort_cb):
            cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh_overview())

        self.ov_summary = ttk.Label(
            page, style="TLabel", font=FONT_BOLD, justify="left",
            text="Load state_trends.csv to see the overview.")
        self.ov_summary.pack(fill="x", padx=16, pady=(0, 2))
        ttk.Label(
            page, style="TLabel", font=("Segoe UI", 9, "italic"),
            foreground=COL_MUTED, wraplength=1150, justify="left",
            text=("These values come from the supplied state_trends.csv. They were "
                  "pre-calculated by a method the file does not document and are NOT "
                  "recomputed here, so they can differ from Page 1. The file holds "
                  "56 tests; at a 5% level a few 'significant' results are expected "
                  "by chance alone.")
        ).pack(fill="x", padx=16, pady=(0, 6))

        body = tk.Frame(page, bg=COL_BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=COL_CARD)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        cols = ("state", "variable", "trend", "change", "p", "sig")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=20)
        heads = {"state": ("State", 130, "w"), "variable": ("Variable", 105, "w"),
                 "trend": ("Trend", 85, "center"), "change": ("Change/yr", 80, "e"),
                 "p": ("P-value", 70, "e"), "sig": ("Signif.", 55, "center")}
        for c in cols:
            text, width, anchor = heads[c]
            self.tree.heading(c, text=text)
            self.tree.column(c, width=width, anchor=anchor, stretch=False)
        self.tree.tag_configure("inc", background="#fdeee0")
        self.tree.tag_configure("dec", background="#e3eefb")
        self.tree.tag_configure("ns", foreground="#6b7785")
        vs = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)

        right = tk.Frame(body, bg=COL_CARD)
        right.grid(row=0, column=1, sticky="nsew")
        self.fig2 = Figure(figsize=(7, 6), dpi=100)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas2, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas2.get_tk_widget().pack(side="top", fill="both", expand=True)
        draw_message(self.fig2, "Load state_trends.csv to see the overview.")

    def populate_overview_controls(self):
        variables = order_variables(self.trends["variable"].unique())
        labels = [BOTH_LABEL] + [VAR_LABELS.get(v, v) for v in variables]
        self.ov_var_cb["values"] = labels
        if self.ov_var_var.get() not in labels:
            self.ov_var_var.set(BOTH_LABEL)

    def selected_overview_variables(self):
        all_vars = order_variables(self.trends["variable"].unique())
        choice = self.ov_var_var.get()
        if choice == BOTH_LABEL:
            return all_vars
        for v in all_vars:
            if VAR_LABELS.get(v, v) == choice:
                return [v]
        return all_vars

    def refresh_overview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.trends is None:
            draw_message(self.fig2, "Load state_trends.csv to see the overview.")
            self.canvas2.draw_idle()
            self.ov_summary.configure(text="Load state_trends.csv to see the overview.")
            return
        try:
            variables = self.selected_overview_variables()
            data = filter_trends(self.trends, variables,
                                 self.ov_sig_var.get(), self.ov_sort_var.get())
            for _, r in data.iterrows():
                tag = ("ns" if not r["significant"]
                       else "inc" if r["change_per_year"] >= 0 else "dec")
                self.tree.insert("", "end", tags=(tag,), values=(
                    r["state"], r["variable"], r["trend"],
                    fmt_num(r["change_per_year"], signed=True),
                    fmt_p(r["p_value"]),
                    "Yes" if r["significant"] else "No"))
            draw_overview_plot(self.fig2, data, variables)
            self.canvas2.draw_idle()

            parts = []
            for v in variables:
                d = self.trends[self.trends["variable"] == v]
                sig = d[d["significant"]]
                n_inc = int((sig["change_per_year"] > 0).sum())
                n_dec = int((sig["change_per_year"] < 0).sum())
                parts.append(f"{v}: {len(sig)} of {len(d)} states marked significant "
                             f"({n_inc} increasing, {n_dec} decreasing)")
            self.ov_summary.configure(text="   |   ".join(parts))
            self.set_status(f"Overview updated ({len(data)} rows shown).")
        except Exception as exc:
            draw_message(self.fig2, f"Could not draw the overview:\n{exc}", color="#b3261e")
            self.canvas2.draw_idle()
            self.set_status(f"Overview error: {exc}")

    # ---- file loading -----------------------------------------------------
    @staticmethod
    def _find_file(*patterns):
        for folder in (APP_DIR, Path.cwd()):
            for pat in patterns:
                matches = sorted(folder.glob(pat))
                if matches:
                    return matches[0]
        return None

    def auto_load(self):
        y = self._find_file("yearly_series.csv", "yearly_series*.csv")
        t = self._find_file("state_trends.csv", "state_trends*.csv")
        if y:
            self.load_yearly(y, quiet=True)
        if t:
            self.load_trends(t, quiet=True)
        missing = []
        if self.yearly is None:
            missing.append("yearly_series.csv")
        if self.trends is None:
            missing.append("state_trends.csv")
        if missing:
            self.set_status("Not loaded: " + ", ".join(missing) +
                            ". Use the Load buttons at the top to select the file(s).")
        else:
            self.set_status("Both CSV files loaded.")

    def load_yearly(self, path, quiet=False):
        try:
            df = load_yearly_csv(path)
        except DataError as exc:
            if quiet:
                self.set_status(str(exc).replace("\n", " "))
            else:
                messagebox.showerror("Could not load yearly CSV", str(exc))
            return False
        self.yearly = df
        self.yearly_path = Path(path)
        self.populate_series_controls()
        self.update_file_label()
        dropped = df.attrs.get("dropped_rows", 0)
        if dropped:
            self.set_status(f"Loaded yearly data; {dropped} row(s) skipped "
                            "because year or value was not a number.")
        self.run_analysis()
        return True

    def load_trends(self, path, quiet=False):
        try:
            df = load_trends_csv(path)
        except DataError as exc:
            if quiet:
                self.set_status(str(exc).replace("\n", " "))
            else:
                messagebox.showerror("Could not load trends CSV", str(exc))
            return False
        self.trends = df
        self.trends_path = Path(path)
        self.populate_overview_controls()
        self.update_file_label()
        self.refresh_overview()
        if self.yearly is not None:
            self.run_analysis()
        return True

    def pick_yearly(self):
        path = filedialog.askopenfilename(
            title="Select yearly_series.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path and self.load_yearly(path):
            self.set_status(f"Loaded {Path(path).name}.")

    def pick_trends(self):
        path = filedialog.askopenfilename(
            title="Select state_trends.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path and self.load_trends(path):
            self.set_status(f"Loaded {Path(path).name}.")

    def update_file_label(self):
        if self.yearly is not None:
            df = self.yearly
            y = (f"Yearly: {self.yearly_path.name} ({len(df)} rows, "
                 f"{df['state'].nunique()} states, "
                 f"{df['year'].min()}–{df['year'].max()})")
        else:
            y = "Yearly: not loaded"
        if self.trends is not None:
            t = f"Trends: {self.trends_path.name} ({len(self.trends)} rows)"
        else:
            t = "Trends: not loaded"
        self.file_label.configure(text=f"{y}     |     {t}")


# --------------------------------------------------------------------------
def main():
    if TK_ERROR is not None:
        print("Could not start: a required library is missing.")
        print(f"Details: {TK_ERROR}")
        print("Run run_windows.bat, or: py -m pip install -r requirements.txt")
        input("Press Enter to close...")
        sys.exit(1)
    try:  # crisp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    EarthTrendApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
