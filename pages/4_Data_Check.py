# FactorFolio - Data Check page (Build Guide Steps 2 and 9)
# Prof. Wysocki's rule: look at the data first. This page shows the raw numbers behind the
# portfolio before any model runs: the hygiene checklist (outline section 7), row counts
# before and after the merge, raw returns, describe() statistics, a 1.5 x IQR boxplot, and
# where every input comes from and how fresh it is (check 16).
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

from core import data as fdata
from core import projections, yahoo

st.set_page_config(page_title="Data Check | FactorFolio", layout="centered")
st.title("Data Check")
st.write("Look at the data first. Before any model runs, this page shows the raw numbers "
         "behind your portfolio and flags anything unusual.")

# Uses the homepage portfolio once it exists (Step 4); until then, the sample portfolio
portfolio = st.session_state.get("portfolio", fdata.DEFAULT_PORTFOLIO)
ticker_text = st.text_input("Tickers to check (separate with commas)", value=", ".join(portfolio.keys()),
                            help="Starts with your homepage portfolio. Type other tickers to inspect them here.")
tickers = fdata.clean_tickers(ticker_text)
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

with st.spinner("Downloading monthly prices from Yahoo Finance..."):
    ds = fdata.build_dataset(tickers, weights=portfolio if set(tickers) == set(portfolio) else None)

if ds["error"]:
    st.error(ds["error"])
    st.stop()

window = ds["window"]
cols = ds["tickers"] + ([fdata.BENCHMARK] if fdata.BENCHMARK in window.columns and fdata.BENCHMARK not in ds["tickers"]
                        else [])
if not ds["tickers"]:
    st.error(fdata.no_data_message(ds))
    st.stop()

# 1. Checklist
st.subheader("1. Data hygiene checklist")
icons = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
for c in ds["checks"]:   # a list (not a table) so long results wrap on phones
    st.markdown(f"{icons[c['status']]} **{c['check']}:** {c['detail']}")

# 2. Row counts (Lecture 17: always recount after a merge)
st.subheader("2. Row counts before and after the merge")
c1, c2, c3 = st.columns(3)
c1.metric("Monthly returns", ds["counts"]["return_months"])
c2.metric("Matched to factors", ds["counts"]["merged_months"])
c3.metric("Months used", ds["counts"]["window_months"])
st.caption(f"Window: {ds['as_of']['window_start']} to {ds['as_of']['window_end']}. "
           f"Prices run through {ds['as_of']['prices']}; Fama-French factors through {ds['as_of']['factors']}.")

returns_window = window[cols].copy()
returns_window.index = returns_window.index.astype(str)

# 3. Raw monthly returns
st.subheader("3. Raw monthly returns")
st.caption("Prices are adjusted for dividends and splits, so these are close to the Lecture 12 holding period return.")
st.dataframe(returns_window.sort_index(ascending=False).style.format("{:.2%}"), width="stretch")

# 4. Summary statistics (describe)
st.subheader("4. Summary statistics")
stats = returns_window.describe().T
stats["count"] = stats["count"].astype(int)
pct_cols = [c for c in stats.columns if c != "count"]
st.dataframe(stats.style.format({**{c: "{:.2%}" for c in pct_cols}, "count": "{:d}"}), width="stretch")
st.caption("std is each stock's monthly volatility, one of the risk score's inputs.")

# 5. Boxplot and extreme months (1.5 x IQR rule)
st.subheader("5. Boxplot of monthly returns")
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([returns_window[c].dropna() for c in cols], whis=1.5)
ax.set_xticks(range(1, len(cols) + 1))
ax.set_xticklabels(cols)
ax.axhline(0, color="grey", linewidth=0.8)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("Monthly return")
st.pyplot(fig)
plt.close(fig)
outliers = ds["outliers"]
if outliers.empty:
    st.write("No months fall outside the whiskers.")
else:
    st.write("Months outside the whiskers (flagged, not deleted):")
    st.dataframe(outliers.style.format({"Return": "{:.2%}", "Low fence": "{:.2%}", "High fence": "{:.2%}"}),
                 hide_index=True, width="stretch")

# 6. Fama-French factors
st.subheader("6. Fama-French factors")
factors_window = window[fdata.FACTOR_COLS].copy()
factors_window.index = factors_window.index.astype(str)
st.dataframe(factors_window.sort_index(ascending=False).style.format("{:.2%}"), width="stretch")
st.caption("From Ken French's Data Library, converted from percent to decimals. "
           "mktrf is the market's excess return; smb is size; hml is value; rf is the risk-free rate.")

# 7. Missing values
st.subheader("7. Missing values")
st.dataframe(window[cols + fdata.FACTOR_COLS].isna().sum().rename("Missing months").to_frame(),
             width="stretch")

# 8. Where every input comes from (check 16: inputs are not stale)
st.subheader("8. Where every input comes from")
TODAY = pd.Timestamp.today().normalize()
held = ds["tickers"]
rf = projections.risk_free_rate()
with st.spinner("Checking analyst forecasts and financial statements..."):
    eps = yahoo.lookup_many(yahoo.eps_forecasts, held)
    fin = yahoo.lookup_many(yahoo.financials, held)


def where_from(results, backup_date):
    """'live for AAPL, MSFT; saved backup from 2026-11-05 for KO' and an icon."""
    groups = {"live": [], "snapshot": [], None: []}
    for t, r in results.items():
        groups[r["source"]].append(t)
    parts = []
    if groups["live"]:
        parts.append("live for " + ", ".join(groups["live"]))
    if groups["snapshot"]:
        parts.append(f"saved backup from {backup_date} for " + ", ".join(groups["snapshot"]))
    if groups[None]:
        parts.append("missing for " + ", ".join(groups[None]))
    icon = "fail" if not groups["live"] and not groups["snapshot"] else ("warn" if groups[None] else "pass")
    return icon, "; ".join(parts)


def days_old(date_text):
    return (TODAY - pd.Timestamp(date_text)).days if date_text else None


eps_icon, eps_text = where_from(eps, yahoo.snapshot_date())
fin_icon, fin_text = where_from(fin, yahoo.financials_snapshot_date())
years = sorted({r["fiscal_year_end"] for r in fin.values() if r.get("fiscal_year_end")})
fin_text += f" (latest fiscal years: {', '.join(years)})" if years else ""
rows = [
    ("pass", "Monthly prices", f"Yahoo Finance, downloaded live, through {ds['as_of']['prices']}."),
    ("pass", "Fama-French factors", f"Ken French's Data Library (data/ff_factors_monthly.csv), through "
                                    f"{ds['as_of']['factors']}. Check 8 above says whether it needs an update."),
    ("pass" if rf["ok"] else "warn", "Risk-free rate", rf["note"]),
    (eps_icon, "Analyst EPS forecasts", f"Yahoo Finance eps_trend: {eps_text}."),
    (fin_icon, "Financial statements", f"Yahoo Finance balance_sheet and financials: {fin_text}."),
    ("info", "Risk score reference stocks", "CAPM estimates for 66 S&P 100 stocks from the Lecture 17 WRDS "
                                            "class file (monthly returns from June 2021), made offline in "
                                            "notebooks/training_wrds.ipynb. No raw WRDS data is in the app."),
    ("info", "Bear and bull markets", "25th and 75th percentiles of annual U.S. market returns, 1927 to 2025 "
                                      "(Ken French Data Library, data/market_scenarios.csv)."),
]
for name, date in [("analyst forecast", yahoo.snapshot_date()), ("financial statement", yahoo.financials_snapshot_date())]:
    age = days_old(date)
    if age is not None and age > 14:
        rows.append(("warn", f"Old {name} backup", f"The saved backup is {age} days old. Rerun "
                                                   "notebooks/yahoo_backup.ipynb in Colab to refresh it."))
for status, name, text in rows:   # a list, so long lines wrap on phones
    st.markdown(f"{icons[status]} **{name}:** {text}".replace("$", "\\$"))

st.caption("For education only, not investment advice. Live Yahoo answers are kept for a day; when Yahoo refuses "
           "a request, the app uses the saved backups and says so.")
