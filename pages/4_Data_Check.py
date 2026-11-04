# FactorFolio - Data Check page (Build Guide Step 2)
# Prof. Wysocki's rule: look at the data first. This page shows the raw numbers behind the
# portfolio before any model runs: the hygiene checklist (outline section 7), row counts
# before and after the merge, raw returns, describe() statistics and a 1.5 x IQR boxplot.
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import PercentFormatter

from core import data as fdata

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
cols = ds["tickers"] + ([fdata.BENCHMARK] if fdata.BENCHMARK in window.columns else [])
if not ds["tickers"]:
    st.error("None of these tickers have enough monthly data to analyze.")
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
