"""
FactorFolio data layer (Build Guide Step 2).

Downloads monthly prices from Yahoo Finance, turns them into monthly returns, loads the
monthly Fama-French factors, merges the two on year-month, and runs the data hygiene
checks from section 7 of the project outline.

Code patterns adapted from the AC420 class notebooks (Prof. Wysocki):
- Lecture 12 & 13 and Lecture 17: merge stock returns with Fama-French factors, then recount rows
- Lecture 17 study guide: merge on year-month with to_period('M'), not on the exact date
- Lectures 1 to 3: look at the data first (describe(), boxplots, the 1.5 x IQR rule)
- Lecture 16 & 17 comparables notebook: try/except around downloads so one bad ticker can't crash the app
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    cache_data = st.cache_data(ttl=86400, show_spinner=False)  # keep downloads for one day
except ImportError:  # lets these functions run in Colab without Streamlit
    def cache_data(func):
        return func

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WINDOW_MONTHS = 60   # estimation window (outline section 5.1)
MIN_MONTHS = 36      # check 3: minimum history for a regression
BENCHMARK = "SPY"
DEFAULT_PORTFOLIO = {"AAPL": 30, "MSFT": 20, "NVDA": 20, "KO": 15, "JNJ": 15}
FACTOR_COLS = ["mktrf", "smb", "hml", "rf"]


def clean_tickers(raw):
    """'aapl, msft ' -> ['AAPL', 'MSFT']. Yahoo writes class shares with a dash (BRK.B -> BRK-B)."""
    if isinstance(raw, str):
        raw = raw.replace(";", ",").replace(" ", ",").split(",")
    tickers = []
    for t in raw:
        t = str(t).strip().upper().replace(".", "-")
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def check_weights(weights):
    """Check 2: portfolio weights (in %) should add up to 100."""
    total = float(sum(weights.values()))
    ok = abs(total - 100) < 0.01
    detail = "Weights add to 100%." if ok else f"Weights add to {total:.1f}%, not 100%. Use Normalize to rescale them."
    return {"check": "2. Weights add to 100%", "status": "pass" if ok else "warn", "detail": detail}


def normalize_weights(weights):
    """Rescale weights so they add to 100% (keeps each stock's share of the total)."""
    total = float(sum(weights.values()))
    if total <= 0:
        return weights
    return {t: w * 100 / total for t, w in weights.items()}


def download_monthly_prices(tickers):
    """Monthly dividend-adjusted closing prices, one column per ticker, indexed by year-month.

    Returns (prices, error_message). error_message is None when the download worked.
    Failed downloads are not cached, so a Yahoo rate limit doesn't stick for a day.
    """
    try:
        return _download_close(tuple(tickers)), None
    except Exception as e:  # network problems, Yahoo rate limits, no data
        return pd.DataFrame(), f"Yahoo Finance download failed: {e}"


@cache_data
def _download_close(tickers):
    tickers = list(tickers)
    start = (pd.Timestamp.today() - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, interval="1mo", auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        raise ValueError("no data returned. Check the tickers or try again in a minute.")

    # yfinance returns columns like ('Close', 'AAPL'); older versions use flat columns for one ticker
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=-1)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.copy()

    # Monthly bars are dated on the 1st; convert to year-month (Lecture 17 fix)
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = idx.to_period("M")
    close.index.name = "ym"
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def monthly_returns(prices):
    """Monthly returns from adjusted prices (dividends included, so close to the Lecture 12 HPR)."""
    rets = prices.pct_change(fill_method=None).iloc[1:]   # the first month has no return
    this_month = pd.Timestamp.today().to_period("M")
    return rets[rets.index < this_month]                    # drop the unfinished current month


@cache_data
def load_factors():
    """Monthly Fama-French factors from Ken French's library, converted from percent to decimals."""
    ff = pd.read_csv(DATA_DIR / "ff_factors_monthly.csv")
    ff["ym"] = pd.to_datetime(ff["yyyymm"].astype(str), format="%Y%m").dt.to_period("M")
    ff = ff.set_index("ym")[FACTOR_COLS] / 100   # check 4: Ken French posts percents
    return ff


def iqr_outliers(returns):
    """Check 9: flag months outside the boxplot whiskers (1.5 x IQR rule from Lectures 1 to 3)."""
    rows = []
    for t in returns.columns:
        s = returns[t].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for ym, r in s[(s < low) | (s > high)].items():
            rows.append({"Ticker": t, "Month": str(ym), "Return": r, "Low fence": low, "High fence": high})
    return pd.DataFrame(rows, columns=["Ticker", "Month", "Return", "Low fence", "High fence"])


def build_dataset(tickers, weights=None):
    """Everything the models need for a list of tickers, plus the data hygiene check results.

    Returns a dict. The key table is 'window': the last 60 months where both the stock
    returns and the Fama-French factors exist, with one column per ticker (plus SPY) and
    the factor columns mktrf, smb, hml and rf.
    """
    requested = clean_tickers(tickers)
    download_list = requested + ([BENCHMARK] if BENCHMARK not in requested else [])
    prices, error = download_monthly_prices(tuple(download_list))
    factors = load_factors()
    result = {"requested": requested, "tickers": [], "prices": prices, "factors": factors,
              "returns": pd.DataFrame(), "window": pd.DataFrame(), "excess": pd.DataFrame(),
              "outliers": iqr_outliers(pd.DataFrame()), "missing_tickers": [], "short_history": [],
              "counts": {}, "as_of": {}, "checks": [], "error": error}
    if error:
        return result

    checks = []

    # Check 1: tickers are real (Yahoo returned prices for them)
    missing = [t for t in download_list if t not in prices.columns or prices[t].dropna().empty]
    found = [t for t in requested if t not in missing]
    checks.append({"check": "1. Tickers are real",
                   "status": "pass" if not missing else "warn",
                   "detail": "Prices found for every ticker." if not missing
                   else "No Yahoo Finance data for: " + ", ".join(missing) + ". They are ignored."})

    # Check 2: weights (only when the homepage passes them in)
    if weights:
        checks.append(check_weights(weights))

    returns = monthly_returns(prices.drop(columns=[t for t in missing if t in prices.columns]))
    returns.index.name = "ym"

    # Check 4: factor units after dividing by 100
    max_abs = factors.abs().max().max()
    checks.append({"check": "4. Factor units are decimals",
                   "status": "pass" if max_abs <= 1 else "fail",
                   "detail": f"Largest factor value is {max_abs:.4f} (decimals, e.g. 0.0281 = 2.81%)." if max_abs <= 1
                   else "Factor values above 1: the percent-to-decimal conversion was missed."})

    # Checks 5 and 6: merge on year-month, then recount (Lecture 17)
    combined = pd.merge(returns.reset_index(), factors.reset_index(), on="ym", how="inner")
    unmatched = returns.index.difference(factors.index)
    early_gaps = [m for m in unmatched if m <= factors.index.max()]
    checks.append({"check": "5. Dates line up",
                   "status": "pass" if not early_gaps else "fail",
                   "detail": "Both files use year-month keys; the only unmatched months are newer than the factor file."
                   if not early_gaps else f"{len(early_gaps)} months inside the factor file's range did not match."})
    window = combined.tail(WINDOW_MONTHS).set_index("ym")
    checks.append({"check": "6. Merge kept the right rows",
                   "status": "pass" if len(window) >= MIN_MONTHS else "warn",
                   "detail": f"{len(returns)} monthly returns before the merge, {len(combined)} after; "
                             f"using the last {len(window)} months."})

    # Check 3: enough history for each stock inside the window
    stock_cols = [t for t in found if t in window.columns]
    short = [t for t in stock_cols if window[t].notna().sum() < MIN_MONTHS]
    keep = [t for t in stock_cols if t not in short]
    checks.append({"check": "3. Enough history",
                   "status": "pass" if not short else "warn",
                   "detail": f"Every stock has at least {MIN_MONTHS} months." if not short
                   else "Fewer than " + str(MIN_MONTHS) + " months, left out of the regressions: " + ", ".join(short)})

    # Check 7: missing values inside the window
    cols_in_window = keep + ([BENCHMARK] if BENCHMARK in window.columns else [])
    n_missing = int(window[cols_in_window + FACTOR_COLS].isna().sum().sum())
    checks.append({"check": "7. No missing values",
                   "status": "pass" if n_missing == 0 else "warn",
                   "detail": "No missing values in the window (the first month, which has no return, was dropped)."
                   if n_missing == 0 else f"{n_missing} missing values in the window (months before a stock started trading)."})

    # Check 8: factor file is current
    last_return, last_factor = returns.index.max(), factors.index.max()
    lag = (last_return - last_factor).n if last_return > last_factor else 0
    checks.append({"check": "8. Factor file is current",
                   "status": "pass" if lag <= 3 else "warn",
                   "detail": f"Prices run through {last_return}; Fama-French factors through {last_factor}."
                   + ("" if lag <= 3 else " Download a newer factor file from Ken French's library.")})

    # Check 9: extreme months (flagged and kept)
    outliers = iqr_outliers(window[cols_in_window])
    checks.append({"check": "9. Extreme months",
                   "status": "info",
                   "detail": f"{len(outliers)} months fall outside the 1.5 x IQR whiskers. They are flagged, not deleted: "
                             "real events like earnings surprises are information (Lecture 14)."})

    # Check 10: excess returns built correctly
    excess = window[cols_in_window].sub(window["rf"], axis=0)
    checks.append({"check": "10. Excess returns built correctly",
                   "status": "pass",
                   "detail": "Excess return = stock return - rf. mktrf is already an excess return, so rf is not subtracted again."})

    order = {c["check"]: int(c["check"].split(".")[0]) for c in checks}
    result.update({
        "tickers": keep, "returns": returns, "window": window, "excess": excess, "outliers": outliers,
        "missing_tickers": missing, "short_history": short,
        "counts": {"return_months": len(returns), "merged_months": len(combined), "window_months": len(window)},
        "as_of": {"prices": str(last_return), "factors": str(last_factor),
                  "window_start": str(window.index.min()) if len(window) else "",
                  "window_end": str(window.index.max()) if len(window) else ""},
        "checks": sorted(checks, key=lambda c: order[c["check"]]),
    })
    return result
