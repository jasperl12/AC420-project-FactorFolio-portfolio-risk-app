"""
Yahoo Finance lookups that need a session token: analyst EPS forecasts (eps_trend), company info
(info) and financial statements (balance_sheet, financials).

Yahoo often refuses these requests from shared cloud servers like Streamlit Cloud's, and yfinance
hides the refusal: it logs the error and returns an empty table. So this module:
  1. treats an empty answer as a failed download (it raises, so st.cache_data never caches it),
  2. keeps the error yfinance logged, to show the user what Yahoo said,
  3. pauses between requests, and stops asking for 10 minutes after 3 refusals in a row,
  4. falls back to backup files saved in Colab by notebooks/yahoo_backup.ipynb, with their date shown.
Successful answers are cached for a day. Prices come from a different part of Yahoo that does not
need the token, so core/data.py downloads them directly.

Patterns from the AC420 class notebooks: yf.Ticker(t).eps_trend and .at['0y', 'current']
(Lecture 15 & 16), info.get(...) inside try/except (Lecture 16 & 17), and ticker.financials with
label-based .loc rows (Lecture 3).
"""
import logging
import re
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    cache_data = st.cache_data(ttl=86400, show_spinner=False)  # keep Yahoo answers for one day
except ImportError:  # lets these functions run in Colab without Streamlit
    def cache_data(func):
        return func

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_FILE = DATA_DIR / "eps_snapshot.csv"               # analyst EPS forecasts and company info
FINANCIALS_FILE = DATA_DIR / "financials_snapshot.csv"      # balance sheet and income statement items
LIVE_PAUSE = 0.25    # seconds between Yahoo requests, like the Step 1 Colab cell
RETRY_AFTER = 600    # after Yahoo refuses, wait 10 minutes before asking it again
STOP_AFTER = 3       # three refusals in a row: stop asking Yahoo for now and use the backup

# Yahoo's row labels for the Altman Z-score inputs (Lecture 3: check the labels before using .loc)
BALANCE_ROWS = {"current_assets": ["Current Assets"],
                "current_liabilities": ["Current Liabilities"],
                "total_assets": ["Total Assets"],
                "retained_earnings": ["Retained Earnings"],
                "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
                "shares": ["Ordinary Shares Number", "Share Issued"],
                "stockholders_equity": ["Stockholders Equity", "Common Stock Equity"]}
INCOME_ROWS = {"ebit": ["EBIT", "Operating Income"],
               "revenue": ["Total Revenue", "Operating Revenue"]}


# ---------- Keeping track of what Yahoo said ----------

class _YahooMessages(logging.Handler):
    """Keeps the most recent warning or error that yfinance logged."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.last = None

    def emit(self, record):
        self.last = record.getMessage()[:300]


_yf_logger = logging.getLogger("yfinance")
_messages = next((h for h in _yf_logger.handlers if type(h).__name__ == "_YahooMessages"), None)
if _messages is None:          # add it once, even when Streamlit reloads this file
    _messages = _YahooMessages()
    _yf_logger.addHandler(_messages)

# Shared by every visitor of the app (like st.cache_data)
_status = {"blocked_until": 0.0, "message": None}
_live_ok = {}     # key -> time a live answer was saved in the one-day cache
_failed_at = {}   # key -> time Yahoo last refused it


def _can_ask_yahoo(key):
    """False while Yahoo is refusing us, unless the answer is already in the cache (no new request)."""
    now = time.time()
    if now - _live_ok.get(key, -1e12) < 86400:
        return True
    if now < _status["blocked_until"]:
        return False
    return now - _failed_at.get(key, -1e12) > RETRY_AFTER


def _record(key, ok, message=None):
    if ok:
        _live_ok.setdefault(key, time.time())
    else:
        _failed_at[key] = time.time()
        _status["message"] = message


def status():
    """What Yahoo said the last time it refused a request, and whether we are waiting before asking again."""
    return {"blocked": time.time() < _status["blocked_until"], "message": _status["message"]}


def said(message=None):
    """A short version of what Yahoo said, e.g. ' (Yahoo said: 401 Unauthorized, Invalid Crumb)'."""
    msg = message if message is not None else _status["message"]
    if not msg:
        return ""
    code = re.search(r"HTTP Error (\d+)", msg)
    words = re.findall(r'"(?:code|description)":"([^"]+)"', msg)
    short = ", ".join(([code.group(1) + " " + words[0]] + words[1:]) if code and words else words) or msg[:80]
    return f" (Yahoo said: {short})"


def lookup_many(func, tickers):
    """Run one lookup (eps_forecasts, company_info or financials) for many tickers.

    After 3 refusals in a row, stop asking Yahoo for 10 minutes and use the backup for the rest.
    """
    results, refused = {}, 0
    for t in tickers:
        r = func(t, allow_live=refused < STOP_AFTER)
        results[t] = r
        if r["source"] == "live":
            refused = 0
        elif r["error"]:
            refused += 1
            if refused == STOP_AFTER:
                _status["blocked_until"] = time.time() + RETRY_AFTER
    return results


# ---------- Backup files (saved in Colab) ----------

@cache_data
def load_snapshot():
    """Backup analyst forecasts and company info, one row per ticker."""
    cols = ["ticker", "name", "sector", "industry", "eps_0y", "eps_1y", "yahoo_beta", "forward_pe", "as_of"]
    if not SNAPSHOT_FILE.exists():
        return pd.DataFrame(columns=cols).set_index("ticker")
    return pd.read_csv(SNAPSHOT_FILE).drop_duplicates("ticker").set_index("ticker")


def snapshot_date():
    snap = load_snapshot()
    return None if snap.empty else str(snap["as_of"].max())


@cache_data
def load_financials_snapshot():
    """Backup balance sheet and income statement items, one row per ticker (latest fiscal year)."""
    cols = ["ticker", "fiscal_year_end", *BALANCE_ROWS, *INCOME_ROWS, "price", "as_of"]
    if not FINANCIALS_FILE.exists():
        return pd.DataFrame(columns=cols).set_index("ticker")
    return pd.read_csv(FINANCIALS_FILE).drop_duplicates("ticker").set_index("ticker")


def financials_snapshot_date():
    snap = load_financials_snapshot()
    return None if snap.empty else str(snap["as_of"].max())


# ---------- Analyst EPS forecasts (Lecture 15 & 16) ----------

@cache_data
def _eps_trend(ticker):
    """Live analyst EPS table. Raises when Yahoo sends nothing, so a failure is never cached."""
    _messages.last = None
    for attempt in range(2):                      # one retry after a short wait
        time.sleep(LIVE_PAUSE if attempt == 0 else 1.0)
        eps = yf.Ticker(ticker).eps_trend
        if eps is not None and not eps.empty and "0y" in eps.index and "+1y" in eps.index:
            return eps
    raise LookupError(_messages.last or "Yahoo Finance sent no analyst EPS table")


def eps_forecasts(ticker, allow_live=True):
    """Analyst EPS for one ticker: live from Yahoo, else from the backup, else missing.

    Returns a dict with eps1 ('0y'), eps2 ('+1y'), table (the full live eps_trend, or None),
    source ('live', 'snapshot' or None) and error (what went wrong with the live request, if anything).
    """
    error = None
    key = f"eps:{ticker}"
    if allow_live and _can_ask_yahoo(key):
        try:
            eps = _eps_trend(ticker)
            _record(key, True)
            return {"eps1": float(eps.at["0y", "current"]), "eps2": float(eps.at["+1y", "current"]),
                    "table": eps, "source": "live", "error": None}
        except Exception as e:   # Yahoo refused, rate limit, network problem or no coverage
            error = str(e)
            _record(key, False, error)
    snap = load_snapshot()
    if ticker in snap.index and pd.notna(snap.at[ticker, "eps_0y"]) and pd.notna(snap.at[ticker, "eps_1y"]):
        return {"eps1": float(snap.at[ticker, "eps_0y"]), "eps2": float(snap.at[ticker, "eps_1y"]),
                "table": None, "source": "snapshot", "error": error}
    return {"eps1": float("nan"), "eps2": float("nan"), "table": None, "source": None,
            "error": error or _status["message"]}


# ---------- Company info (Lecture 16 & 17) ----------

@cache_data
def _company_info(ticker):
    """Live Yahoo info. Raises when Yahoo sends an empty answer, so a failure is never cached."""
    _messages.last = None
    time.sleep(LIVE_PAUSE)
    info = yf.Ticker(ticker).info or {}
    if not info.get("longName") and not info.get("shortName"):
        raise LookupError(_messages.last or "Yahoo Finance sent no company info")
    return {"name": info.get("longName", ticker), "sector": info.get("sector"),
            "industry": info.get("industry"), "yahoo_beta": info.get("beta"),
            "yahoo_forward_pe": info.get("forwardPE")}


def company_info(ticker, allow_live=True):
    """Name, sector, industry and Yahoo's beta and forward P/E (cross-checks): live, else the backup."""
    error = None
    key = f"info:{ticker}"
    if allow_live and _can_ask_yahoo(key):
        try:
            result = _company_info(ticker)
            _record(key, True)
            return {**result, "source": "live", "error": None}
        except Exception as e:
            error = str(e)
            _record(key, False, error)
    snap = load_snapshot()
    if ticker in snap.index:
        r = snap.loc[ticker]
        clean = lambda x: None if pd.isna(x) else x
        return {"name": clean(r["name"]) or ticker, "sector": clean(r["sector"]), "industry": clean(r["industry"]),
                "yahoo_beta": clean(r["yahoo_beta"]), "yahoo_forward_pe": clean(r["forward_pe"]),
                "source": "snapshot", "error": error}
    return {"name": ticker, "sector": None, "industry": None, "yahoo_beta": None, "yahoo_forward_pe": None,
            "source": None, "error": error}


# ---------- Financial statements for the Altman Z-score (Lecture 3 and Lecture 10) ----------

def statement_items(balance_sheet, income_statement):
    """Pull the Altman Z inputs from the latest fiscal year (the first column) of Yahoo's statements.

    Like the Lecture 3 notebook's df.loc[selected_rows], but with .get so a missing or renamed
    row gives NaN instead of a KeyError (Yahoo's labels differ by company).
    """
    bs, inc = balance_sheet.iloc[:, 0], income_statement.iloc[:, 0]
    items = {"fiscal_year_end": pd.Timestamp(balance_sheet.columns[0]).strftime("%Y-%m-%d")}
    for name, labels in BALANCE_ROWS.items():
        items[name] = next((float(bs.get(l)) for l in labels if pd.notna(bs.get(l))), float("nan"))
    for name, labels in INCOME_ROWS.items():
        items[name] = next((float(inc.get(l)) for l in labels if pd.notna(inc.get(l))), float("nan"))
    return items


@cache_data
def _statements(ticker):
    """Live balance sheet and income statement items. Raises on an empty answer (never cached)."""
    _messages.last = None
    time.sleep(LIVE_PAUSE)
    stock = yf.Ticker(ticker)
    bs, inc = stock.balance_sheet, stock.financials
    if bs is None or bs.empty or inc is None or inc.empty:
        raise LookupError(_messages.last or "Yahoo Finance sent no financial statements")
    return statement_items(bs, inc)


def financials(ticker, allow_live=True):
    """Altman Z inputs for one ticker: live from Yahoo, else from the backup, else missing."""
    error = None
    key = f"fin:{ticker}"
    if allow_live and _can_ask_yahoo(key):
        try:
            items = _statements(ticker)
            _record(key, True)
            return {**items, "source": "live", "error": None}
        except Exception as e:
            error = str(e)
            _record(key, False, error)
    snap = load_financials_snapshot()
    if ticker in snap.index:
        r = snap.loc[ticker]
        items = {"fiscal_year_end": str(r["fiscal_year_end"])}
        items.update({k: float(r[k]) if pd.notna(r[k]) else float("nan") for k in [*BALANCE_ROWS, *INCOME_ROWS]})
        return {**items, "source": "snapshot", "error": error}
    return {"fiscal_year_end": None, **{k: float("nan") for k in [*BALANCE_ROWS, *INCOME_ROWS]},
            "source": None, "error": error or _status["message"]}


def backup_note(sources, what, backup_date):
    """One sentence when some answers came from the backup, e.g. for 'financial statements'."""
    used = [t for t, src in sources.items() if src == "snapshot"]
    if not used:
        return None
    return (f"Yahoo Finance isn't sending live {what} right now{said()}, so {', '.join(used)} "
            f"{'uses' if len(used) == 1 else 'use'} the saved backup from {backup_date}.")
