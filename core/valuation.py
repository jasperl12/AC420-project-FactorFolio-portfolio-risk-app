"""
FactorFolio fair value: DCF and peer comparables (Build Guide Step 6; outline section 5.6).

Adapted from the AC420 class notebooks (Prof. Wysocki):
- Lecture 15 & 16 "Stock Valuation Using YFinance": eps = stock.eps_trend,
  eps1 = eps.at['0y', 'current'], eps2 = eps.at['+1y', 'current'], re = rf + beta*erp,
  g = eps2/eps1 - 1 and the three-part price1 formula (PV of year 1 + PV of year 2 +
  PV of the growing perpetuity from year 3, discounted two years).
- Lecture 16 & 17 "Stock Valuation Using Comparables": loop over the tickers with try/except,
  info.get(...) instead of info[...], a table of peer P/E ratios, predicted price = peer P/E x EPS,
  and the if / elif / else 5% band.
Fixes from the outline (section 10): the risk-free rate is today's 30-year Treasury yield, not
the notebook's March 2025 5.3%; beta is our own regression beta (Yahoo's is only a cross-check);
peers use the median P/E (one outlier peer flipped the AMZN call); every P/E is price / next-year
EPS for the stock and its peers alike; and eps1 is always defined before it is used.
Labels say "looks underpriced / fairly priced / overpriced", never BUY or SELL.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from core import yahoo
from core.projections import ERP

try:
    import streamlit as st
    cache_data = st.cache_data(ttl=86400, show_spinner=False)  # keep Yahoo data for one day
except ImportError:  # lets these functions run in Colab without Streamlit
    def cache_data(func):
        return func

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
G_LT = {"Bear": 0.02, "Base": 0.03, "Bull": 0.04}   # long-term growth by scenario (outline 5.6)
BAND = 0.05          # the notebook's 5% band
MIN_PEERS = 3        # check 19: fewer than 3 usable peers means no comps value
MAX_PEERS = 6        # keeps the number of Yahoo requests small
HIGH_GROWTH = 0.25   # check 18: short-term growth above 25% is flagged (first guess)
BETA_GAP = 0.5       # check 13: our beta vs. Yahoo's beta
SENSITIVE_GAP = 0.02 # cost of equity within 2 points of long-term growth: the value is very sensitive


# ---------- The class formulas ----------

def cost_of_equity(rf, beta, erp=ERP):
    """CAPM cost of equity: re = Rf + beta x ERP (Lecture 15)."""
    return rf + beta * erp


def dcf_price(eps1, eps2, re, g_lt=0.03, g=None):
    """The Lecture 15 & 16 notebook's price1 formula.

    Year 1 and year 2 EPS are discounted one and two years. Year 3 EPS = eps2 x (1 + g), where
    g = eps2/eps1 - 1 unless given, starts a growing perpetuity at g_lt; that perpetuity's value
    sits at year 2, so it is discounted two years.
    """
    if g is None:
        g = eps2 / eps1 - 1
    price1 = (eps1 / (1 + re)
              + eps2 / ((1 + re) * (1 + re))
              + eps2 * (1 + g) / ((1 + re) * (1 + re) * (re - g_lt)))
    return price1


def dcf_problems(eps1, eps2, re, g_lt):
    """Check 14: reasons the DCF can't be used for a stock (an empty list means it can)."""
    if pd.isna(eps1) or pd.isna(eps2):
        return ["no analyst EPS forecasts"]
    problems = []
    if eps1 <= 0 or eps2 <= 0:
        problems.append("EPS forecast at or below zero")
    if re <= g_lt:
        problems.append("cost of equity at or below long-term growth")
    return problems


def peer_pe_summary(peer_pes):
    """Median and mean of the peers' P/E ratios after dropping missing or negative ones (checks 19 and 20)."""
    pes = pd.Series(peer_pes, dtype="float64")
    pes = pes[pes.notna() & (pes > 0)]
    return {"n_peers": int(len(pes)), "median_pe": pes.median(), "mean_pe": pes.mean()}


def comps_price(peer_pes, eps):
    """Lecture 16 comparables: predicted price = peer P/E x the stock's EPS, with the median and the mean."""
    s = peer_pe_summary(peer_pes)
    s["price_median"] = s["median_pe"] * eps
    s["price_mean"] = s["mean_pe"] * eps
    return s


def price_label(price, low, high, band=BAND):
    """The notebook's 5% band, applied to a fair-value range from low to high."""
    if price < low * (1 - band):
        return "looks underpriced"
    elif price > high * (1 + band):
        return "looks overpriced"
    else:
        return "fairly priced"


def implied_return(eps1, price, g_lt):
    """Market-implied return E/P + g: the return that makes today's price fair (Lecture 16)."""
    return eps1 / price + g_lt


# ---------- Peers from the S&P 100 list ----------

@cache_data
def load_universe():
    """S&P 100 tickers, sectors and industries (Step 1) with the class-data CAPM beta where we have one."""
    tickers = pd.read_csv(DATA_DIR / "sp100_tickers.csv")
    betas = pd.read_csv(DATA_DIR / "sp100_estimates.csv")[["ticker", "beta"]]
    return tickers.merge(betas, on="ticker", how="left")


def pick_peers(ticker, industry, sector, beta, universe=None, max_peers=MAX_PEERS):
    """Peers = other S&P 100 stocks in the same Yahoo industry; fewer than 3 -> the same sector.

    Lecture 16: comparables should have the same risk, so when there are more candidates than
    max_peers, keep the ones whose beta is closest to the stock's beta. In the sector fallback,
    stocks from the same industry still come first.
    Returns (list of peer tickers, 'industry' or 'sector').
    """
    u = load_universe() if universe is None else universe
    u = u[u["ticker"] != ticker]
    candidates, basis = u[u["industry"] == industry], "industry"
    if len(candidates) < MIN_PEERS:
        candidates, basis = u[u["sector"] == sector], "sector"
    candidates = candidates.assign(other_industry=(candidates["industry"] != industry).astype(int),
                                   gap=(candidates["beta"] - beta).abs())
    candidates = candidates.sort_values(["other_industry", "gap", "ticker"], na_position="last")
    return candidates["ticker"].head(max_peers).tolist(), basis


# ---------- Yahoo Finance downloads ----------
# Analyst forecasts and company info come from core/yahoo.py, which handles Yahoo refusing
# requests from Streamlit Cloud (backup files, no caching of failures). Prices for all stocks and
# peers come from one batched download, cached for a day.

@cache_data
def _last_prices(tickers):
    raw = yf.download(list(tickers), period="5d", interval="1d", auto_adjust=False,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise ValueError("no price data")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.ffill()
    return close.iloc[-1].dropna().to_dict(), close.index[-1].strftime("%Y-%m-%d")


def last_prices(tickers):
    """Latest closing price for many tickers in one download. Returns (prices, as_of, error)."""
    try:
        prices, as_of = _last_prices(tuple(sorted(set(tickers))))
        return prices, as_of, None
    except Exception as e:
        return {}, None, f"Couldn't download current prices from Yahoo Finance ({e})."


# ---------- Putting it together for a portfolio ----------

def value_portfolio(stocks, weights, rf, scenario="Base", erp=ERP, g_lt_base=0.03):
    """DCF, comps and the fair-value range for every stock in the portfolio.

    stocks: the 'stocks' table from core.models.analyze_portfolio (index = ticker, has 'beta').
    weights: fractions that add to 1 (result['weights']).
    g_lt_base: long-term growth in the base case; bear is 1 point lower and bull 1 point higher.
    Returns a dict with 'table' (one row per stock), 'peers' (one table per stock),
    'eps_tables', 'portfolio' (the overall label) and 'error'.
    """
    g_lt = {"Bear": g_lt_base - 0.01, "Base": g_lt_base, "Bull": g_lt_base + 0.01}
    universe = load_universe()
    lookup = universe.set_index("ticker")
    tickers = list(stocks.index)

    info, peers, basis = yahoo.lookup_many(yahoo.company_info, tickers), {}, {}
    for t in tickers:
        if t in lookup.index:   # use the same labels as the peers' file when we can
            info[t]["sector"], info[t]["industry"] = lookup.at[t, "sector"], lookup.at[t, "industry"]
        peers[t], basis[t] = pick_peers(t, info[t]["industry"], info[t]["sector"], stocks.at[t, "beta"], universe)

    everyone = sorted(set(tickers) | {p for lst in peers.values() for p in lst})
    prices, as_of, error = last_prices(everyone)
    if error:
        return {"error": error, "table": pd.DataFrame(), "peers": {}, "eps_tables": {}, "portfolio": None}
    found = yahoo.lookup_many(yahoo.eps_forecasts, tickers + [p for p in everyone if p not in tickers])
    eps = {t: (f["eps1"], f["eps2"]) for t, f in found.items()}
    source = {t: f["source"] for t, f in found.items()}
    eps_tables = {t: found[t]["table"] for t in tickers}
    status = {"live": [t for t in everyone if source[t] == "live"],
              "snapshot": [t for t in everyone if source[t] == "snapshot"],
              "missing": [t for t in everyone if source[t] is None],
              "own_missing": [t for t in tickers if source[t] is None],
              "snapshot_date": yahoo.snapshot_date(), "yahoo_message": yahoo.status()["message"]}
    if all(source[t] is None for t in tickers):
        return {"error": "Yahoo Finance isn't sending analyst EPS forecasts to this app right now, and there is "
                         "no saved backup for these stocks." + yahoo.said(), "table": pd.DataFrame(),
                "peers": {}, "eps_tables": {}, "portfolio": None, "forecasts": status}

    rows, peer_tables = [], {}
    for t in tickers:
        eps1, eps2 = eps[t]
        price = prices.get(t, float("nan"))
        beta = stocks.at[t, "beta"]
        re = cost_of_equity(rf, beta, erp)
        flags = []

        # DCF (Lecture 15 & 16) in all three scenarios
        g = eps2 / eps1 - 1 if pd.notna(eps1) and pd.notna(eps2) and eps1 > 0 else float("nan")
        dcf = {}
        for name, gl in g_lt.items():
            problems = dcf_problems(eps1, eps2, re, gl)
            dcf[name] = dcf_price(eps1, eps2, re, gl) if not problems else float("nan")
        problems = dcf_problems(eps1, eps2, re, g_lt[scenario])
        if source[t] is None and problems == ["no analyst EPS forecasts"]:
            problems = ["Yahoo didn't send analyst forecasts and there is no saved backup"]
        flags += problems
        if not problems and re - g_lt[scenario] < SENSITIVE_GAP:
            flags.append(f"cost of equity ({re:.1%}) is close to long-term growth ({g_lt[scenario]:.0%}), "
                         "so the DCF value is very sensitive")
        dcf_g0 = float("nan")
        if not problems and g < 0:      # check 18: the Lecture 16 sensitivity check with g = 0%
            dcf_g0 = dcf_price(eps1, eps2, re, g_lt[scenario], g=0)
            flags.append(f"EPS falls {abs(g):.0%} next year; value with growth set to 0% is ${dcf_g0:,.2f}")
        elif g > HIGH_GROWTH:
            flags.append(f"short-term growth of {g:.0%} is very high")
        yb = info[t]["yahoo_beta"]
        if yb is not None and abs(yb - beta) > BETA_GAP:
            flags.append(f"our beta {beta:.2f} vs. Yahoo's {yb:.2f}")

        # Peer comparables (Lecture 16 & 17): P/E = price / next-year EPS for everyone
        data_list = []
        for p in peers[t]:
            p_eps2 = eps[p][1]
            p_price = prices.get(p, float("nan"))
            pe = p_price / p_eps2 if pd.notna(p_eps2) and p_eps2 > 0 else float("nan")
            data_list.append({"Ticker": p, "Company Name": lookup.at[p, "name"],
                              "Industry": lookup.at[p, "industry"], "Price": p_price,
                              "EPS next year": p_eps2, "P/E": pe})
        peer_df = pd.DataFrame(data_list, columns=["Ticker", "Company Name", "Industry", "Price",
                                                   "EPS next year", "P/E"])
        peer_tables[t] = peer_df
        comps = comps_price(peer_df["P/E"], eps2)
        if comps["n_peers"] < MIN_PEERS or pd.isna(eps2) or eps2 <= 0:
            comps["price_median"] = comps["price_mean"] = float("nan")
            if comps["n_peers"] == 0:
                flags.append("no peers with forecasts, so no comps value")
            elif comps["n_peers"] < MIN_PEERS:
                flags.append(f"only {comps['n_peers']} peers with forecasts, so no comps value")

        # Fair-value range: from the lower to the higher of the DCF and comps values
        values = [v for v in (dcf[scenario], comps["price_median"]) if pd.notna(v)]
        if values and pd.notna(price):
            low, high = min(values), max(values)
            mid = (low + high) / 2
            label = price_label(price, low, high)
            upside = mid / price - 1
        else:
            low = high = mid = upside = float("nan")
            label = "left out"
        implied = implied_return(eps1, price, g_lt[scenario]) if pd.notna(eps1) and eps1 > 0 else float("nan")

        rows.append({
            "ticker": t, "name": info[t]["name"], "weight": weights.get(t, 0), "sector": info[t]["sector"],
            "industry": info[t]["industry"], "price": price, "eps1": eps1, "eps2": eps2, "g": g,
            "beta": beta, "yahoo_beta": yb, "re": re,
            "dcf_bear": dcf["Bear"], "dcf_base": dcf["Base"], "dcf_bull": dcf["Bull"], "dcf": dcf[scenario],
            "dcf_g0": dcf_g0, "own_pe": price / eps2 if pd.notna(eps2) and eps2 > 0 else float("nan"),
            "yahoo_forward_pe": info[t]["yahoo_forward_pe"], "peer_basis": basis[t],
            "n_peers": comps["n_peers"], "median_pe": comps["median_pe"], "mean_pe": comps["mean_pe"],
            "comps_median": comps["price_median"], "comps_mean": comps["price_mean"],
            "low": low, "high": high, "mid": mid, "upside": upside, "label": label,
            "implied_return": implied, "flags": "; ".join(flags),
        })
    table = pd.DataFrame(rows).set_index("ticker")
    return {"error": None, "table": table, "peers": peer_tables, "eps_tables": eps_tables,
            "portfolio": portfolio_label(table), "prices_as_of": as_of, "scenario": scenario,
            "g_lt": g_lt, "rf": rf, "erp": erp, "forecasts": status, "source": source}


def forecast_note(val):
    """A plain-English note when some forecasts are not live. Returns (kind, text) or None.

    kind is 'warning' when one of your own stocks had to be left out, otherwise 'info'.
    """
    status = val.get("forecasts") or {}
    snap, own_missing = status.get("snapshot", []), status.get("own_missing", [])
    if not snap and not own_missing:
        return None
    parts = []
    if snap:
        parts.append(f"Yahoo Finance isn't sending live analyst forecasts right now{yahoo.said(status.get('yahoo_message'))}, so these values "
                     f"use the saved backup from {status.get('snapshot_date')}.")
    if own_missing:
        lead = "" if snap else f"Yahoo Finance isn't sending analyst forecasts right now{yahoo.said(status.get('yahoo_message'))}. "
        parts.append(f"{lead}No forecasts for {', '.join(own_missing)} (not in the saved backup either), so "
                     f"{'it is' if len(own_missing) == 1 else 'they are'} left out. Try again in about 10 minutes.")
    return ("warning" if own_missing else "info"), " ".join(parts)


def portfolio_label(table):
    """Weighted average upside to each stock's range midpoint, put through the same 5% band."""
    valued = table[table["label"] != "left out"]
    left_out = table[table["label"] == "left out"]
    if valued.empty:
        return {"label": "left out", "upside": float("nan"), "n_valued": 0, "left_out": list(left_out.index)}
    w = valued["weight"] / valued["weight"].sum()
    upside = float((w * valued["upside"]).sum())
    # Treat the portfolio as $1 of stock whose fair value is $(1 + upside)
    label = price_label(1.0, 1 + upside, 1 + upside)
    return {"label": label, "upside": upside, "n_valued": len(valued), "left_out": list(left_out.index)}


def explain(summary, table):
    """Plain-English fair-value message for the homepage (outline section 3)."""
    label, upside = summary["label"], summary["upside"]
    if label == "left out":
        text = ("None of your stocks could be valued: they need positive analyst EPS forecasts "
                "and current prices.")
    elif label == "fairly priced":
        text = ("Your stocks look fairly priced: on average, today's prices sit close to the middle of the range "
                "from our DCF estimate to what similar companies trade at.")
    else:
        direction = "above" if upside > 0 else "below"
        text = (f"Your stocks {label.replace('looks', 'look')}: on average, the middle of their fair-value ranges "
                f"is about {abs(upside):.0%} {direction} today's prices. Each range runs from our DCF estimate "
                "to what similar companies trade at.")
    text += " This is a check, not a buy signal."
    if summary["left_out"]:
        reasons = []
        for t in summary["left_out"]:
            why = table.at[t, "flags"].split(";")[0] if table.at[t, "flags"] else "missing data"
            reasons.append(f"{t} ({why})")
        text += " Left out: " + ", ".join(reasons) + "."
    return text
