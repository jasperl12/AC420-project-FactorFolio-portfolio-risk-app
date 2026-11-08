"""
FactorFolio suggestions engine (Build Guide Step 8; outline section 5.7).

For each S&P 100 stock you don't hold, build a test portfolio: your holdings scaled to 90% plus
that stock at 10%. Rescore it with the same risk score as the homepage, keep the stocks that move
the score into or toward your comfort band, and rank them by
  1. distance from the band after adding the stock (closest first),
  2. whether the stock adds a sector you don't own yet (diversification, Lecture 12),
  3. fair-value upside (the middle of the DCF to comps range vs. today's price, Lectures 15 and 16).

Speed: all candidates' monthly prices come from one batched yfinance download, cached for a day.
Each test portfolio needs only one CAPM regression, because its monthly return is simply
0.9 x your portfolio's return + 0.1 x the candidate's return (portfolio math from Lecture 12).
Fair values for candidates use the saved analyst-forecast backup, so there is no request per stock.
"""
from pathlib import Path

import pandas as pd

from core import data as fdata
from core import models, scoring, valuation, yahoo

try:
    import streamlit as st
    cache_data = st.cache_data(ttl=86400, show_spinner=False)
except ImportError:
    st = None

    def cache_data(func):
        return func

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANDIDATE_WEIGHT = 0.10    # outline: 3 stocks at 10% each
TOP_N = 3


@cache_data
def load_universe():
    """S&P 100 tickers with a sector (BK has none in sp100_tickers.csv, so it is skipped)."""
    u = pd.read_csv(DATA_DIR / "sp100_tickers.csv").dropna(subset=["sector"])
    return u.reset_index(drop=True)


@cache_data
def candidate_stats():
    """One cached download and regression run for every S&P 100 stock.

    Returns (stats table, window of monthly returns with the factor columns). Raises when the
    download fails, so a failure is never cached.
    """
    u = load_universe()
    ds = fdata.build_dataset(list(u["ticker"]))
    if ds["error"] or ds["window"].empty:
        raise RuntimeError(ds["error"] or "No price data for the S&P 100 stocks.")
    window = ds["window"]
    fin = yahoo.load_financials_snapshot()
    info = u.set_index("ticker")
    rows = []
    for t in ds["tickers"]:
        res = models.analyze_stock(window, t)
        price = ds["prices"][t].dropna().iloc[-1]
        zone = None
        if t in fin.index:   # Altman zone from the saved statements and today's price
            zone = models.altman_for_stock(fin.loc[t].to_dict(), price, info.at[t, "sector"],
                                           info.at[t, "industry"])["zone"]
        rows.append({"ticker": t, "name": info.at[t, "name"], "sector": info.at[t, "sector"],
                     "industry": info.at[t, "industry"], "price": price, "beta": res["beta"], "r2": res["r2"],
                     "s": res["s"], "h": res["h"], "alpha": res["alpha"], "alpha_p": res["alpha_p"],
                     "volatility": res["volatility"], "zone": zone,
                     "style": ("Small" if res["s"] > 0 else "Large") + " " + ("value" if res["h"] > 0 else "growth")})
    return pd.DataFrame(rows).set_index("ticker"), window


@cache_data
def candidate_values(rf, scenario="Base"):
    """Fair-value upside for every candidate from the saved forecasts (DCF and peer comps, Step 6)."""
    stats, _ = candidate_stats()
    g_lt = valuation.G_LT[scenario]
    universe = valuation.load_universe()
    eps = {t: yahoo.eps_forecasts(t, allow_live=False) for t in stats.index}
    pe = {t: stats.at[t, "price"] / e["eps2"] for t, e in eps.items() if pd.notna(e["eps2"]) and e["eps2"] > 0}
    rows = []
    for t in stats.index:
        eps1, eps2 = eps[t]["eps1"], eps[t]["eps2"]
        price, beta = stats.at[t, "price"], stats.at[t, "beta"]
        re = valuation.cost_of_equity(rf, beta)
        dcf = (valuation.dcf_price(eps1, eps2, re, g_lt)
               if not valuation.dcf_problems(eps1, eps2, re, g_lt) else float("nan"))
        peers, _ = valuation.pick_peers(t, stats.at[t, "industry"], stats.at[t, "sector"], beta, universe)
        comps = valuation.comps_price([pe.get(p, float("nan")) for p in peers], eps2)
        comp_value = comps["price_median"] if comps["n_peers"] >= valuation.MIN_PEERS and eps2 > 0 else float("nan")
        values = [v for v in (dcf, comp_value) if pd.notna(v)]
        mid = (min(values) + max(values)) / 2 if values else float("nan")
        label = valuation.price_label(price, min(values), max(values)) if values else "left out"
        rows.append({"ticker": t, "dcf": dcf, "comps": comp_value, "upside": mid / price - 1, "label": label})
    return pd.DataFrame(rows).set_index("ticker")


def band_distance(score, comfort):
    """How far a score is from the comfort band (0 when inside it)."""
    low, high = scoring.COMFORT_RANGES[comfort]
    if score < low:
        return low - score
    if score >= high and comfort != "Aggressive":
        return score - high
    return 0.0


def test_scores(port_returns, held_zones, comfort, held_sectors, rf, scenario="Base"):
    """Score every candidate as a 10% addition to the portfolio.

    port_returns: the portfolio's monthly return series (core.models.analyze_portfolio 'returns').
    held_zones: table with 'zone' and 'weight' for the current holdings (core.models.altman_table).
    Returns one row per candidate, best first, with the new score and why it ranks where it does.
    """
    stats, window = candidate_stats()
    values = candidate_values(round(rf, 4), scenario)
    held = set(held_zones.index)
    w_old = held_zones["weight"] / held_zones["weight"].sum()
    rows = []
    for t in stats.index:
        if t in held:
            continue
        both = pd.concat([port_returns.rename("P"), window[t].rename("C")], axis=1, join="inner")
        both = both.join(window[models.FF3_FACTORS + ["rf"]], how="inner").dropna()
        if len(both) < fdata.MIN_MONTHS:
            continue
        test = (1 - CANDIDATE_WEIGHT) * both["P"] + CANDIDATE_WEIGHT * both["C"]
        capm = models.run_capm(test - both["rf"], both["mktrf"])
        result = {"beta": capm["beta"], "volatility": test.std(), "firm_specific": 1 - capm["r2"]}
        zones = pd.concat([pd.DataFrame({"zone": held_zones["zone"], "weight": w_old * (1 - CANDIDATE_WEIGHT)}),
                           pd.DataFrame({"zone": [stats.at[t, "zone"]], "weight": [CANDIDATE_WEIGHT]}, index=[t])])
        new = scoring.risk_score(result, health_points=models.health_points(zones)["points"])
        rows.append({"ticker": t, "new_score": new["score"], "new_band": new["band"],
                     "distance": band_distance(new["score"], comfort),
                     "new_sector": stats.at[t, "sector"] not in held_sectors})
    out = pd.DataFrame(rows).set_index("ticker").join(stats).join(values)
    out = out.sort_values(["distance", "new_sector", "upside"], ascending=[True, False, False], na_position="last")
    return out


def suggest(table, score_now, comfort, top_n=TOP_N):
    """The top candidates that move the score into or toward the comfort band."""
    now = band_distance(score_now, comfort)
    if now == 0:
        return table.iloc[0:0]
    return table[table["distance"] < now].head(top_n)


def helps(table, score_now, comfort):
    """True for candidates that move the score toward the band (or keep it inside)."""
    now = band_distance(score_now, comfort)
    return (table["distance"] < now) | ((now == 0) & (table["distance"] == 0))


def add_to_holdings(tickers, weight_each=CANDIDATE_WEIGHT * 100):
    """Add stocks at 10% each and scale the current holdings down to make room (the Add buttons)."""
    holdings = st.session_state.get("holdings")
    if holdings is None:
        holdings = pd.DataFrame({"Ticker": list(fdata.DEFAULT_PORTFOLIO),
                                 "Weight (%)": [float(w) for w in fdata.DEFAULT_PORTFOLIO.values()]})
    current = {}
    for _, row in holdings.iterrows():
        raw = row.get("Ticker")
        t = fdata.clean_tickers("" if pd.isna(raw) else str(raw))
        w = pd.to_numeric(row.get("Weight (%)"), errors="coerce")
        if t and pd.notna(w) and w > 0:
            current[t[0]] = current.get(t[0], 0) + float(w)
    new = [t for t in tickers if t not in current]
    room = 100 - weight_each * len(new)
    total = sum(current.values()) or 1
    weights = {t: round(w / total * room, 1) for t, w in current.items()}
    weights.update({t: float(weight_each) for t in new})
    st.session_state.holdings = pd.DataFrame({"Ticker": list(weights), "Weight (%)": list(weights.values())})
    st.session_state.editor_base = st.session_state.holdings.copy()
    st.session_state.pop("holdings_editor", None)   # restart the homepage table with the new rows
    st.session_state.portfolio = weights
    st.session_state.just_added = new
