"""
FactorFolio bear, base and bull projections (Build Guide Step 5; outline section 5.5).

Market return for the year in each scenario:
  - Bear: 25th percentile of annual U.S. market returns 1927 to 2025 (data/market_scenarios.csv)
  - Base: risk-free rate + equity risk premium (Lecture 15: 30-year Treasury yield + 5.7%)
  - Bull: 75th percentile of annual U.S. market returns 1927 to 2025
The portfolio's expected return comes from the CAPM prediction form (Lecture 14):
  E(R_p) = Rf + beta x (market return - Rf)
SPY's return in each scenario is the market return itself.
Dollar values grow year by year: V_t = V_0 x (1 + r_1)(1 + r_2)...(1 + r_t), the Lecture 15
future value formula FV = PV x (1 + r) repeated once per year.
Default path: the scenario return in year 1, then the base return in every later year.
Historical alpha is left out, because Lecture 12 warns past alpha may not repeat.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    import streamlit as st
    cache_data = st.cache_data(ttl=86400, show_spinner=False)  # keep the yield for one day
except ImportError:  # lets these functions run in Colab without Streamlit
    def cache_data(func):
        return func

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ERP = 0.057            # equity risk premium from the Lecture 15 notebook
DEFAULT_RF = 0.045     # only used if today's 30-year Treasury yield can't be downloaded
SCENARIOS = ["Bear", "Base", "Bull"]
SCENARIO_HELP = {
    "Bear": "A bad year for stocks (1 in 4 years since 1927 were this bad or worse)",
    "Base": "A normal year: the risk-free rate plus the usual reward for owning stocks",
    "Bull": "A strong year for stocks (1 in 4 years since 1927 were this good or better)",
}


def load_scenarios():
    """Bear and bull market returns saved by the Step 1 training notebook (Ken French data)."""
    df = pd.read_csv(DATA_DIR / "market_scenarios.csv")
    return dict(zip(df["name"], df["value"]))


def risk_free_rate():
    """Today's 30-year Treasury yield from Yahoo Finance (^TYX), as a decimal.

    Yahoo posts ^TYX in percent (4.7 means 4.7%), so divide by 100 (check 14: rates as decimals).
    If the download fails or the number looks wrong, fall back to DEFAULT_RF and say so.
    """
    try:
        rf, as_of = _download_tyx()
    except Exception:  # network problems, Yahoo rate limits, no data
        return {"rf": DEFAULT_RF, "as_of": None, "ok": False,
                "note": f"Couldn't download today's 30-year Treasury yield from Yahoo Finance, so the projections use {DEFAULT_RF:.1%}."}
    if not 0 < rf < 0.15:
        return {"rf": DEFAULT_RF, "as_of": as_of, "ok": False,
                "note": f"The downloaded 30-year Treasury yield ({rf:.2%}) looks wrong, so the projections use {DEFAULT_RF:.1%}."}
    return {"rf": rf, "as_of": as_of, "ok": True,
            "note": f"30-year Treasury yield {rf:.2%} on {as_of} (Yahoo Finance ^TYX)."}


@cache_data
def _download_tyx():
    """Last closing value of ^TYX. Raises an error instead of returning bad data, so failures aren't cached."""
    tyx = yf.Ticker("^TYX").history(period="1mo")
    close = tyx["Close"].dropna()
    if close.empty:
        raise ValueError("no ^TYX data")
    return float(close.iloc[-1]) / 100, close.index[-1].strftime("%Y-%m-%d")


def market_returns(rf, erp=ERP, scenarios=None):
    """Market return for one year in each scenario."""
    s = load_scenarios() if scenarios is None else scenarios
    return {"Bear": s["bear_market_return"], "Base": rf + erp, "Bull": s["bull_market_return"]}


def expected_return(beta, market_return, rf):
    """CAPM prediction form: E(R) = Rf + beta x (market return - Rf)."""
    return rf + beta * (market_return - rf)


def growth_path(amount, first_year_return, later_return, years):
    """Dollar value at the end of each year: year 0 is today's amount."""
    values = [amount]
    for year in range(1, years + 1):
        r = first_year_return if year == 1 else later_return
        values.append(values[-1] * (1 + r))
    return values


def ff3_extra(s, h, scenarios=None):
    """Fama-French add-on to the yearly return: s x average SMB + h x average HML (outline 5.5)."""
    sc = load_scenarios() if scenarios is None else scenarios
    return s * sc["avg_smb"] + h * sc["avg_hml"]


def project(amount, years, beta, rf, scenario, erp=ERP, every_year=False, scenarios=None, extra=0.0):
    """Portfolio vs. SPY for one scenario.

    every_year=False: scenario return in year 1, then the base return (the outline's default).
    every_year=True: the scenario return in every year.
    extra: a yearly return added to the portfolio only (the FF3 size and value add-on, or
    historical alpha x 12 when the user turns it on). 0 on the homepage.
    """
    market = market_returns(rf, erp, scenarios)
    port_first = expected_return(beta, market[scenario], rf) + extra
    port_later = port_first if every_year else expected_return(beta, market["Base"], rf) + extra
    spy_first = market[scenario]
    spy_later = spy_first if every_year else market["Base"]

    path = pd.DataFrame({
        "Year": range(0, years + 1),
        "Your portfolio": growth_path(amount, port_first, port_later, years),
        "SPY": growth_path(amount, spy_first, spy_later, years),
    })
    final = path.iloc[-1]
    return {
        "scenario": scenario, "amount": amount, "years": years, "beta": beta, "rf": rf, "erp": erp,
        "market": market, "every_year": every_year, "extra": extra,
        "portfolio_return_first": port_first, "portfolio_return_later": port_later,
        "spy_return_first": spy_first, "spy_return_later": spy_later,
        "path": path,
        "portfolio_value": final["Your portfolio"], "spy_value": final["SPY"],
        "difference": final["Your portfolio"] - final["SPY"],
        "difference_pct": final["Your portfolio"] / final["SPY"] - 1,
        "portfolio_year1": path.loc[min(1, years), "Your portfolio"],
        "spy_year1": path.loc[min(1, years), "SPY"],
    }


def assumptions_table(proj):
    """One row per scenario: the year-1 market, portfolio and SPY returns."""
    rows = []
    for name in SCENARIOS:
        m = proj["market"][name]
        rows.append({"Scenario": name, "Market return": f"{m:.1%}",
                     "Your portfolio": f"{expected_return(proj['beta'], m, proj['rf']) + proj['extra']:.1%}",
                     "SPY": f"{m:.1%}"})
    return pd.DataFrame(rows)


def money(x):
    return f"${x:,.0f}"


def explain(proj):
    """Plain-English sentences for the homepage (outline section 3, scenario template)."""
    amount, years, name = proj["amount"], proj["years"], proj["scenario"]
    p1, s1 = proj["portfolio_year1"], proj["spy_year1"]
    if name == "Base":
        text = (f"In a normal year, your portfolio is expected to earn about {proj['portfolio_return_first']:.1%}, "
                f"vs. {proj['spy_return_first']:.1%} for SPY.")
    else:
        if p1 < amount:
            verb = "fall to"
        else:
            verb = "grow to" if name == "Bull" else "still end the year at"
        label = "bear" if name == "Bear" else "bull"
        text = f"In a {label} market year, your {money(amount)} could {verb} about {money(p1)}, vs. {money(s1)} in SPY."
    if years > 1 and (name == "Base" or proj["every_year"]):
        text += f" After {years} years, it could be worth about {money(proj['portfolio_value'])}, vs. {money(proj['spy_value'])} in SPY."
    elif years > 1:
        text += (f" If markets go back to normal after that, it could be worth about {money(proj['portfolio_value'])} "
                 f"after {years} years, vs. {money(proj['spy_value'])} in SPY.")
    if abs(proj["beta"] - 1) < 0.1:
        text += (f" Because your beta is close to 1 ({proj['beta']:.2f}), the CAPM expects your portfolio to track SPY; "
                 "stocks with higher or lower betas are what pull the two lines apart.")
    return text
