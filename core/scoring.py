"""
FactorFolio risk score (Build Guide Step 4; outline section 5.4).

The 0 to 100 score combines course metrics, each turned into a 0 to 100 percentile against
the S&P 100 stocks estimated offline in notebooks/training_wrds.ipynb (data/sp100_estimates.csv):
  - Market risk:      portfolio beta from the CAPM regression (Lecture 12)          weight 40%
  - Volatility:       standard deviation of monthly portfolio returns (Lectures 1-3) weight 25%
  - Diversification:  firm-specific share of risk, 1 - R-squared (Lectures 12-13)   weight 20%
  - Financial health: Altman Z-score zones (Lecture 10), added in Step 7             weight 15%
Until the financial health part exists, the other weights are rescaled to add to 100%.
Percentile = share of reference stocks below the value, the rank idea from Lecture 3.
"""
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data(show_spinner=False)
except ImportError:
    def cache_data(func):
        return func

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEIGHTS = {"market": 0.40, "volatility": 0.25, "diversification": 0.20, "health": 0.15}

# Score bands and the matching comfort levels (outline section 5.4)
COMFORT_RANGES = {"Conservative": (0, 35), "Moderate": (35, 65), "Aggressive": (65, 100)}
RISK_WORDS = {"Conservative": "low risk", "Moderate": "moderate risk", "Aggressive": "high risk"}
COMFORT_HELP = {
    "Conservative": "Steadier returns, even if growth is slower",
    "Moderate": "Normal ups and downs for solid growth",
    "Aggressive": "Big swings for a chance at bigger gains",
}


@cache_data
def load_reference():
    """CAPM estimates for the S&P 100 reference stocks (built offline from WRDS class data)."""
    return pd.read_csv(DATA_DIR / "sp100_estimates.csv")


def percentile_rank(values, x):
    """Share of reference values below x, from 0 to 100 (the Lecture 3 rank idea)."""
    values = pd.Series(values).dropna()
    return float((values < x).mean() * 100)


def band_for(score):
    """'Conservative' for 0 to 35, 'Moderate' for 35 to 65, 'Aggressive' for 65 to 100."""
    if score < 35:
        return "Conservative"
    if score < 65:
        return "Moderate"
    return "Aggressive"


def risk_score(result, reference=None, health_points=None):
    """0 to 100 risk score for one result from core.models.analyze_stock / analyze_portfolio.

    result needs the keys 'beta', 'volatility' and 'firm_specific'.
    health_points: 0 to 100 from the Altman Z-score zones (Step 7); None leaves that part out.
    """
    ref = load_reference() if reference is None else reference
    components = [
        {"key": "market", "name": "Market risk", "metric": "Beta", "value": result["beta"],
         "value_text": f"{result['beta']:.2f}",
         "score": percentile_rank(ref["beta"], result["beta"])},
        {"key": "volatility", "name": "Volatility", "metric": "Monthly standard deviation",
         "value": result["volatility"], "value_text": f"{result['volatility']:.1%}",
         "score": percentile_rank(ref["volatility"], result["volatility"])},
        {"key": "diversification", "name": "Diversification", "metric": "Firm-specific share (1 - R²)",
         "value": result["firm_specific"], "value_text": f"{result['firm_specific']:.0%}",
         "score": percentile_rank(ref["firm_specific"], result["firm_specific"])},
    ]
    if health_points is not None:
        components.append({"key": "health", "name": "Financial health", "metric": "Altman Z-score zones",
                           "value": health_points, "value_text": f"{health_points:.0f} points",
                           "score": float(health_points)})
    total_weight = sum(WEIGHTS[c["key"]] for c in components)
    for c in components:
        c["weight"] = WEIGHTS[c["key"]] / total_weight           # rescaled to add to 1
        c["points"] = c["score"] * c["weight"]
    score = sum(c["points"] for c in components)
    band = band_for(score)
    return {"score": score, "band": band, "risk_words": RISK_WORDS[band], "components": components}


def comfort_fit(score, comfort):
    """Compare the score with the chosen comfort level: 'fits', 'above' or 'below'."""
    low, high = COMFORT_RANGES[comfort]
    level = comfort.lower()
    if score < low:
        return {"status": "below",
                "message": f"This is safer than your {level} comfort level, so you could take on more risk."}
    if score >= high and comfort != "Aggressive":
        return {"status": "above",
                "message": f"This is riskier than your {level} comfort level."}
    return {"status": "fits", "message": f"This fits your {level} comfort level."}


def explain(result, scored):
    """Plain-English sentences for the homepage (outline section 3)."""
    beta, fs, vol = result["beta"], result["firm_specific"], result["volatility"]
    move = abs(10 * beta)
    if beta >= 0:
        market = (f"If the stock market drops 10%, your portfolio would typically drop about {move:.0f}%, "
                  f"and rise about {move:.0f}% when the market gains 10%.")
    else:
        market = "Your portfolio tends to move against the stock market."
    diversification = (f"About {fs:.0%} of your portfolio's ups and downs come from news about individual companies. "
                       + ("Stocks from other industries would lower this." if fs > 0.4
                          else "That's fairly well diversified."))
    volatility = f"In a typical month, your portfolio moves up or down about {vol:.1%}."
    headline = f"Your portfolio scores {scored['score']:.0f} out of 100: {scored['risk_words']}."
    return {"headline": headline, "market": market, "diversification": diversification, "volatility": volatility}
