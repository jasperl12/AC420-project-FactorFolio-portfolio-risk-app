"""
FactorFolio models (Build Guide Step 3): CAPM and Fama-French 3-factor regressions,
the R-squared risk split, and the portfolio's monthly return series.

Adapted from the AC420 class notebooks (Prof. Wysocki):
- Lecture 12 & 13: y = RET - rf, X = sm.add_constant(X), model = sm.OLS(y, X).fit()
- Lecture 17: missing='drop', and pulling single numbers out with model.params,
  model.pvalues and model.rsquared so many stocks can be compared in one table
- Lecture 12 study guide (4.7): Var(R) = beta^2 x Var(Mkt) + Var(e), so R-squared is the
  systematic share of risk and 1 - R-squared is the firm-specific share
- Lecture 18: Sharpe ratio = mean(R - rf) / std(R - rf), return per unit of risk

Known-answer tests are in notebooks/known_answer_tests.ipynb.
"""
import pandas as pd
import statsmodels.api as sm

FF3_FACTORS = ["mktrf", "smb", "hml"]
SIGNIFICANCE = 0.05   # Lecture 17: alpha only counts if p < 0.05


def _fit(y, X):
    """OLS with an intercept. Without sm.add_constant there is no alpha (data check 11)."""
    X = sm.add_constant(X)
    return sm.OLS(y, X, missing="drop").fit()


def run_capm(y, mktrf):
    """CAPM: (R - Rf) = alpha + beta x (Mkt - Rf) + e.

    y: the stock's or portfolio's excess return (return - rf), one value per month.
    mktrf: the market's excess return (already excess, so rf is not subtracted again).
    Returns a dict of results; the fitted statsmodels model is under 'model'.
    """
    model = _fit(y, mktrf.rename("mktrf"))
    return {
        "model_name": "CAPM",
        "n_months": int(model.nobs),
        "alpha": model.params["const"],
        "alpha_t": model.tvalues["const"],
        "alpha_p": model.pvalues["const"],
        "alpha_significant": bool(model.pvalues["const"] < SIGNIFICANCE),
        "beta": model.params["mktrf"],
        "beta_t": model.tvalues["mktrf"],
        "beta_p": model.pvalues["mktrf"],
        "beta_ci": tuple(model.conf_int().loc["mktrf"]),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "model": model,
    }


def run_ff3(y, factors):
    """Fama-French 3-factor: (R - Rf) = alpha + b1 x MKTRF + b2 x SMB + b3 x HML + e.

    factors: a table with the columns mktrf, smb and hml (double brackets in the class notebook).
    """
    model = _fit(y, factors[FF3_FACTORS])
    return {
        "model_name": "FF3",
        "n_months": int(model.nobs),
        "alpha": model.params["const"],
        "alpha_t": model.tvalues["const"],
        "alpha_p": model.pvalues["const"],
        "alpha_significant": bool(model.pvalues["const"] < SIGNIFICANCE),
        "beta": model.params["mktrf"],
        "s": model.params["smb"],
        "h": model.params["hml"],
        "beta_p": model.pvalues["mktrf"],
        "s_p": model.pvalues["smb"],
        "h_p": model.pvalues["hml"],
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "model": model,
    }


def risk_split(capm_result):
    """Split total risk into systematic (market) and firm-specific parts (Lecture 12, 4.7).

    With an intercept, Var(y) = Var(fitted) + Var(residual), and R-squared = Var(fitted) / Var(y).
    """
    model = capm_result["model"]
    systematic = model.fittedvalues.var()
    firm_specific = model.resid.var()
    total = systematic + firm_specific
    return {
        "total_variance": total,
        "systematic_variance": systematic,
        "firm_specific_variance": firm_specific,
        "systematic_share": systematic / total,       # equals R-squared
        "firm_specific_share": firm_specific / total,  # equals 1 - R-squared
    }


def alpha_label(result):
    """Plain-English reading of alpha that respects significance (Lecture 17)."""
    a, p = result["alpha"], result["alpha_p"]
    if p >= SIGNIFICANCE:
        return f"Not significant (alpha {a:.2%} per month, p = {p:.2f}): no evidence of beating its risk-adjusted benchmark."
    direction = "beat" if a > 0 else "fell short of"
    return f"Significant (alpha {a:.2%} per month, p = {p:.3f}): it {direction} its risk-adjusted benchmark in this period."


def portfolio_returns(window, weights):
    """Monthly portfolio return = weighted sum of the stocks' returns.

    weights: {ticker: weight in %}. Stocks missing from the window are dropped and the
    remaining weights rescaled to 100%. Only months where every stock has a return are used.
    Returns (series of monthly returns, weights used as fractions that add to 1).
    """
    available = {t: w for t, w in weights.items() if t in window.columns and w > 0}
    if not available:
        raise ValueError("None of the portfolio's stocks are in the data window.")
    total = sum(available.values())
    w = pd.Series({t: v / total for t, v in available.items()})
    stock_returns = window[list(w.index)].dropna()
    return stock_returns.mul(w, axis=1).sum(axis=1).rename("PORTFOLIO"), w


def sharpe_ratio(returns, rf):
    """Lecture 18 Sharpe ratio: mean(R - rf) / std(R - rf), monthly, plus the yearly version (x sqrt 12).

    The Assignment #5 notebook divides by returns.std() with a constant rf of 0.25% a month; that is
    the same number only when rf never changes. Ken French's monthly rf changes every month, so this
    uses the standard deviation of the excess returns, the formula on the Lecture 18 slide.
    It describes the past only: Lecture 18 shows a high past Sharpe ratio doesn't predict a high future one.
    """
    excess = (pd.Series(returns) - pd.Series(rf)).dropna()
    monthly = excess.mean() / excess.std()
    return {"monthly": monthly, "annual": monthly * 12 ** 0.5}


def analyze_stock(window, col):
    """CAPM, FF3 and the risk split for one return column of the data window."""
    data = window[[col] + FF3_FACTORS + ["rf"]].dropna()
    y = data[col] - data["rf"]                       # excess return (Lecture 12)
    capm = run_capm(y, data["mktrf"])
    ff3 = run_ff3(y, data)
    split = risk_split(capm)
    sharpe = sharpe_ratio(data[col], data["rf"])
    return {
        "n_months": capm["n_months"],
        "alpha": capm["alpha"], "alpha_p": capm["alpha_p"],
        "beta": capm["beta"], "r2": capm["r2"],
        "ff3_alpha": ff3["alpha"], "ff3_alpha_p": ff3["alpha_p"],
        "ff3_beta": ff3["beta"], "s": ff3["s"], "h": ff3["h"],
        "ff3_r2": ff3["r2"], "ff3_adj_r2": ff3["adj_r2"],
        "volatility": data[col].std(),               # monthly standard deviation
        "firm_specific": split["firm_specific_share"],
        "sharpe": sharpe["monthly"], "sharpe_annual": sharpe["annual"],   # Lecture 18, past only
        "capm": capm, "ff3": ff3, "split": split,
    }


SUMMARY_COLS = ["n_months", "alpha", "alpha_p", "beta", "r2", "ff3_alpha", "ff3_alpha_p",
                "ff3_beta", "s", "h", "ff3_r2", "ff3_adj_r2", "volatility", "firm_specific", "sharpe"]


def analyze_portfolio(ds, weights, benchmark="SPY"):
    """Run the models for every stock, the benchmark and the whole portfolio.

    ds: the dict from core.data.build_dataset(). weights: {ticker: weight in %}.
    Returns a dict with:
      'stocks'    table with one row per stock (same columns as data/sp100_estimates.csv)
      'portfolio' full results for the portfolio (CAPM, FF3, risk split, volatility)
      'benchmark' full results for SPY
      'weights'   the weights actually used (fractions that add to 1)
      'returns'   the portfolio's monthly return series
    """
    window = ds["window"]
    stock_list = [t for t in ds["tickers"] if t in weights]
    port_ret, w = portfolio_returns(window, {t: weights[t] for t in stock_list})

    rows, details = [], {}
    for t in w.index:
        res = analyze_stock(window, t)
        details[t] = res
        rows.append({"ticker": t, "weight": w[t], **{k: res[k] for k in SUMMARY_COLS}})
    stocks = pd.DataFrame(rows).set_index("ticker")

    port_window = window.loc[port_ret.index].copy()
    port_window["PORTFOLIO"] = port_ret
    portfolio = analyze_stock(port_window, "PORTFOLIO")
    bench = analyze_stock(window, benchmark) if benchmark in window.columns else None

    return {"stocks": stocks, "stock_details": details, "portfolio": portfolio,
            "benchmark": bench, "weights": w, "returns": port_ret}


# ---------- Altman Z-score (Lecture 10; Build Guide Step 7) ----------
# Z = 1.2 T1 + 1.4 T2 + 3.3 T3 + 0.6 T4 + 0.99 T5
#   T1 = working capital / total assets       T2 = retained earnings / total assets
#   T3 = EBIT / total assets                  T4 = market value of equity / total liabilities
#   T5 = sales / total assets
# Zones: safe above 2.99, grey 1.80 to 2.99, distress below 1.80.
# Altman built Z on manufacturing firms, so banks, insurers and other financial firms are left out
# (their balance sheets work differently, Lecture 10). Step 8 calibration also leaves out utilities
# and real estate: their heavy, regulated debt put almost all of them in the distress zone, which
# made an all-utilities portfolio score riskier than three tech stocks' worth of beta suggests.
ALTMAN_WEIGHTS = {"T1": 1.2, "T2": 1.4, "T3": 3.3, "T4": 0.6, "T5": 0.99}
ZONE_POINTS = {"safe": 0, "grey": 50, "distress": 100}   # risk score points (outline 5.4)


def altman_z(T1, T2, T3, T4, T5):
    """Altman's Z-score from the five ratios (Lecture 10)."""
    return 1.2 * T1 + 1.4 * T2 + 3.3 * T3 + 0.6 * T4 + 0.99 * T5


def altman_zone(z):
    if z > 2.99:
        return "safe"
    elif z >= 1.80:
        return "grey"
    else:
        return "distress"


NOT_APPLICABLE = {"Financial Services": "financial firm", "Utilities": "utility", "Real Estate": "real estate"}


def altman_not_applicable(sector, industry):
    """Why Altman Z doesn't apply (check 15), or None when it does.

    Financial firms (Lecture 10), plus utilities and real estate (Step 8 calibration).
    """
    industry = industry if isinstance(industry, str) else ""      # missing labels come in as NaN
    if sector in NOT_APPLICABLE:
        return NOT_APPLICABLE[sector]
    if any(w in industry for w in ("Bank", "Insurance")):
        return "financial firm"
    return None


def altman_ratios(items, price):
    """T1 to T5 from statement items (core/yahoo.py) and today's share price."""
    ta, tl = items["total_assets"], items["total_liabilities"]
    market_value = items["shares"] * price                 # market value of equity
    return {"T1": (items["current_assets"] - items["current_liabilities"]) / ta,
            "T2": items["retained_earnings"] / ta,
            "T3": items["ebit"] / ta,
            "T4": market_value / tl,
            "T5": items["revenue"] / ta}


def altman_for_stock(items, price, sector=None, industry=None):
    """Z, zone and the five ratios for one stock, or why it can't be scored."""
    row = {"T1": float("nan"), "T2": float("nan"), "T3": float("nan"), "T4": float("nan"),
           "T5": float("nan"), "z": float("nan"), "zone": None,
           "fiscal_year_end": items.get("fiscal_year_end"), "source": items.get("source")}
    needed = ["current_assets", "current_liabilities", "total_assets", "retained_earnings",
              "total_liabilities", "shares", "ebit", "revenue"]
    why_not = altman_not_applicable(sector, industry)
    if why_not:
        return {**row, "status": f"not applicable ({why_not})"}
    items = dict(items)
    if pd.isna(items.get("total_liabilities")) and pd.notna(items.get("total_assets")) \
            and pd.notna(items.get("stockholders_equity")):
        # Yahoo sometimes leaves total liabilities blank: use the balance sheet identity
        items["total_liabilities"] = items["total_assets"] - items["stockholders_equity"]
    missing = [k for k in needed if pd.isna(items.get(k))]
    if len(missing) == len(needed) and items.get("source") is None:
        return {**row, "status": "Yahoo didn't send financial statements (no saved backup)"}
    if missing or pd.isna(price) or items["total_assets"] <= 0 or items["total_liabilities"] <= 0:
        why = ", ".join(k.replace("_", " ") for k in missing) or "price or totals"
        return {**row, "status": f"missing data ({why})"}
    ratios = altman_ratios(items, price)
    z = altman_z(**ratios)
    return {**row, **ratios, "z": z, "zone": altman_zone(z), "status": "scored"}


def health_points(table, weight_col="weight"):
    """Risk score points from Altman zones (safe 0, grey 50, distress 100), weighted by portfolio weight.

    Stocks that can't be scored are left out and the weights rescaled (outline 5.4).
    """
    scored = table[table["zone"].isin(list(ZONE_POINTS))]
    if scored.empty:
        return {"points": None, "share_at_risk": None, "n_scored": 0}
    w = scored[weight_col] / scored[weight_col].sum()
    points = float((w * scored["zone"].map(ZONE_POINTS)).sum())
    at_risk = float(w[scored["zone"] != "safe"].sum())
    return {"points": points, "share_at_risk": at_risk, "n_scored": len(scored)}


def altman_table(tickers, weights, prices):
    """Altman Z for each stock in the portfolio, using core/yahoo.py for statements and sectors.

    prices: {ticker: latest price}. weights: fractions or percents (only their ratios matter).
    """
    from core import yahoo   # downloads live in core/yahoo.py; the math above stays download-free
    info = yahoo.lookup_many(yahoo.company_info, tickers)
    statements = yahoo.lookup_many(yahoo.financials, tickers)
    rows = []
    for t in tickers:
        items, price = statements[t], prices.get(t, float("nan"))
        res = altman_for_stock(items, price, info[t]["sector"], info[t]["industry"])
        # Price-to-book for the style map: book value per share = stockholders' equity / shares (Lecture 16)
        bvps = items["stockholders_equity"] / items["shares"] if items.get("shares") else float("nan")
        pb = price / bvps if pd.notna(bvps) and bvps > 0 else float("nan")
        older = items.get("older_items")
        rows.append({"ticker": t, "weight": weights.get(t, 0), "sector": info[t]["sector"],
                     "industry": info[t]["industry"], "price": price, "pb": pb, **res,
                     "older_items": older if isinstance(older, str) else ""})
    return pd.DataFrame(rows).set_index("ticker")


def market_health_points(fin_snapshot, sectors):
    """A stand-in for SPY's health points: the S&P 100 backup, weighted by market value like an index fund.

    fin_snapshot: core.yahoo.load_financials_snapshot(). sectors: table with ticker, sector, industry.
    """
    if fin_snapshot.empty:
        return None
    lookup = sectors.set_index("ticker")
    rows = []
    for t, r in fin_snapshot.iterrows():
        if t not in lookup.index:
            continue
        res = altman_for_stock(r.to_dict(), r["price"], lookup.at[t, "sector"], lookup.at[t, "industry"])
        if res["zone"]:
            rows.append({"zone": res["zone"], "weight": r["shares"] * r["price"]})
    return health_points(pd.DataFrame(rows))["points"] if rows else None
