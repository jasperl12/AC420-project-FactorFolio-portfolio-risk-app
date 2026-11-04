"""
FactorFolio models (Build Guide Step 3): CAPM and Fama-French 3-factor regressions,
the R-squared risk split, and the portfolio's monthly return series.

Adapted from the AC420 class notebooks (Prof. Wysocki):
- Lecture 12 & 13: y = RET - rf, X = sm.add_constant(X), model = sm.OLS(y, X).fit()
- Lecture 17: missing='drop', and pulling single numbers out with model.params,
  model.pvalues and model.rsquared so many stocks can be compared in one table
- Lecture 12 study guide (4.7): Var(R) = beta^2 x Var(Mkt) + Var(e), so R-squared is the
  systematic share of risk and 1 - R-squared is the firm-specific share

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


def analyze_stock(window, col):
    """CAPM, FF3 and the risk split for one return column of the data window."""
    data = window[[col] + FF3_FACTORS + ["rf"]].dropna()
    y = data[col] - data["rf"]                       # excess return (Lecture 12)
    capm = run_capm(y, data["mktrf"])
    ff3 = run_ff3(y, data)
    split = risk_split(capm)
    return {
        "n_months": capm["n_months"],
        "alpha": capm["alpha"], "alpha_p": capm["alpha_p"],
        "beta": capm["beta"], "r2": capm["r2"],
        "ff3_alpha": ff3["alpha"], "ff3_alpha_p": ff3["alpha_p"],
        "ff3_beta": ff3["beta"], "s": ff3["s"], "h": ff3["h"],
        "ff3_r2": ff3["r2"], "ff3_adj_r2": ff3["adj_r2"],
        "volatility": data[col].std(),               # monthly standard deviation
        "firm_specific": split["firm_specific_share"],
        "capm": capm, "ff3": ff3, "split": split,
    }


SUMMARY_COLS = ["n_months", "alpha", "alpha_p", "beta", "r2", "ff3_alpha", "ff3_alpha_p",
                "ff3_beta", "s", "h", "ff3_r2", "ff3_adj_r2", "volatility", "firm_specific"]


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
