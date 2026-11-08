# FactorFolio - Risk Deep Dive page (Build Guide Step 7; outline section 4)
# The math behind the homepage risk score: the score's four parts, the CAPM scatterplot and
# regression results (Lectures 12, 13 and 17), the Sharpe ratio (Lecture 18), the R-squared risk
# split, each stock's beta and R-squared, the Fama-French style map, and Altman Z-scores (Lecture 10).
# Layout follows the Lecture 14 streamlit_app.py (st.tabs); charts use Plotly Express (Lecture 5).
import pandas as pd
import plotly.express as px
import streamlit as st

from core import data as fdata
from core import models, scoring, yahoo

st.set_page_config(page_title="Risk Deep Dive | FactorFolio", layout="centered")
st.title("Risk Deep Dive")
st.write("The math behind your risk score: beta, R², Fama-French factors and Altman Z-scores.")

ZONE_COLORS = {"safe": "#16a34a", "grey": "#ca8a04", "distress": "#dc2626"}


def pct(x, decimals=1):
    return "n/a" if pd.isna(x) else f"{x:.{decimals}%}"


def num(x, decimals=2):
    return "n/a" if pd.isna(x) else f"{x:.{decimals}f}"


# ---------- The homepage portfolio, same steps as the homepage ----------
portfolio = st.session_state.get("portfolio", fdata.DEFAULT_PORTFOLIO)
st.caption("Your homepage portfolio: " + ", ".join(f"{t} {w:.0f}%" for t, w in portfolio.items())
           + ". Change it on the Portfolio Builder page.")
with st.spinner("Downloading prices and running the regressions..."):
    ds = fdata.build_dataset(list(portfolio), weights=portfolio)
if ds["error"]:
    st.error(ds["error"])
    st.stop()
usable = {t: w for t, w in portfolio.items() if t in ds["tickers"]}
if not usable:
    st.error(fdata.no_data_message(ds))
    st.stop()
result = models.analyze_portfolio(ds, usable)
port, stocks, weights = result["portfolio"], result["stocks"], result["weights"]
held = list(weights.index)
prices_now = {t: ds["prices"][t].dropna().iloc[-1] for t in held if t in ds["prices"].columns}
with st.spinner("Looking up financial statements for the Altman Z-score..."):
    z_table = models.altman_table(held, weights.to_dict(), prices_now)
health = models.health_points(z_table)
scored = scoring.risk_score(port, health_points=health["points"])
spy = (scoring.risk_score(result["benchmark"], health_points=scoring.spy_health_points())
       if result["benchmark"] else None)

tabs = st.tabs(["Score", "CAPM", "Each stock", "Style", "Health"])

# ---------- Tab 1: how the score is built ----------
with tabs[0]:
    st.subheader("Your risk score, part by part")
    c1, c2 = st.columns(2)
    c1.metric("Your portfolio", f"{scored['score']:.0f} / 100", scored["risk_words"], delta_color="off",
              delta_arrow="off")
    if spy:
        c2.metric("SPY (S&P 500 fund)", f"{spy['score']:.0f} / 100", spy["risk_words"], delta_color="off",
                  delta_arrow="off")
    parts = pd.DataFrame([{"Part": c["name"], "Part score": c["score"], "Weight": c["weight"],
                           "Points": c["points"], "Measure": c["metric"], "Your value": c["value_text"]}
                          for c in scored["components"]])
    bar = px.bar(parts, x="Part score", y="Part", orientation="h", text=[f"{v:.0f}" for v in parts["Part score"]],
                 hover_data={"Weight": ":.0%", "Points": ":.1f", "Your value": True})
    bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 100],
                      xaxis_title="Part score (0 to 100)", yaxis_title="")
    bar.update_yaxes(autorange="reversed")
    bar.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
    table = parts.assign(**{"Part score": parts["Part score"].round(0).astype(int),
                            "Weight": [f"{w:.0%}" for w in parts["Weight"]],
                            "Points": parts["Points"].round(1)})
    st.dataframe(table[["Part", "Part score", "Weight", "Points", "Your value", "Measure"]],
                 hide_index=True, width="stretch")
    st.write(f"**Total: {scored['score']:.1f} points**, {scored['risk_words']} ({scoring.bands_text()}).")
    with st.expander("What does this mean?"):
        st.write("The first three parts compare your portfolio with 66 large U.S. stocks (S&P 100 members in "
                 "the class WRDS data): a part score of 70 means your portfolio is riskier on that measure than "
                 "70% of them (the Lecture 3 rank idea). Financial health is different: each company in "
                 "Altman's safe zone adds 0 points, grey 50 and distress 100, weighted by how much you hold. "
                 "Points = part score × weight, and the points add up to the risk score.")

# ---------- Tab 2: CAPM scatterplot and regression results ----------
with tabs[1]:
    st.subheader("CAPM: your returns vs. the market")
    choices = ["Your portfolio"] + held + (["SPY"] if result["benchmark"] else [])
    pick = st.selectbox("Show", choices, key="dd_pick")
    if pick == "Your portfolio":
        res = port
        y_raw = result["returns"]
    elif pick == "SPY":
        res = result["benchmark"]
        y_raw = ds["window"]["SPY"]
    else:
        res = result["stock_details"][pick]
        y_raw = ds["window"][pick]
    window = ds["window"].loc[y_raw.dropna().index]
    scatter_df = pd.DataFrame({"Month": window.index.astype(str),
                               "Market excess return": window["mktrf"],
                               "Excess return": y_raw.loc[window.index] - window["rf"]})
    fig = px.scatter(scatter_df, x="Market excess return", y="Excess return", trendline="ols",
                     hover_data=["Month"], trendline_color_override="#dc2626")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    fit = px.get_trendline_results(fig).px_fit_results.iloc[0]
    slope, intercept = fit.params[1], fit.params[0]
    match = "matches" if abs(slope - res["beta"]) < 1e-8 else "does not match"
    st.write(f"Slope of the red line = **beta = {slope:.2f}** ({match} the regression's beta of {res['beta']:.2f}). "
             f"Where the line crosses zero = **alpha = {intercept:.2%} a month**.")
    st.write(models.alpha_label(res["capm"]))
    with st.expander("What does this mean?"):
        st.write("Each dot is one month: how much the market beat the risk-free rate (across) and how much "
                 f"{'your portfolio' if pick == 'Your portfolio' else pick} beat it (up). The red line is the CAPM "
                 "regression (Lecture 12). Its slope is beta: how much the returns move when the market moves 1%. "
                 "Its intercept is alpha: the average monthly return the market doesn't explain. Following "
                 "Lecture 17, alpha only counts as real when its p-value is below 0.05; otherwise it could be luck.")

    st.subheader("Sharpe ratio: return per unit of risk")
    shown = {"Your portfolio": port}
    if pick not in ("Your portfolio", "SPY"):
        shown[pick] = res
    if result["benchmark"]:
        shown["SPY"] = result["benchmark"]
    cols = st.columns(len(shown))
    for col, (name, r) in zip(cols, shown.items()):
        col.metric(f"{name} (monthly)", f"{r['sharpe']:.2f}")
    who = "your portfolio" if pick in ("Your portfolio", "SPY") else pick
    focus = shown["Your portfolio"] if who == "your portfolio" else res
    vs_spy = f", vs. {result['benchmark']['sharpe']:.2f} for SPY" if result["benchmark"] else ""
    st.write(f"Over the last {focus['n_months']} months, {who} earned {focus['sharpe']:.2f} of extra return "
             f"(above the risk-free rate) for every unit of risk{vs_spy}. Higher is better, but a high past "
             "Sharpe ratio doesn't predict a high future one (Lecture 18).")
    st.caption("Yearly versions (monthly × √12): " +
               ", ".join(f"{name} {r['sharpe_annual']:.2f}" for name, r in shown.items()) + ".")
    with st.expander("What does this mean?"):
        st.write("The Sharpe ratio (Lecture 18) is the average monthly return above the risk-free rate divided by "
                 "how much that return bounces around: mean(R - rf) / std(R - rf). It lets you compare a calm "
                 "investment with a wild one fairly. A negative Sharpe ratio means it did worse than the "
                 "risk-free rate. FactorFolio uses Ken French's monthly risk-free rate, which changes every month, "
                 "so it divides by the standard deviation of the excess returns, as on the Lecture 18 slide.")
        st.write("Why FactorFolio shows it but never chases it: try enough stocks or strategies and the best past "
                 "Sharpe ratio looks great by luck alone (after 1,000 tries, the expected best is about 3.26 even "
                 "with zero skill), and past Sharpe ratios were unrelated to future ones in Lecture 18's "
                 "out-of-sample test. So the risk score and the stock suggestions don't use it.")

    st.subheader("CAPM vs. Fama-French 3-factor")
    capm_m, ff3_m = res["capm"]["model"], res["ff3"]["model"]

    def cell(model, name, as_pct=False):
        if name not in model.params:
            return ""
        v = f"{model.params[name]:.2%}" if as_pct else f"{model.params[name]:.2f}"
        return f"{v} (t {model.tvalues[name]:.1f}, p {model.pvalues[name]:.2f})"

    reg = pd.DataFrame({
        "": ["Alpha (per month)", "Beta (market)", "Size (SMB)", "Value (HML)", "R²", "Adjusted R²", "Months"],
        "CAPM": [cell(capm_m, "const", True), cell(capm_m, "mktrf"), "", "",
                 num(capm_m.rsquared), num(capm_m.rsquared_adj), str(int(capm_m.nobs))],
        "Fama-French 3-factor": [cell(ff3_m, "const", True), cell(ff3_m, "mktrf"), cell(ff3_m, "smb"),
                                 cell(ff3_m, "hml"), num(ff3_m.rsquared), num(ff3_m.rsquared_adj),
                                 str(int(ff3_m.nobs))],
    })
    st.dataframe(reg, hide_index=True, width="stretch")
    with st.expander("What does this mean?"):
        st.write("Fama-French adds two more factors to the CAPM (Lecture 13): SMB, how small companies did vs. "
                 "large ones, and HML, how value companies did vs. growth companies. A t-stat above about 2 (p below "
                 "0.05) means the number is statistically different from zero. If adjusted R² rises from CAPM to "
                 "Fama-French, the extra factors explain more of the ups and downs.")

    st.subheader("Where the risk comes from")
    split = res["split"]
    pie = px.pie(names=["Market (systematic)", "Company-specific"],
                 values=[split["systematic_share"], split["firm_specific_share"]], hole=0.35,
                 color_discrete_sequence=["#2563eb", "#f59e0b"])
    pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(pie, width="stretch", config={"displayModeBar": False})
    with st.expander("What does this mean?"):
        st.write("Total risk splits into two parts (Lecture 12): Var(R) = beta² × Var(market) + Var(company "
                 "news). R² is the market's share and 1 - R² is the company-specific share. Company-specific "
                 "risk can be diversified away by owning stocks from different industries; market risk can't.")

# ---------- Tab 3: each stock ----------
with tabs[2]:
    st.subheader("Each stock's beta and R²")
    per = stocks.reset_index().rename(columns={"ticker": "Stock"})
    per["Weight"] = per["weight"]
    fig = px.scatter(per, x="beta", y="r2", size="Weight", text="Stock", size_max=30, opacity=0.6,
                     hover_data={"Weight": ":.0%", "beta": ":.2f", "r2": ":.2f"},
                     labels={"beta": "Beta (market risk)", "r2": "R² (share of risk from the market)"})
    fig.update_traces(textposition="top center")
    fig.add_vline(x=1, line_dash="dot", line_color="#6b7280", annotation_text="market beta = 1",
                  annotation_position="bottom right")
    fig.add_scatter(x=[port["beta"]], y=[port["r2"]], mode="markers+text", text=["Portfolio"],
                    textposition="middle right", marker=dict(symbol="star", size=16, color="#111827"),
                    showlegend=False)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    each = pd.DataFrame({
        "Stock": stocks.index,
        "Weight": [pct(w, 0) for w in stocks["weight"]],
        "Beta": [num(b) for b in stocks["beta"]],
        "R²": [num(r) for r in stocks["r2"]],
        "Alpha / month": [pct(a, 2) for a in stocks["alpha"]],
        "Alpha p": [num(p) for p in stocks["alpha_p"]],
        "Monthly volatility": [pct(v) for v in stocks["volatility"]],
        "Sharpe (monthly)": [num(v) for v in stocks["sharpe"]],
    })
    st.dataframe(each, hide_index=True, width="stretch")
    low_r2 = [t for t, r in stocks["r2"].items() if r < 0.20]
    if low_r2:
        st.caption(f"Beta may be unreliable for {', '.join(low_r2)}: the market explains less than 20% of "
                   f"{'its' if len(low_r2) == 1 else 'their'} ups and downs (R² below 0.20, check 12, Lecture 13).")
    sig = [t for t, p in stocks["alpha_p"].items() if p < models.SIGNIFICANCE]
    st.caption("Significant alpha (p below 0.05): " + (", ".join(sig) if sig else "none") +
               ". The rest could be luck (Lecture 17).")
    with st.expander("What does this mean?"):
        st.write("Like the Lecture 13 comparison of two stocks: beta (across) says how much a stock swings with "
                 "the market, and R² (up) says how much of its risk comes from the market rather than company "
                 "news. Bubble size is how much you hold. The star is your whole portfolio: mixing stocks raises "
                 "R² because company-specific surprises partly cancel out.")

# ---------- Tab 4: style map ----------
with tabs[3]:
    st.subheader("Style map: size and value")
    style = stocks[["weight", "s", "h"]].copy()
    style["P/B"] = z_table["pb"].reindex(style.index)
    style["Style"] = [("Small" if s > 0 else "Large") + " " + ("value" if h > 0 else "growth")
                      for s, h in zip(style["s"], style["h"])]
    sdf = style.reset_index().rename(columns={"ticker": "Stock"})
    fig = px.scatter(sdf, x="h", y="s", size="weight", text="Stock", size_max=30, opacity=0.6,
                     hover_data={"weight": ":.0%", "P/B": ":.1f", "Style": True},
                     labels={"h": "Value loading h (HML)", "s": "Size loading s (SMB)"})
    fig.update_traces(textposition="top center")
    fig.add_hline(y=0, line_color="#9ca3af")
    fig.add_vline(x=0, line_color="#9ca3af")
    fig.add_scatter(x=[port["h"]], y=[port["s"]], mode="markers+text", text=["Portfolio"],
                    textposition="middle right", marker=dict(symbol="star", size=16, color="#111827"),
                    showlegend=False)
    xs = [abs(v) for v in list(sdf["h"]) + [port["h"]]] + [0.3]
    ys = [abs(v) for v in list(sdf["s"]) + [port["s"]]] + [0.3]
    xr, yr = max(xs) * 1.3, max(ys) * 1.3
    for x, y, label in [(xr, yr, "Small value"), (-xr, yr, "Small growth"),
                        (xr, -yr, "Large value"), (-xr, -yr, "Large growth")]:
        fig.add_annotation(x=x * 0.95, y=y * 0.95, text=label, showarrow=False, font=dict(color="#6b7280"),
                           xanchor="right" if x > 0 else "left", yanchor="top" if y > 0 else "bottom")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[-xr, xr], yaxis_range=[-yr, yr])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.dataframe(pd.DataFrame({"Stock": style.index, "Size s": [num(v) for v in style["s"]],
                               "Value h": [num(v) for v in style["h"]], "Style": style["Style"].values,
                               "P/B": [num(v, 1) for v in style["P/B"]]}),
                 hide_index=True, width="stretch")
    st.caption(f"Your portfolio: size loading {port['s']:.2f}, value loading {port['h']:.2f} "
               f"({'small' if port['s'] > 0 else 'large'}-company, {'value' if port['h'] > 0 else 'growth'} tilt).")
    with st.expander("What does this mean?"):
        st.write("The Fama-French loadings (Lecture 13) place each stock on a map. Up (s above 0) means it "
                 "behaves like small companies; down means large. Right (h above 0) means it behaves like value "
                 "companies, which have high book value compared with their price; left means growth companies. "
                 "Price-to-book (P/B) is shown as a check: value stocks usually have a low P/B, because "
                 "book-to-price is the measure behind HML (Lecture 16).")

# ---------- Tab 5: Altman Z-scores ----------
with tabs[4]:
    st.subheader("Financial health: Altman Z-score (Lecture 10)")
    st.latex(r"\begin{aligned} Z &= 1.2\,T_1 + 1.4\,T_2 + 3.3\,T_3 \\ &\quad + 0.6\,T_4 + 0.99\,T_5 "
             r"\end{aligned}")   # two lines fit a phone
    scored_z = z_table[z_table["zone"].notna()]
    if not scored_z.empty:
        zfig = px.bar(scored_z.reset_index(), x="ticker", y="z", color="zone", color_discrete_map=ZONE_COLORS,
                      text=[f"{v:.2f}" for v in scored_z["z"]], labels={"ticker": "Stock", "z": "Z-score"})
        zfig.add_hrect(y0=min(0, scored_z["z"].min()), y1=1.80, fillcolor="#dc2626", opacity=0.08, line_width=0)
        zfig.add_hrect(y0=1.80, y1=2.99, fillcolor="#ca8a04", opacity=0.08, line_width=0)
        zfig.add_hline(y=1.80, line_dash="dot", line_color="#dc2626")
        zfig.add_hline(y=2.99, line_dash="dot", line_color="#16a34a")
        zfig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Zone")
        st.plotly_chart(zfig, width="stretch", config={"displayModeBar": False})
    ztab = pd.DataFrame({
        "Stock": z_table.index,
        "Year end": [d or "n/a" for d in z_table["fiscal_year_end"]],
        "T1 WC/TA": [num(v) for v in z_table["T1"]],
        "T2 RE/TA": [num(v) for v in z_table["T2"]],
        "T3 EBIT/TA": [num(v) for v in z_table["T3"]],
        "T4 MVE/TL": [num(v) for v in z_table["T4"]],
        "T5 Sales/TA": [num(v) for v in z_table["T5"]],
        "Z": [num(v) for v in z_table["z"]],
        "Zone": [z if pd.notna(z) else s for z, s in zip(z_table["zone"], z_table["status"])],
    })
    st.dataframe(ztab, hide_index=True, width="stretch")
    if health["points"] is None:
        st.write("None of your stocks could get a Z-score, so financial health is left out of the risk score.")
    else:
        st.write(f"**Health points: {health['points']:.0f} of 100.** {health['share_at_risk']:.0%} of the money "
                 f"in scored stocks is in the grey or distress zone (safe 0 points, grey 50, distress 100, "
                 f"weighted by how much you hold).")
    note = yahoo.backup_note(z_table["source"].to_dict(), "financial statements", yahoo.financials_snapshot_date())
    if note:
        st.caption(note)
    st.caption("T1 = working capital / total assets, T2 = retained earnings / total assets, T3 = EBIT / total "
               "assets, T4 = market value of equity / total liabilities, T5 = sales / total assets. Statements "
               "are the latest fiscal year from Yahoo Finance; the market value uses today's price. When Yahoo "
               "leaves total liabilities blank, the app uses total assets minus equity. Zones: safe above 2.99, "
               "grey 1.80 to 2.99, distress below 1.80.")
    older = {t: o for t, o in z_table["older_items"].items() if o}
    if older:
        st.caption("Yahoo left these line items blank for the latest year, so the prior year's values are used: "
                   + "; ".join(f"{t}: {o.replace('_', ' ')}" for t, o in older.items()) + ".")
    with st.expander("What does this mean?"):
        st.write("Altman's Z-score combines five ratios into one number: higher is safer (Lecture 10). In class, "
                 "General Motors scored 0.67 at the end of 2007, deep in the distress zone, about 17 months "
                 "before it filed for bankruptcy. Z is a screening tool, not a verdict. Altman built it on "
                 "manufacturing companies, so banks and insurers are left out. Utilities and real estate firms "
                 "are left out too: their large, steady debt put almost all of them in the distress zone even "
                 "though they rarely fail, so FactorFolio doesn't count it against them. Asset-light companies "
                 "(like software firms) can also look different from the firms in his sample.")

st.caption("For education only, not investment advice. Regressions use the last "
           f"{len(ds['window'])} months of Yahoo Finance prices and Ken French factors.")
