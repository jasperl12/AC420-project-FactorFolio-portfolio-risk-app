# FactorFolio - AC420 project (Jasper Lin)
# Homepage: the Portfolio Builder (Build Guide Steps 4 to 7: inputs, risk score with the Altman
# Z-score, bear/base/bull projections vs. SPY, allocation chart, fair-value card).
# Layout follows the Lecture 14 streamlit_app.py from Assignment #4; st.navigation names the pages.
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data as fdata
from core import models, projections, scoring, valuation, yahoo

st.set_page_config(page_title="FactorFolio", layout="centered")

MAX_STOCKS = 15
SECTORS_FILE = fdata.DATA_DIR / "sp100_tickers.csv"


def remember(key, default):
    """Widget values are cleared when you visit another page; keep a copy so they come back."""
    return st.session_state.get(f"saved_{key}", default)


@st.cache_data(show_spinner=False)
def sector_lookup():
    df = pd.read_csv(SECTORS_FILE)
    return dict(zip(df["ticker"], df["sector"]))


def holdings_from_table(table):
    """Clean the editable table into {ticker: weight %}; duplicate tickers are added together."""
    weights = {}
    for _, row in table.iterrows():
        tickers = fdata.clean_tickers(str(row.get("Ticker") or ""))
        weight = pd.to_numeric(row.get("Weight (%)"), errors="coerce")
        if tickers and pd.notna(weight) and weight > 0:
            weights[tickers[0]] = weights.get(tickers[0], 0) + float(weight)
    return weights


def normalize_table():
    """Normalize button: rescale the weights in the table to add to 100%."""
    weights = fdata.normalize_weights(holdings_from_table(st.session_state.holdings))
    st.session_state.holdings = pd.DataFrame({"Ticker": list(weights), "Weight (%)": [round(w, 1) for w in weights.values()]})
    st.session_state.editor_base = st.session_state.holdings.copy()
    st.session_state.pop("holdings_editor", None)   # restart the table widget with the new numbers


def signed_money(x):
    return f"+${x:,.0f}" if x >= 0 else f"-${-x:,.0f}"


def growth_chart(path):
    """Line chart of dollar value by year, portfolio vs. SPY (Plotly Express, Lecture 5)."""
    fig = px.line(path, x="Year", y=["Your portfolio", "SPY"], markers=True,
                  color_discrete_map={"Your portfolio": "#2563eb", "SPY": "#9ca3af"})
    fig.update_traces(hovertemplate="%{fullData.name}: $%{y:,.0f}<extra></extra>")
    fig.update_traces(selector=dict(name="Your portfolio"), line_width=3)
    fig.update_traces(selector=dict(name="SPY"), line_dash="dash")   # still visible when the lines overlap
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title="Value ($)", yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                      legend_title_text="", legend=dict(orientation="h", x=0, y=1.12))
    fig.update_xaxes(dtick=1)
    return fig


def gauge(score, spy_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 36}, "valueformat": ".0f"},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 35, 65, 100]},
               "bar": {"color": "#1f2937", "thickness": 0.25},
               "steps": [{"range": [0, 35], "color": "#bbf7d0"},
                         {"range": [35, 65], "color": "#fde68a"},
                         {"range": [65, 100], "color": "#fecaca"}],
               "threshold": {"line": {"color": "#2563eb", "width": 4}, "thickness": 0.9,
                             "value": spy_score if spy_score is not None else score}}))
    fig.update_layout(height=240, margin=dict(l=30, r=30, t=20, b=20))
    return fig


def portfolio_builder():
    st.title("FactorFolio")
    st.write("Build a stock portfolio that matches your comfort with risk. "
             "Pick your stocks, and FactorFolio scores the risk in plain English.")

    # ---------- Inputs ----------
    st.subheader("Your portfolio")
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount to invest ($)", min_value=100, max_value=10_000_000, step=1_000,
                             value=remember("amount", 10_000), key="amount")
    horizon = c2.slider("Time horizon (years)", 1, 10, value=remember("horizon", 5), key="horizon")
    comfort = st.radio("Comfort with risk", list(scoring.COMFORT_RANGES), horizontal=True,
                       index=list(scoring.COMFORT_RANGES).index(remember("comfort", "Moderate")),
                       captions=[scoring.COMFORT_HELP[c] for c in scoring.COMFORT_RANGES], key="comfort")
    st.session_state.saved_amount, st.session_state.saved_horizon, st.session_state.saved_comfort = amount, horizon, comfort

    if "holdings" not in st.session_state:
        st.session_state.holdings = pd.DataFrame({"Ticker": list(fdata.DEFAULT_PORTFOLIO),
                                                  "Weight (%)": [float(w) for w in fdata.DEFAULT_PORTFOLIO.values()]})
    if "holdings_editor" not in st.session_state:   # first time on this page: start from the saved table
        st.session_state.editor_base = st.session_state.holdings.copy()
    st.caption("Type a ticker and its weight. Add rows at the bottom of the table; select a row to delete it.")
    table = st.data_editor(
        st.session_state.editor_base, key="holdings_editor", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance symbol, e.g. AAPL"),
                       "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0,
                                                                   step=1.0, format="%.1f")})
    st.session_state.holdings = table
    weights = holdings_from_table(table)

    if not weights:
        st.warning("Add at least one stock with a weight above 0.")
        return
    if len(weights) > MAX_STOCKS:
        st.warning(f"FactorFolio handles up to {MAX_STOCKS} stocks; only the first {MAX_STOCKS} are used.")
        weights = dict(list(weights.items())[:MAX_STOCKS])
    total = sum(weights.values())
    if abs(total - 100) > 0.01:
        w1, w2 = st.columns([3, 1])
        w1.warning(f"Your weights add to {total:.1f}%. The score below rescales them to 100%.")
        w2.button("Normalize to 100%", on_click=normalize_table)
    st.session_state.portfolio = weights          # shared with the other pages

    # ---------- Data and models ----------
    with st.spinner("Downloading prices and running the CAPM and Fama-French regressions..."):
        ds = fdata.build_dataset(list(weights), weights=weights)
    if ds["error"]:
        st.error(ds["error"])
        return
    usable = {t: w for t, w in weights.items() if t in ds["tickers"]}
    if not usable:
        st.error("None of these stocks have enough monthly data to analyze. Check the tickers.")
        return
    problems = [c for c in ds["checks"] if c["status"] in ("warn", "fail") and not c["check"].startswith("2.")]
    for c in problems:
        st.warning(f"{c['detail']}")
    result = models.analyze_portfolio(ds, usable)
    port = result["portfolio"]
    held = list(result["weights"].index)
    prices_now = {t: ds["prices"][t].dropna().iloc[-1] for t in held if t in ds["prices"].columns}
    with st.spinner("Looking up financial statements for the Altman Z-score..."):
        z_table = models.altman_table(held, result["weights"].to_dict(), prices_now)
    health = models.health_points(z_table)
    scored = scoring.risk_score(port, health_points=health["points"])
    spy = (scoring.risk_score(result["benchmark"], health_points=scoring.spy_health_points())
           if result["benchmark"] else None)
    words = scoring.explain(port, scored, health)
    fit = scoring.comfort_fit(scored["score"], comfort)
    st.session_state.analysis = {"score": scored["score"], "band": scored["band"], "beta": port["beta"],
                                 "health": health["points"]}

    # ---------- Risk score card ----------
    st.subheader("Risk score")
    st.plotly_chart(gauge(scored["score"], spy["score"] if spy else None), width="stretch",
                    config={"displayModeBar": False})
    if spy:
        st.caption(f"Blue line: an S&P 500 fund (SPY) scores {spy['score']:.0f}. "
                   "Green = low risk (0 to 35), yellow = moderate (35 to 65), red = high (65 to 100).")
    st.markdown(f"**{words['headline']}**")
    st.write(words["market"])
    st.write(words["diversification"])
    st.write(words["volatility"])
    st.write(words["health"])
    if fit["status"] == "fits":
        st.success(fit["message"])
    elif fit["status"] == "above":
        st.warning(fit["message"])
    else:
        st.info(fit["message"])

    with st.expander("How this score is built"):
        st.write("Each part compares your portfolio with 66 large U.S. stocks (S&P 100 members in the class "
                 "WRDS data). A part's score is the share of those stocks your portfolio is riskier than.")
        breakdown = pd.DataFrame([{"Part": c["name"], "Measure": c["metric"], "Your portfolio": c["value_text"],
                                   "Part score (0 to 100)": round(c["score"]), "Weight": f"{c['weight']:.0%}",
                                   "Points": round(c["points"], 1)} for c in scored["components"]])
        st.dataframe(breakdown, hide_index=True, width="stretch")
        st.caption("Financial health uses Altman's Z-score (Lecture 10): a company in the safe zone adds 0 points, "
                   "grey 50 and distress 100, weighted by how much of it you hold. Banks and insurers are left out. "
                   "SPY's health comes from the S&P 100 backup, weighted by market value.")
        fin_note = yahoo.backup_note(z_table["source"].to_dict(), "financial statements",
                                     yahoo.financials_snapshot_date())
        if fin_note:
            st.caption(fin_note)
        st.page_link("pages/1_Risk_Deep_Dive.py", label="See every part of the score in the Risk Deep Dive",
                     icon="🧮")

    # ---------- Bear, base and bull projections ----------
    st.subheader("Bear, base and bull projections")
    scenario = st.radio("Market scenario for the next year", projections.SCENARIOS, horizontal=True,
                        index=projections.SCENARIOS.index(remember("scenario", "Base")), key="scenario")
    st.session_state.saved_scenario = scenario
    st.caption(projections.SCENARIO_HELP[scenario])
    rf = projections.risk_free_rate()
    if not rf["ok"]:
        st.warning(rf["note"])
    proj = projections.project(amount, horizon, port["beta"], rf["rf"], scenario)
    st.plotly_chart(growth_chart(proj["path"]), width="stretch", config={"displayModeBar": False})
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Your portfolio in {horizon} yr", f"${proj['portfolio_value']:,.0f}",
              f"{proj['portfolio_value'] / amount - 1:+.1%}")
    m2.metric(f"SPY in {horizon} yr", f"${proj['spy_value']:,.0f}", f"{proj['spy_value'] / amount - 1:+.1%}")
    if abs(proj["difference_pct"]) < 0.0005:   # under 0.05%: call it a tie instead of showing -0.0%
        m3.metric("Difference", signed_money(proj["difference"]), "about the same as SPY", delta_color="off",
                  delta_arrow="off")
    else:
        m3.metric("Difference", signed_money(proj["difference"]), f"{proj['difference_pct']:+.1%} vs. SPY")
    st.markdown(projections.explain(proj).replace("$", "\\$"))   # \$ stops Streamlit reading $...$ as math
    st.session_state.analysis.update({"scenario": scenario, "rf": rf["rf"]})

    if abs(port["beta"] - 1) < 0.1:
        move_word = "about as much as"
    else:
        move_word = "more than" if port["beta"] > 1 else "less than"
    with st.expander("How these projections work"):
        st.write(f"Each scenario sets the stock market's return for the next year. Your portfolio's return comes "
                 f"from the CAPM: risk-free rate + beta × (market return - risk-free rate). With a beta of "
                 f"{port['beta']:.2f}, your portfolio tends to move {move_word} the market. "
                 "SPY's return is the market return itself.")
        st.write("After the first year, both go back to the base return for the rest of the time horizon. "
                 "Past alpha is left out because it may not repeat (Lecture 12).")
        st.dataframe(projections.assumptions_table(proj), hide_index=True, width="stretch")
        st.caption(f"Risk-free rate: {rf['note']} Equity risk premium: {projections.ERP:.1%} (Lecture 15). "
                   "Bear and bull: the 25th and 75th percentiles of annual U.S. market returns from 1927 to 2025 "
                   "(Ken French Data Library). These are scenarios, not predictions.")

    # ---------- Allocation ----------
    st.subheader("Allocation")
    sectors = sector_lookup()
    w = result["weights"]
    alloc = pd.DataFrame({"Ticker": w.index, "Weight": w.values,
                          "Sector": [sectors.get(t, "Other") for t in w.index]})
    alloc["Amount"] = [f"${x:,.0f}" for x in alloc["Weight"] * amount]
    pie = px.pie(alloc, names="Ticker", values="Weight", hole=0.35, hover_data=["Amount", "Sector"])
    pie.update_traces(textinfo="label+percent")
    pie.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(pie, width="stretch", config={"displayModeBar": False})
    by_sector = alloc.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
    st.write("**By sector:** " + ", ".join(f"{s} {v:.0%}" for s, v in by_sector.items()))
    if by_sector.iloc[0] > 0.5:
        st.caption(f"Over half of your money is in one sector ({by_sector.index[0]}), so news about that "
                   "industry can move your whole portfolio.")
    dropped = [t for t in weights if t not in usable]
    if dropped:
        st.caption("Left out because there was not enough data: " + ", ".join(dropped))

    # ---------- Fair value (Lectures 15 and 16) ----------
    st.subheader("Fair value check")
    with st.spinner("Looking up analyst EPS forecasts for your stocks and similar companies..."):
        val = valuation.value_portfolio(result["stocks"], result["weights"], rf["rf"], scenario)
    if val["error"]:
        st.warning(val["error"] + " The fair-value check will come back when Yahoo Finance responds.")
    else:
        fv = val["table"]
        note = valuation.forecast_note(val)
        if note:
            (st.warning if note[0] == "warning" else st.caption)(note[1])
        st.info(valuation.explain(val["portfolio"], fv).replace("$", "\\$"))
        card = pd.DataFrame({
            "Stock": fv.index,
            "Price": [f"${p:,.2f}" if pd.notna(p) else "n/a" for p in fv["price"]],
            "Fair-value range": [f"${lo:,.0f} to ${hi:,.0f}" if pd.notna(lo) else "n/a"
                                 for lo, hi in zip(fv["low"], fv["high"])],
            "Label": fv["label"].values,
        })
        st.dataframe(card, hide_index=True, width="stretch")
        st.caption(f"Each range runs from the DCF value ({scenario.lower()} case, long-term growth "
                   f"{val['g_lt'][scenario]:.0%}; Lecture 15) to the peer comparables value (median P/E of similar "
                   f"S&P 100 stocks × next-year EPS; Lecture 16). Prices as of {val['prices_as_of']}.")
    st.page_link("pages/2_Projections_and_Valuation.py", label="See the DCF, peer comparables and what-if controls",
                 icon="📈")
    st.page_link("pages/4_Data_Check.py", label="Look at the data behind this score", icon="🔍")
    rf_date = f" on {rf['as_of']}" if rf["as_of"] else ""
    st.caption(f"For education only, not investment advice. Prices from Yahoo Finance through "
               f"{ds['as_of']['prices']}; Fama-French factors from Ken French's Data Library through "
               f"{ds['as_of']['factors']}; risk-free rate {rf['rf']:.2%}{rf_date}.")


pages = [
    st.Page(portfolio_builder, title="Portfolio Builder", icon="📊", default=True),
    st.Page("pages/1_Risk_Deep_Dive.py", title="Risk Deep Dive", icon="🧮"),
    st.Page("pages/2_Projections_and_Valuation.py", title="Projections & Valuation", icon="📈"),
    st.Page("pages/3_Stock_Ideas.py", title="Stock Ideas", icon="💡"),
    st.Page("pages/4_Data_Check.py", title="Data Check", icon="🔍"),
]
st.navigation(pages).run()
