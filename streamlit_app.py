# FactorFolio - AC420 project (Jasper Lin)
# Homepage: the Portfolio Builder (Build Guide Step 4: inputs, risk score, allocation chart).
# Layout follows the Lecture 14 streamlit_app.py from Assignment #4; st.navigation names the pages.
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data as fdata
from core import models, scoring

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
    scored = scoring.risk_score(port)
    spy = scoring.risk_score(result["benchmark"]) if result["benchmark"] else None
    words = scoring.explain(port, scored)
    fit = scoring.comfort_fit(scored["score"], comfort)
    st.session_state.analysis = {"score": scored["score"], "band": scored["band"]}

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
        st.caption("Financial health (Altman Z-score, Lecture 10) joins the score in Step 7; until then "
                   "the other weights are rescaled to add to 100%.")

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

    st.info(f"Bear, base and bull projections of your ${amount:,.0f} over {horizon} years vs. SPY: coming in Step 5.")
    st.page_link("pages/4_Data_Check.py", label="Look at the data behind this score", icon="🔍")
    st.caption(f"For education only, not investment advice. Prices from Yahoo Finance through "
               f"{ds['as_of']['prices']}; Fama-French factors from Ken French's Data Library through "
               f"{ds['as_of']['factors']}.")


pages = [
    st.Page(portfolio_builder, title="Portfolio Builder", icon="📊", default=True),
    st.Page("pages/1_Risk_Deep_Dive.py", title="Risk Deep Dive", icon="🧮"),
    st.Page("pages/2_Projections_and_Valuation.py", title="Projections & Valuation", icon="📈"),
    st.Page("pages/3_Stock_Ideas.py", title="Stock Ideas", icon="💡"),
    st.Page("pages/4_Data_Check.py", title="Data Check", icon="🔍"),
]
st.navigation(pages).run()
