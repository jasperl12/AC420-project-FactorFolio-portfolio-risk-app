# FactorFolio - AC420 project (Jasper Lin)
# Homepage: the Portfolio Builder dashboard (Build Guide Steps 4 to 8). A control bar holds the inputs,
# four KPI cards give the answer at a glance, and a 2 x 2 grid shows the risk gauge, the bear/base/bull
# growth chart, the allocation and the fair-value check. Longer explanations sit behind
# "What does this mean?" buttons, and balance-your-risk suggestions appear only when the score is
# outside your comfort level.
# Layout starts from the Lecture 14 streamlit_app.py from Assignment #4 (st.columns, st.metric);
# st.popover, st.segmented_control and bordered st.container come from the Streamlit docs.
# st.navigation names the pages; core/theme.py adds the light/dark switch.
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data as fdata
from core import models, projections, scoring, suggestions, theme, valuation, yahoo

st.set_page_config(page_title="FactorFolio", layout="centered")

MAX_STOCKS = 15
SECTORS_FILE = fdata.DATA_DIR / "sp100_tickers.csv"
CHART_CONFIG = {"displayModeBar": False}
CARD_CHART_HEIGHT = 250
# keeps the dashboard from stretching too wide on big monitors (81rem = 1300px at 100% text size; in rem
# so the page gets wider as the text gets bigger)
DASHBOARD_CSS = "<style>[data-testid='stMainBlockContainer'] {max-width: 81rem; padding-top: 3rem;}</style>"


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
        raw = row.get("Ticker")
        tickers = fdata.clean_tickers("" if pd.isna(raw) else str(raw))
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
    c = theme.colors()
    fig = px.line(path, x="Year", y=["Your portfolio", "SPY"], markers=True,
                  color_discrete_map={"Your portfolio": c["portfolio"], "SPY": c["spy"]})
    fig.update_traces(hovertemplate="%{fullData.name}: $%{y:,.0f}<extra></extra>")
    fig.update_traces(selector=dict(name="Your portfolio"), line_width=3)
    fig.update_traces(selector=dict(name="SPY"), line_dash="dash")   # still visible when the lines overlap
    fig.update_layout(height=CARD_CHART_HEIGHT, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis_title=None, xaxis_title=None, yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                      legend_title_text="", legend=dict(orientation="h", x=0, y=1.15))
    fig.update_xaxes(dtick=1)
    return fig


def gauge(score, spy_score):
    c = theme.colors()
    low_cut, high_cut = scoring.COMFORT_RANGES["Moderate"]       # band cutoffs (Step 8 calibration)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": " / 100", "font": {"size": 34}, "valueformat": ".0f"},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, low_cut, high_cut, 100]},
               "bar": {"color": c["ink"], "thickness": 0.25},
               "steps": [{"range": [0, low_cut], "color": c["low"]},
                         {"range": [low_cut, high_cut], "color": c["mid"]},
                         {"range": [high_cut, 100], "color": c["high"]}],
               "threshold": {"line": {"color": c["portfolio"], "width": 4}, "thickness": 0.9,
                             "value": spy_score if spy_score is not None else score}}))
    fig.update_layout(height=CARD_CHART_HEIGHT - 20, margin=dict(l=30, r=30, t=30, b=10))
    return fig


def grid_row():
    """Two cards side by side, or one above the other when the text size is above 130%."""
    if theme.scale() <= 1.3:
        return st.columns(2, gap="medium")
    return st.container(), st.container()


def card_header(title, button="What does this mean?"):
    """A card's title on the left and a small button on the right that opens the details."""
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown(f"**{title}**")
        return st.popover(button, type="tertiary", icon=":material/info:")


def stock_editor():
    """The editable stock table (inside the Edit stocks button). Returns the edited table."""
    if "holdings" not in st.session_state:
        st.session_state.holdings = pd.DataFrame({"Ticker": list(fdata.DEFAULT_PORTFOLIO),
                                                  "Weight (%)": [float(w) for w in fdata.DEFAULT_PORTFOLIO.values()]})
    if "holdings_editor" not in st.session_state:   # first time on this page: start from the saved table
        base = st.session_state.holdings.copy()
        # an empty table comes back with number columns; the editor needs text for tickers
        st.session_state.editor_base = base.astype({"Ticker": "object", "Weight (%)": "float64"})
    st.caption(f"Type a ticker and its weight (up to {MAX_STOCKS} stocks). Add rows at the bottom; "
               "select a row to delete it.")
    table = st.data_editor(
        st.session_state.editor_base, key="holdings_editor", num_rows="dynamic", hide_index=True, width=int(340 * theme.scale()),
        column_config={"Ticker": st.column_config.TextColumn("Ticker", help="Yahoo Finance symbol, e.g. AAPL"),
                       "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0,
                                                                   step=1.0, format="%.1f")})
    st.session_state.holdings = table
    return table


def control_bar():
    """The inputs in one bordered row: amount, years, comfort with risk and the stock table."""
    with st.container(border=True):
        # bigger text needs more room: one row of four, then two rows, then comfort and stocks on their own rows
        if theme.scale() <= 1.1:
            c1, c2, c3, c4 = st.columns([1.1, 1.3, 2.0, 1.0], vertical_alignment="bottom")
        else:
            c1, c2 = st.columns(2, vertical_alignment="bottom")
            if theme.scale() <= 1.5:
                c3, c4 = st.columns([2, 1], vertical_alignment="bottom")
            else:
                c3, c4 = st.container(), st.container()
        amount = c1.number_input("Amount to invest ($)", min_value=100, max_value=10_000_000, step=1_000,
                                 value=remember("amount", 10_000), key="amount")
        horizon = c2.slider("Time horizon (years)", 1, 10, value=remember("horizon", 5), key="horizon")
        comfort = c3.segmented_control(
            "Comfort with risk", list(scoring.COMFORT_RANGES), required=True, width="stretch",
            default=remember("comfort", "Moderate"), key="comfort",
            help="  \n".join(f"**{c}:** {scoring.COMFORT_HELP[c]}" for c in scoring.COMFORT_RANGES))
        with c4.popover("Edit stocks", icon=":material/edit:", width="stretch"):
            table = stock_editor()
        st.session_state.saved_amount, st.session_state.saved_horizon = amount, horizon
        st.session_state.saved_comfort = comfort

        weights = holdings_from_table(table)
        if weights:
            st.caption("Your stocks: " + " · ".join(f"{t} {w:g}%" for t, w in weights.items()))
        else:
            st.warning("Add at least one stock with a weight above 0 (use Edit stocks).")
            return amount, horizon, comfort, weights
        if len(weights) > MAX_STOCKS:
            st.warning(f"FactorFolio handles up to {MAX_STOCKS} stocks; only the first {MAX_STOCKS} are used.")
            weights = dict(list(weights.items())[:MAX_STOCKS])
        total = sum(weights.values())
        if abs(total - 100) > 0.01:
            w1, w2 = st.columns([4, 1], vertical_alignment="center")
            w1.warning(f"Your weights add to {total:.1f}%. The dashboard rescales them to 100%.")
            w2.button("Normalize to 100%", on_click=normalize_table, width="stretch")
    return amount, horizon, comfort, weights


def portfolio_builder():
    st.set_page_config(page_title="FactorFolio", layout="wide")
    st.html(DASHBOARD_CSS)

    if theme.scale() <= 1.3:
        head, controls = st.columns([3, 2], vertical_alignment="center")
    else:                             # very large text: the size and dark mode controls go under the title
        head, controls = st.container(), st.container()
    head.title("FactorFolio", anchor=False)
    head.caption("Build a stock portfolio that matches your comfort with risk, scored in plain English.")
    align = "right" if theme.scale() <= 1.3 else "left"
    with controls.container(horizontal=True, horizontal_alignment=align, vertical_alignment="center", gap="medium"):
        st.caption("Text size")
        theme.size_buttons(key="size_home")
        theme.switch(key="theme_home")

    amount, horizon, comfort, weights = control_bar()
    added = st.session_state.pop("just_added", None)
    if added:
        st.toast(f"Added {', '.join(added)} at 10%{' each' if len(added) > 1 else ''}; your other stocks were "
                 "scaled down to make room.", icon=":material/check_circle:")
    if not weights:
        return
    st.session_state.portfolio = weights          # shared with the other pages

    # ---------- Data and models (same steps as before; only the layout changed) ----------
    with st.spinner("Downloading prices and running the CAPM and Fama-French regressions..."):
        ds = fdata.build_dataset(list(weights), weights=weights)
    if ds["error"]:
        st.error(ds["error"])
        return
    usable = {t: w for t, w in weights.items() if t in ds["tickers"]}
    if not usable:
        st.error(fdata.no_data_message(ds))
        return
    problems = [c["detail"] for c in ds["checks"] if c["status"] in ("warn", "fail") and not c["check"].startswith("2.")]
    if problems:
        st.warning("  \n".join(problems))
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
    level = comfort.lower()
    low, high = scoring.COMFORT_RANGES[comfort]

    # the scenario buttons sit in the growth card, but the KPI row above it needs their value first
    scenario = st.session_state.get("scenario") or remember("scenario", "Base")
    rf = projections.risk_free_rate()
    proj = projections.project(amount, horizon, port["beta"], rf["rf"], scenario)
    st.session_state.analysis = {"score": scored["score"], "band": scored["band"], "beta": port["beta"],
                                 "health": health["points"], "scenario": scenario, "rf": rf["rf"]}

    # ---------- KPI row ----------
    if theme.scale() <= 1.0:          # bigger text: a 2 x 2 grid of cards, then one card per row, so nothing is cut off
        k1, k2, k3, k4 = st.columns(4)
    elif theme.scale() <= 1.5:
        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
    else:
        k1, k2, k3, k4 = (st.container() for _ in range(4))
    fit_words = {"fits": "Fits your comfort level", "above": "Riskier than your comfort",
                 "below": "Safer than your comfort"}[fit["status"]]
    fit_color = {"fits": "green", "above": "orange", "below": "blue"}[fit["status"]]
    spy_text = f" SPY (an S&P 500 fund) scores {spy['score']:.0f}." if spy else ""
    k1.metric("Risk score", f"{scored['score']:.0f} / 100", fit_words, delta_color=fit_color, delta_arrow="off",
              border=True, help=f"0 is the safest and 100 the riskiest. {scoring.bands_text().capitalize()}.{spy_text}")
    tie = abs(proj["difference_pct"]) < 0.0005      # under 0.05%: call it a tie instead of showing -0.0%
    k2.metric(f"In {horizon} years ({scenario.lower()} case)", f"${proj['portfolio_value']:,.0f}",
              "about the same as SPY" if tie else f"{signed_money(proj['difference'])} vs. SPY",
              delta_color="off" if tie else "normal", delta_arrow="off" if tie else "auto", border=True,
              help=(f"What ${amount:,.0f} could grow to, from the CAPM and the {scenario.lower()} market scenario. "
                    f"SPY would reach ${proj['spy_value']:,.0f}. A scenario, not a prediction.").replace("$", "\\$"))
    spy_sharpe = f"SPY {result['benchmark']['sharpe_annual']:.2f}" if result["benchmark"] else None
    k3.metric("Sharpe ratio (past 5 years)", f"{port['sharpe_annual']:.2f}", spy_sharpe, delta_color="gray",
              delta_arrow="off", border=True,
              help="Past yearly extra return per unit of risk (Lecture 18). It isn't part of the risk score, "
                   "because a high past Sharpe ratio doesn't predict a high future one.")
    fair_slot = k4.empty()
    fair_slot.metric("Fair value", "...", "checking prices", delta_color="gray", delta_arrow="off", border=True)

    # ---------- What it means (and, when needed, what to do) ----------
    banner = st.container(border=True)
    icon = {"fits": ":green[:material/check_circle:]", "above": ":orange[:material/warning:]",
            "below": ":blue[:material/info:]"}[fit["status"]]
    banner.markdown(f"{icon} **{words['headline']}** {fit['message']}")
    ideas_slot = banner.container()

    # ---------- Row 1: risk gauge | growth chart ----------
    left, right = grid_row()
    with left.container(border=True, height="stretch"):
        more = card_header("Risk score vs. SPY")
        st.plotly_chart(theme.scaled(gauge(scored["score"], spy["score"] if spy else None)), width="stretch",
                        config=CHART_CONFIG)
        if spy:
            st.caption(f"Blue line: SPY scores {spy['score']:.0f}. Green, yellow and red: {scoring.bands_text()}.")
        with more:
            st.markdown(f"**{words['headline']}**")
            for part in ("market", "diversification", "volatility", "health"):
                st.write(words[part])
            st.markdown("**How this score is built**")
            st.write("Each part compares your portfolio with 66 large U.S. stocks (S&P 100 members in the class "
                     "WRDS data). A part's score is the share of those stocks your portfolio is riskier than.")
            breakdown = pd.DataFrame([{"Part": c["name"], "Your portfolio": c["value_text"],
                                       "Part score": round(c["score"]), "Weight": f"{c['weight']:.0%}",
                                       "Points": round(c["points"], 1)} for c in scored["components"]])
            st.dataframe(breakdown, hide_index=True, width="stretch")
            st.caption("Financial health uses Altman's Z-score (Lecture 10): a company in the safe zone adds 0 "
                       "points, grey 50 and distress 100, weighted by how much of it you hold. Banks, insurers, "
                       "utilities and real estate firms are left out. SPY's health comes from the S&P 100 backup, "
                       "weighted by market value.")
            fin_note = yahoo.backup_note(z_table["source"].to_dict(), "financial statements",
                                         yahoo.financials_snapshot_date())
            if fin_note:
                st.caption(fin_note)
            st.page_link("pages/1_Risk_Deep_Dive.py", label="See every part of the score", icon="🧮")

    with right.container(border=True, height="stretch"):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.markdown("**Growth vs. SPY**")
            scenario = st.segmented_control(
                "Market scenario for the next year", projections.SCENARIOS, required=True, key="scenario",
                default=remember("scenario", "Base"), label_visibility="collapsed")
            more = st.popover("What does this mean?", type="tertiary", icon=":material/info:")
        st.session_state.saved_scenario = scenario
        if not rf["ok"]:
            st.warning(rf["note"])
        st.plotly_chart(theme.scaled(growth_chart(proj["path"])), width="stretch", config=CHART_CONFIG)
        st.caption(projections.SCENARIO_HELP[scenario])
        # beta is an estimate, so show what its 95% confidence interval means for the outcome
        br = projections.beta_range(amount, horizon, port["beta"], port["beta_low"], port["beta_high"], rf["rf"])
        st.caption(f":material/swap_vert: **Range for your beta** ({port['beta_low']:.2f} to {port['beta_high']:.2f}): "
                   f"worst \\${br['worst']['value']:,.0f}, most likely \\${br['likely']['value']:,.0f}, "
                   f"best \\${br['best']['value']:,.0f} in {horizon} year{'s' if horizon > 1 else ''}.")
        if abs(port["beta"] - 1) < 0.1:
            move_word = "about as much as"
        else:
            move_word = "more than" if port["beta"] > 1 else "less than"
        with more:
            st.markdown(projections.explain(proj).replace("$", "\\$"))   # \$ stops Streamlit reading $...$ as math
            st.markdown("**How these projections work**")
            st.write(f"Each scenario sets the stock market's return for the next year. Your portfolio's return "
                     f"comes from the CAPM: risk-free rate + beta × (market return - risk-free rate). With a beta "
                     f"of {port['beta']:.2f}, your portfolio tends to move {move_word} the market. "
                     "SPY's return is the market return itself.")
            st.write("After the first year, both go back to the base return for the rest of the time horizon. "
                     "Past alpha is left out because it may not repeat (Lecture 12).")
            st.dataframe(projections.assumptions_table(proj), hide_index=True, width="stretch")
            st.markdown("**Worst, most likely and best case**")
            st.write(f"Your beta of {port['beta']:.2f} is an estimate from {port['n_months']} months, so the "
                     f"regression's 95% confidence interval ({port['beta_low']:.2f} to {port['beta_high']:.2f}) gives "
                     "a low and a high beta. Each cell is your portfolio's value at the end of your time horizon.")
            st.dataframe(theme.highlight_cases(projections.beta_range_table(br)), hide_index=True, width="stretch")
            st.caption(f"Risk-free rate: {rf['note']} Equity risk premium: {projections.ERP:.1%} (Lecture 15). "
                       "Bear and bull: the 25th and 75th percentiles of annual U.S. market returns from 1927 to "
                       "2025 (Ken French Data Library). These are scenarios, not predictions.")
            st.page_link("pages/2_Projections_and_Valuation.py", label="Try your own assumptions", icon="📈")

    # ---------- Row 2: allocation | fair value ----------
    left, right = grid_row()
    sectors = sector_lookup()
    w = result["weights"]
    alloc = pd.DataFrame({"Ticker": w.index, "Weight": w.values,
                          "Sector": [sectors.get(t, "Other") for t in w.index]})
    alloc["Amount"] = [f"${x:,.0f}" for x in alloc["Weight"] * amount]
    by_sector = alloc.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
    with left.container(border=True, height="stretch"):
        more = card_header("Allocation", "See amounts")
        pie = px.pie(alloc, names="Ticker", values="Weight", hole=0.4, hover_data=["Amount", "Sector"])
        pie.update_traces(textinfo="label+percent")
        pie.update_layout(height=CARD_CHART_HEIGHT, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(theme.scaled(pie), width="stretch", config=CHART_CONFIG)
        st.caption("By sector: " + ", ".join(f"{s} {v:.0%}" for s, v in by_sector.items()))
        if by_sector.iloc[0] > 0.5:
            st.caption(f":orange[:material/warning:] Over half of your money is in one sector "
                       f"({by_sector.index[0]}), so news about that industry can move your whole portfolio.")
        dropped = [t for t in weights if t not in usable]
        if dropped:
            st.caption("Left out because there was not enough data: " + ", ".join(dropped))
        with more:
            st.dataframe(alloc.assign(Weight=[f"{x:.1%}" for x in alloc["Weight"]]), hide_index=True,
                         width="stretch")

    with right.container(border=True, height="stretch"):
        more = card_header("Fair value check")
        with st.spinner("Looking up analyst EPS forecasts..."):
            val = valuation.value_portfolio(result["stocks"], result["weights"], rf["rf"], scenario)
        if val["error"]:
            st.warning(val["error"] + " The fair-value check will come back when Yahoo Finance responds.")
            fair_slot.metric("Fair value", "n/a", "Yahoo didn't respond", delta_color="gray", delta_arrow="off",
                             border=True)
        else:
            fv = val["table"]
            card = pd.DataFrame({
                "Stock": fv.index,
                "Price": [f"${p:,.2f}" if pd.notna(p) else "n/a" for p in fv["price"]],
                "Fair-value range": [f"${lo:,.0f} to ${hi:,.0f}" if pd.notna(lo) else "n/a"
                                     for lo, hi in zip(fv["low"], fv["high"])],
                "Label": fv["label"].values,
            })
            st.dataframe(card, hide_index=True, width="stretch", height=min(36 * (len(card) + 1) + 3, 290))
            st.caption(f"Prices as of {val['prices_as_of']}. A check, not a buy signal.")
            summary = val["portfolio"]
            label_color = {"looks underpriced": "green", "looks overpriced": "red"}.get(summary["label"], "gray")
            if summary["label"] == "left out":
                kpi_label, kpi_value, delta = "Fair value", "n/a", "no stock could be valued"
            else:   # "Your stocks look" + "Underpriced" keeps the hedged wording of the fair-value labels
                kpi_label = "Your stocks look"
                kpi_value = summary["label"].replace("looks ", "").capitalize()
                delta = f"fair value {summary['upside']:+.0%} vs. prices"
            fair_slot.metric(kpi_label, kpi_value, delta, delta_color=label_color,
                             delta_arrow="off", border=True,
                             help="Each stock's range runs from a DCF value (Lecture 15) to what similar companies "
                                  "trade at (peer comparables, Lecture 16). A check, not a buy signal.")
            with more:
                note = valuation.forecast_note(val)
                if note:
                    (st.warning if note[0] == "warning" else st.caption)(note[1])
                st.write(valuation.explain(summary, fv).replace("$", "\\$"))
                st.caption(f"Each range runs from the DCF value ({scenario.lower()} case, long-term growth "
                           f"{val['g_lt'][scenario]:.0%}; Lecture 15) to the peer comparables value (median P/E of "
                           f"similar S&P 100 stocks × next-year EPS; Lecture 16).")
                st.page_link("pages/2_Projections_and_Valuation.py", label="See the DCF and peer comparables",
                             icon="📈")

    # ---------- Balance your risk (outline 5.7; only when the score is outside the comfort level) ----------
    if fit["status"] != "fits":
        with ideas_slot:
            try:
                with st.spinner("Finding stocks that would balance your risk..."):
                    ideas = suggestions.test_scores(result["returns"], z_table, comfort,
                                                    set(z_table["sector"].dropna()), rf["rf"])
                top = suggestions.suggest(ideas, scored["score"], comfort)
            except Exception as e:
                top = None
                st.caption(f"Suggestions are unavailable right now ({e}). Try again in a few minutes.")
            if top is not None and top.empty:
                st.caption("None of the S&P 100 stocks moves your score toward your comfort level when added at "
                           "10%. Try changing your weights instead.")
            elif top is not None:
                direction = "raise" if fit["status"] == "below" else "lower"
                st.caption(f"Adding one of these at 10% would {direction} your score toward the {level} range "
                           f"({low} to {high}):")
                full = len(weights) >= MAX_STOCKS
                with st.container(horizontal=True, vertical_alignment="center"):
                    for t, row in top.iterrows():
                        new_sector = ", a sector you don't own yet" if row["new_sector"] else ""
                        st.button(f"Add {t} (score {row['new_score']:.0f})", key=f"add_{t}",
                                  on_click=suggestions.add_to_holdings, args=([t],), disabled=full,
                                  help=f"{row['name']}: {row['sector']}{new_sector}. Beta {row['beta']:.2f}.")
                    st.page_link("pages/3_Stock_Ideas.py", label="More ideas", icon="💡")
                if full:
                    st.caption(f"You already hold {MAX_STOCKS} stocks, the most FactorFolio handles.")

    # ---------- Footer ----------
    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        st.page_link("pages/1_Risk_Deep_Dive.py", label="Risk Deep Dive", icon="🧮")
        st.page_link("pages/2_Projections_and_Valuation.py", label="Projections & Valuation", icon="📈")
        st.page_link("pages/3_Stock_Ideas.py", label="Stock Ideas", icon="💡")
        st.page_link("pages/4_Data_Check.py", label="Data Check", icon="🔍")
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
theme.restore_after_reload()                      # inputs saved just before a light/dark switch
theme.apply_request([p.url_path for p in pages])  # the switch was flipped: save, store the theme, reload
page = st.navigation(pages)
theme.apply_scale()                               # the text size chosen with the - / + buttons
if page.url_path != "":                           # the homepage has its own controls next to the title
    st.sidebar.caption("Text size")
    theme.size_buttons(st.sidebar, key="size_sidebar")
    theme.switch(st.sidebar, key="theme_sidebar")
page.run()
