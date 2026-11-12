# FactorFolio - Projections & Valuation page (Build Guide Step 6; outline section 4)
# The assumptions behind the homepage's bear/base/bull projections, with what-if controls,
# and a fair-value range for each stock from the Lecture 15 & 16 DCF and the Lecture 16 & 17
# peer comparables. Layout follows the Lecture 14 streamlit_app.py (st.tabs, st.subheader,
# st.dataframe); charts use Plotly Express (Lecture 5).
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data as fdata
from core import models, projections, theme, valuation

st.set_page_config(page_title="Projections & Valuation | FactorFolio", layout="centered")
st.title("Projections & Valuation")
st.write("The assumptions behind the bear, base and bull projections, plus a fair-value range for each stock.")

SCENARIO_COLORS = {"Bear": "#dc2626", "Base": "#2563eb", "Bull": "#16a34a"}
LABEL_COLORS = {"looks underpriced": "#16a34a", "fairly priced": "#6b7280", "looks overpriced": "#dc2626"}


def money(x, decimals=2):
    return "n/a" if pd.isna(x) else f"${x:,.{decimals}f}"


def pct(x, decimals=1):
    return "n/a" if pd.isna(x) else f"{x:.{decimals}%}"


def remember(key, default):
    """Widget values are cleared when you visit another page; keep a copy so they come back."""
    return st.session_state.get(f"saved_{key}", default)


# ---------- The homepage portfolio ----------
portfolio = st.session_state.get("portfolio", fdata.DEFAULT_PORTFOLIO)
amount = remember("amount", 10_000)
horizon = remember("horizon", 5)
st.caption("Your homepage portfolio: " + ", ".join(f"{t} {w:.0f}%" for t, w in portfolio.items())
           + f" | ${amount:,.0f} for {horizon} years. Change these on the Portfolio Builder page.")

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
port = result["portfolio"]
live = projections.risk_free_rate()
if not live["ok"]:
    st.warning(live["note"])

# ---------- Scenario and what-if controls ----------
scenario = st.radio("Market scenario for the next year", projections.SCENARIOS, horizontal=True,
                    index=projections.SCENARIOS.index(remember("scenario", "Base")), key="val_scenario")
st.session_state.saved_scenario = scenario          # the homepage follows this choice too

with st.expander("What-if controls"):
    st.caption("These change this page only. The homepage always uses today's Treasury yield, "
               "the 5.7% equity risk premium, the CAPM and the scenario in year 1 only.")
    rf = st.slider("Risk-free rate (%)", 0.0, 10.0, value=remember("pv_rf", round(live["rf"] * 100, 2)),
                   step=0.01, format="%.2f%%", key="pv_rf") / 100
    erp = st.slider("Equity risk premium (%)", 3.0, 8.0, value=remember("pv_erp", projections.ERP * 100),
                    step=0.1, format="%.1f%%", key="pv_erp") / 100
    g_lt_base = st.slider("Long-term growth in the base case (%)", 1.0, 5.0, value=remember("pv_glt", 3.0),
                          step=0.25, format="%.2f%%", key="pv_glt",
                          help="Bear uses 1 point less and bull 1 point more (outline 5.6).") / 100
    model_name = st.radio("Projection model", ["CAPM", "Fama-French 3-factor"], horizontal=True,
                          index=["CAPM", "Fama-French 3-factor"].index(remember("pv_model", "CAPM")), key="pv_model")
    every_year = st.toggle("Use the scenario return every year, not just year 1",
                           value=remember("pv_every", False), key="pv_every")
    use_alpha = st.toggle("Include historical alpha", value=remember("pv_alpha", False), key="pv_alpha",
                          help="Off by default: Lecture 12 warns that past alpha may not repeat.")
    for k, v in {"pv_rf": rf * 100, "pv_erp": erp * 100, "pv_glt": g_lt_base * 100, "pv_model": model_name,
                 "pv_every": every_year, "pv_alpha": use_alpha}.items():
        st.session_state[f"saved_{k}"] = v

# Model inputs: CAPM beta, or the FF3 market beta plus s x avg SMB + h x avg HML
if model_name == "CAPM":
    beta, extra, alpha, alpha_p = port["beta"], 0.0, port["alpha"], port["capm"]["alpha_p"]
else:
    beta = port["ff3_beta"]
    extra = projections.ff3_extra(port["s"], port["h"])
    alpha, alpha_p = port["ff3_alpha"], port["ff3"]["alpha_p"]
if use_alpha:
    extra += 12 * alpha                       # monthly alpha -> yearly

tabs = st.tabs(["Projections", "DCF", "Peer comparables", "Fair value"])

# ---------- Tab 1: projections ----------
with tabs[0]:
    st.subheader("All three scenarios")
    projs = {s: projections.project(amount, horizon, beta, rf, s, erp=erp, every_year=every_year, extra=extra)
             for s in projections.SCENARIOS}
    long = []
    for s, p in projs.items():
        for series in ["Your portfolio", "SPY"]:
            long.append(pd.DataFrame({"Year": p["path"]["Year"], "Value": p["path"][series],
                                      "Scenario": s, "Series": series}))
    long = pd.concat(long)

    # Six checkboxes pick which lines to draw. Six lines at once is cluttered, so the default shows your
    # portfolio in all three scenarios plus SPY in the base case as the yardstick.
    shown = []
    for series, label in [("Your portfolio", "Your portfolio"), ("SPY", "SPY (dotted)")]:
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            st.markdown(f"**{label}:**")
            for s in projections.SCENARIOS:
                key = f"show_{'port' if series == 'Your portfolio' else 'spy'}_{s.lower()}"
                default = series == "Your portfolio" or s == "Base"
                if st.checkbox(s, value=remember(key, default), key=key):
                    shown.append((s, series))
                st.session_state[f"saved_{key}"] = st.session_state[key]   # keep the choice across pages
    if not shown:
        st.info("Tick at least one box to show a line.")
    else:
        picked = pd.MultiIndex.from_frame(long[["Scenario", "Series"]]).isin(shown)
        fig = px.line(long[picked], x="Year", y="Value", color="Scenario", line_dash="Series", markers=True,
                      color_discrete_map=SCENARIO_COLORS,
                      line_dash_map={"Your portfolio": "solid", "SPY": "dot"})
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickprefix="$",
                          yaxis_tickformat=",.0f", yaxis_title="Value ($)", legend_title_text="",
                          legend=dict(orientation="h", x=0, y=1.15))
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Solid lines: your portfolio. Dotted lines: the same money in SPY. Colors: red bear, "
                   "blue base, green bull.")

    summary = pd.DataFrame([{"Scenario": s, "Your portfolio": money(p["portfolio_value"], 0),
                             "SPY": money(p["spy_value"], 0),
                             "Difference": f"{'+' if p['difference'] >= 0 else '-'}${abs(p['difference']):,.0f}"}
                            for s, p in projs.items()])
    st.dataframe(summary, hide_index=True, width="stretch")

    # Professor's check-in feedback: beta is an estimate, so show a range for it and the worst, most
    # likely and best case it leads to. The low and high beta are the regression's 95% confidence interval.
    st.subheader("Worst, most likely and best case")
    if model_name == "CAPM":
        beta_low, beta_high = port["beta_low"], port["beta_high"]
    else:
        beta_low, beta_high = port["ff3_beta_low"], port["ff3_beta_high"]
    br = projections.beta_range(amount, horizon, beta, beta_low, beta_high, rf, erp=erp, every_year=every_year,
                                extra=extra)
    case_cols = st.columns(3)
    for col, key, title, color in zip(case_cols, ["worst", "likely", "best"],
                                      ["Worst case", "Most likely", "Best case"], ["red", "blue", "green"]):
        case = br[key]
        beta_words = {"Low beta": "low beta", "Most likely beta": "estimated beta", "High beta": "high beta"}
        col.metric(title, money(case["value"], 0),
                   f"{case['scenario'].lower()} market, {beta_words[case['row']]} {br['betas'][case['row']]:.2f}",
                   delta_color=color, delta_arrow="off", border=True)
    st.dataframe(theme.highlight_cases(projections.beta_range_table(br)), hide_index=True, width="stretch")
    st.caption(f"Beta is an estimate from {port['n_months']} months of data, so the low and high beta are the "
               f"regression's 95% confidence interval ({beta_low:.2f} to {beta_high:.2f}, about beta ± 2 standard "
               f"errors, from statsmodels' conf_int()). Each cell is your portfolio's value in {horizon} "
               f"year{'s' if horizon > 1 else ''}. Most likely: the base market with your estimated beta; worst and "
               "best: the lowest and highest cells.")
    if br["worst"]["row"] == "Low beta" and not every_year:
        st.caption("Why is the low beta the worst case? Only year 1 is a bear year. The base years after it reward "
                   "market risk (the CAPM's more risk, more expected return), so over several years a lower beta "
                   "ends lower. With a 1-year horizon, a high beta would be the worst case in a bear market.")
    elif br["worst"]["row"] == "High beta":
        st.caption("In a bear market a higher beta falls further, so the high beta is the worst case here.")

    st.subheader("Assumptions")
    market = projs["Base"]["market"]
    sources = {"Bear": "25th percentile, 1927 to 2025", "Base": f"Rf {rf:.2%} + ERP {erp:.1%}",
               "Bull": "75th percentile, 1927 to 2025"}
    assumptions = pd.DataFrame([{"Scenario": s, "Market return (year 1)": pct(market[s]),
                                 "Your portfolio (year 1)": pct(projs[s]["portfolio_return_first"]),
                                 "Source": sources[s]} for s in projections.SCENARIOS])
    st.dataframe(assumptions, hide_index=True, width="stretch")
    st.caption("Bear and bull: percentiles of annual U.S. stock market returns from Ken French's Data Library. "
               "Base: the risk-free rate (Rf) plus the equity risk premium (ERP), as in the Lecture 15 notebook.")
    if model_name == "CAPM":
        st.write(f"**CAPM:** your return = risk-free rate + beta × (market return - risk-free rate), "
                 f"with beta = {beta:.2f}.")
    else:
        sc = projections.load_scenarios()
        st.write(f"**Fama-French 3-factor:** your return = risk-free rate + {beta:.2f} × (market return - "
                 f"risk-free rate) + size loading {port['s']:.2f} × average SMB ({sc['avg_smb']:.1%}) + value "
                 f"loading {port['h']:.2f} × average HML ({sc['avg_hml']:.1%}). The size and value add-on is "
                 f"{projections.ff3_extra(port['s'], port['h']):+.2%} a year.")
    sig = "significant" if alpha_p < models.SIGNIFICANCE else "not significant"
    if use_alpha:
        st.write(f"Historical alpha is **on**: {alpha:.2%} a month ({12 * alpha:+.1%} a year, p = {alpha_p:.2f}, "
                 f"{sig}), added every year.")
    else:
        st.caption(f"Historical alpha is off. Your portfolio's {model_name} alpha was {alpha:.2%} a month "
                   f"(p = {alpha_p:.2f}, {sig} at the 5% level).")
    st.caption(f"Path: {'the scenario return every year' if every_year else 'the scenario return in year 1, then the base return'}. "
               f"Risk-free rate source: {live['note']} These are scenarios, not predictions.")

# ---------- Fair value for every stock ----------
with st.spinner("Looking up analyst EPS forecasts for your stocks and similar companies..."):
    val = valuation.value_portfolio(result["stocks"], result["weights"], rf, scenario, erp=erp, g_lt_base=g_lt_base)
if val["error"]:
    for tab in tabs[1:]:
        with tab:
            st.warning(val["error"] + " Try again in a few minutes.")
    st.stop()
table = val["table"]
g_lt = val["g_lt"]
note = valuation.forecast_note(val)

# ---------- Tab 2: DCF ----------
with tabs[1]:
    st.subheader("Discounted cash flow (Lecture 15)")
    if note:
        (st.warning if note[0] == "warning" else st.info)(note[1])
    st.write("Each stock's value is today's worth of its next two years of analyst EPS forecasts, plus a "
             "growing perpetuity from year 3. The discount rate is the CAPM cost of equity.")
    st.latex(r"\begin{aligned} P &= \frac{EPS_1}{1+r_e} + \frac{EPS_2}{(1+r_e)^2} \\ "
             r"&\quad + \frac{EPS_2(1+g)}{(1+r_e)^2\,(r_e - g_{LT})} \end{aligned}")   # two lines fit a phone
    st.caption(f"r_e = {rf:.2%} + beta × {erp:.1%}; g = EPS_2 / EPS_1 - 1; long-term growth g_LT = "
               f"{g_lt['Bear']:.2%} (bear), {g_lt['Base']:.2%} (base), {g_lt['Bull']:.2%} (bull).")
    st.write("**Inputs**")
    dcf_inputs = pd.DataFrame({
        "Stock": table.index,
        "EPS 0y": [money(x) for x in table["eps1"]],
        "EPS +1y": [money(x) for x in table["eps2"]],
        "g": [pct(x) for x in table["g"]],
        "Our beta": [f"{x:.2f}" for x in table["beta"]],
        "Yahoo beta": ["n/a" if x is None or pd.isna(x) else f"{x:.2f}" for x in table["yahoo_beta"]],
        "Cost of equity": [pct(x, 2) for x in table["re"]],
    })
    st.dataframe(dcf_inputs, hide_index=True, width="stretch")
    st.write("**Values**")
    dcf_values = pd.DataFrame({
        "Stock": table.index,
        "Price": [money(x) for x in table["price"]],
        "Bear": [money(x) for x in table["dcf_bear"]],
        "Base": [money(x) for x in table["dcf_base"]],
        "Bull": [money(x) for x in table["dcf_bull"]],
        f"{scenario} vs. price": [pct(d / p - 1) if pd.notna(d) and pd.notna(p) else "n/a"
                                  for d, p in zip(table["dcf"], table["price"])],
    })
    st.dataframe(dcf_values, hide_index=True, width="stretch")
    notes = table[table["flags"] != ""]["flags"]
    for t, f in notes.items():
        st.markdown(f"- **{t}:** {f}".replace("$", "\\$"))
    st.caption("Beta is our own 60-month regression beta; Yahoo's is shown only as a cross-check. A stock is "
               "left out of the DCF when an EPS forecast is at or below zero, forecasts are missing, or the cost "
               "of equity is at or below long-term growth (check 14).")
    with st.expander("Look at the whole analyst EPS table (check 17)"):
        pick = st.selectbox("Stock", list(table.index), key="eps_pick")
        eps_table = val["eps_tables"].get(pick)
        if val["source"].get(pick) == "snapshot":
            st.write(f"Yahoo isn't sending the live table right now. The saved backup from "
                     f"{val['forecasts']['snapshot_date']} has EPS {table.at[pick, 'eps1']:.2f} for '0y' and "
                     f"{table.at[pick, 'eps2']:.2f} for '+1y'.")
        elif eps_table is None or eps_table.empty:
            st.write("Yahoo Finance didn't send analyst EPS forecasts for this stock.")
        else:
            st.dataframe(eps_table, width="stretch")
            st.caption("'0y' is the current fiscal year and '+1y' is next fiscal year (Lecture 16). The DCF "
                       "reads the 'current' column of those two rows.")

# ---------- Tab 3: peer comparables ----------
with tabs[2]:
    st.subheader("Peer comparables (Lecture 16)")
    if note:
        st.caption(note[1])
    st.write("Price a stock the way the market prices similar companies: predicted price = the peers' median "
             "P/E × the stock's next-year EPS. Peers are other S&P 100 stocks in the same industry, or the "
             "same sector when there are fewer than 3, keeping the 6 with the closest beta.")
    comps_view = pd.DataFrame({
        "Stock": table.index,
        "Peers": [f"{n}, same {b}" for n, b in zip(table["n_peers"], table["peer_basis"])],
        "Median P/E": [f"{x:.1f}" if pd.notna(x) else "n/a" for x in table["median_pe"]],
        "Mean P/E": [f"{x:.1f}" if pd.notna(x) else "n/a" for x in table["mean_pe"]],
        "Value (median)": [money(x) for x in table["comps_median"]],
        "Value (mean)": [money(x) for x in table["comps_mean"]],
    })
    st.dataframe(comps_view, hide_index=True, width="stretch")
    cross = pd.DataFrame({
        "Stock": table.index,
        "Price": [money(x) for x in table["price"]],
        "Own P/E": [f"{x:.1f}" if pd.notna(x) else "n/a" for x in table["own_pe"]],
        "Yahoo forward P/E": ["n/a" if x is None or pd.isna(x) else f"{x:.1f}" for x in table["yahoo_forward_pe"]],
    })
    st.dataframe(cross, hide_index=True, width="stretch")
    st.caption("Every P/E is today's price / next-year EPS (+1y), for the stock and its peers alike. Yahoo's "
               "forward P/E is only a cross-check because it can use a different EPS (Lecture 16). Peers with "
               "missing or negative EPS are dropped before the median and mean are taken.")

    pick = st.selectbox("See one stock's peers", list(table.index), key="peer_pick")
    peer_df = val["peers"][pick]
    row = table.loc[pick]
    if peer_df.empty:
        st.write("No peers found for this stock in the S&P 100 list.")
    else:
        show = peer_df.copy()
        show["Price"] = [money(x) for x in show["Price"]]
        show["EPS next year"] = [money(x) for x in show["EPS next year"]]
        show["P/E"] = [f"{x:.1f}" if pd.notna(x) else "dropped" for x in peer_df["P/E"]]
        st.dataframe(show, hide_index=True, width="stretch")
        usable_pes = peer_df[peer_df["P/E"] > 0]
        if not usable_pes.empty:
            box = px.box(usable_pes, x="P/E", points="all", hover_data=["Ticker"])
            if pd.notna(row["own_pe"]):
                box.add_vline(x=row["own_pe"], line_dash="dash", line_color="#2563eb",
                              annotation_text=f"{pick} {row['own_pe']:.1f}")
            box.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(box, width="stretch", config={"displayModeBar": False})
            st.caption(f"The box shows the peers' P/E ratios (the Lectures 1 to 3 boxplot): a peer far from the "
                       f"others pulls the mean but not the median. Dashed line: {pick}'s own P/E.")
        if pd.notna(row["comps_median"]):
            st.markdown((f"{pick}: {row['median_pe']:.1f} (median peer P/E) × {money(row['eps2'])} (next-year EPS) = "
                         f"**{money(row['comps_median'])}**. With the mean P/E of {row['mean_pe']:.1f}: "
                         f"{money(row['comps_mean'])}. Today's price: {money(row['price'])}.").replace("$", "\\$"))
        else:
            st.markdown(f"No comparables value for {pick}: {row['flags'] or 'not enough data'}.".replace("$", "\\$"))

# ---------- Tab 4: fair-value range ----------
with tabs[3]:
    st.subheader("Fair-value range")
    summary = val["portfolio"]
    st.info(valuation.explain(summary, table).replace("$", "\\$"))
    valued = table[table["label"] != "left out"]
    if not valued.empty:
        low_up = valued["low"] / valued["price"] - 1
        high_up = valued["high"] / valued["price"] - 1
        fig = go.Figure()
        for label, color in LABEL_COLORS.items():
            sel = valued["label"] == label
            if sel.any():
                fig.add_trace(go.Bar(
                    y=valued.index[sel], x=(high_up - low_up)[sel].clip(lower=0.01), base=low_up[sel],
                    orientation="h", name=label, marker_color=color,
                    customdata=valued.loc[sel, ["low", "high", "price"]].values,
                    hovertemplate="%{y}: $%{customdata[0]:,.2f} to $%{customdata[1]:,.2f}"
                                  " (price $%{customdata[2]:,.2f})<extra></extra>"))
        fig.add_vrect(x0=-0.05, x1=0.05, fillcolor="#9ca3af", opacity=0.15, line_width=0)
        fig.add_vline(x=0, line_color=theme.colors()["ink"], annotation_text="today's price", annotation_position="bottom right")
        fig.update_layout(height=120 + 45 * len(valued), margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_tickformat=".0%", xaxis_title="Fair value vs. today's price",
                          legend=dict(orientation="h", x=0, y=-0.35), barmode="overlay")
        fig.update_yaxes(categoryorder="array", categoryarray=list(valued.index), autorange="reversed")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Each bar runs from the lower to the higher of the DCF and comparables values, shown as a % "
                   "above or below today's price. The grey band is ±5%. A stock looks underpriced when its price "
                   "is more than 5% below the low end, and overpriced when it is more than 5% above the high end.")

    fv = pd.DataFrame({
        "Stock": table.index,
        "Label": table["label"].values,
        "Price": [money(x) for x in table["price"]],
        f"DCF ({scenario.lower()})": [money(x) for x in table["dcf"]],
        "Comps (median)": [money(x) for x in table["comps_median"]],
        "Middle vs. price": [pct(x) for x in table["upside"]],
    })
    st.dataframe(fv, hide_index=True, width="stretch")

    st.subheader("Implied return vs. CAPM")
    st.write("The implied return is the yearly return that makes today's price fair: EPS / price + long-term "
             "growth (Lecture 16). The CAPM cost of equity is the return investors should require for the "
             "stock's risk.")
    ir = pd.DataFrame({
        "Stock": table.index,
        "Implied (E/P + g)": [pct(x, 2) for x in table["implied_return"]],
        "CAPM cost of equity": [pct(x, 2) for x in table["re"]],
        "Gap": [pct(a - b, 2) if pd.notna(a) else "n/a" for a, b in zip(table["implied_return"], table["re"])],
    })
    st.dataframe(ir, hide_index=True, width="stretch")
    st.caption("A negative gap means the market accepts a lower return than the CAPM asks for: today's price "
               "already counts on more growth than the DCF assumes, so the DCF value comes out below the price. "
               "A positive gap means the opposite.")

backup = (f" When Yahoo doesn't answer, forecasts come from the saved backup of {val['forecasts']['snapshot_date']}."
          if val["forecasts"]["snapshot_date"] else "")
st.caption(f"For education only, not investment advice. Prices from Yahoo Finance as of {val['prices_as_of']}; "
           f"analyst EPS forecasts from Yahoo Finance (eps_trend), cached for a day.{backup}")
