# FactorFolio - Stock Ideas page (Build Guide Step 8; outline sections 4 and 5.7)
# Every S&P 100 stock you don't own, tested as a 10% addition: how it would change your risk
# score, its beta and R² (Lecture 12), style tilt (Lecture 13) and fair-value upside (Lectures 15
# and 16). Layout follows the Lecture 14 streamlit_app.py; the chart uses Plotly Express (Lecture 5).
import pandas as pd
import plotly.express as px
import streamlit as st

from core import data as fdata
from core import models, projections, scoring, suggestions, theme, yahoo

st.set_page_config(page_title="Stock Ideas | FactorFolio", layout="centered")
st.title("Stock Ideas")
st.write("Browse large U.S. stocks and see how adding each one would change your risk score.")

MAX_STOCKS = 15


def remember(key, default):
    return st.session_state.get(f"saved_{key}", default)


# ---------- The homepage portfolio, same steps as the homepage ----------
portfolio = st.session_state.get("portfolio", fdata.DEFAULT_PORTFOLIO)
comfort = remember("comfort", "Moderate")
low, high = scoring.COMFORT_RANGES[comfort]
added = st.session_state.pop("just_added", None)
if added:
    st.success(f"Added {', '.join(added)} at 10%{' each' if len(added) > 1 else ''}. Your other stocks were "
               "scaled down to make room; "
               "open the Portfolio Builder page to see your new score.")
    portfolio = st.session_state.portfolio
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
held = list(result["weights"].index)
prices_now = {t: ds["prices"][t].dropna().iloc[-1] for t in held if t in ds["prices"].columns}
with st.spinner("Looking up financial statements for the Altman Z-score..."):
    z_table = models.altman_table(held, result["weights"].to_dict(), prices_now)
scored = scoring.risk_score(result["portfolio"], health_points=models.health_points(z_table)["points"])
rf = projections.risk_free_rate()

st.caption("Your portfolio: " + ", ".join(f"{t} {w:.0f}%" for t, w in portfolio.items()) +
           f". Risk score {scored['score']:.0f} ({scored['risk_words']}); your comfort level is {comfort.lower()} "
           f"({low} to {high}). Change these on the Portfolio Builder page.")

try:
    with st.spinner("Testing each S&P 100 stock as a 10% addition to your portfolio..."):
        ideas = suggestions.test_scores(result["returns"], z_table, comfort, set(z_table["sector"].dropna()),
                                        rf["rf"])
except Exception as e:
    st.error(f"Couldn't download the S&P 100 prices right now ({e}). Try again in a few minutes.")
    st.stop()
ideas["helps"] = suggestions.helps(ideas, scored["score"], comfort)
ideas["change"] = ideas["new_score"] - scored["score"]
full = len(held) >= MAX_STOCKS

# ---------- Best fits ----------
top = suggestions.suggest(ideas, scored["score"], comfort)
if not top.empty:
    st.subheader("Best fits for your comfort level")
    st.write(f"Your score of {scored['score']:.0f} is outside the {comfort.lower()} range. These move it the most "
             "toward the range, then favor sectors you don't own and higher fair-value upside:")
    for t, row in top.iterrows():
        c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="center")
        c1.markdown(f"**{t}** {row['name']}  \n{row['sector']}" +
                    (" (a sector you don't own yet)" if row["new_sector"] else ""))
        c2.markdown(f"Score {scored['score']:.0f} → **{row['new_score']:.0f}**  \nbeta {row['beta']:.2f}")
        c3.button("Add", key=f"top_{t}", on_click=suggestions.add_to_holdings, args=([t],), disabled=full,
                  width="stretch")
else:
    st.success(f"Your score already fits your {comfort.lower()} comfort level, so every idea below is optional.")

# ---------- Filters and chart ----------
st.subheader("All ideas")
sectors = st.multiselect("Sectors (leave empty for all)", sorted(ideas["sector"].dropna().unique()), key="idea_sectors")
only_helpful = st.toggle("Only stocks that move my score toward my comfort level (or keep it there)",
                         key="idea_helpful")
view = ideas.copy()
if sectors:
    view = view[view["sector"].isin(sectors)]
if only_helpful:
    view = view[view["helps"]]

if view.empty:
    st.write("No stocks match these filters.")
else:
    plot = view.reset_index()
    plot["Fair-value upside"] = plot["upside"].clip(-1, 1.5)        # keep a few huge DCF values on the chart
    plot["New score"] = plot["new_score"].round(0)
    fig = px.scatter(plot, x="beta", y="Fair-value upside", color="sector", hover_name="name",
                     hover_data={"ticker": True, "New score": True, "label": True, "sector": False},
                     labels={"beta": "Beta (market risk)", "sector": "Sector"})
    stats, _ = suggestions.candidate_stats()
    values = suggestions.candidate_values(round(rf["rf"], 4), "Base")
    mine = stats.loc[[t for t in held if t in stats.index]].join(values)
    if not mine.empty:
        fig.add_scatter(x=mine["beta"], y=mine["upside"].fillna(0).clip(-1, 1.5), mode="markers+text",
                        text=list(mine.index), textposition="top center", name="You own",
                        marker=dict(symbol="star", size=13, color=theme.colors()["ink"]))
    fig.add_hline(y=0, line_color="#9ca3af")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.25, x=0, title_text=""))
    st.plotly_chart(theme.scaled(fig), width="stretch", config={"displayModeBar": False})
    st.caption("Up and to the left: lower market risk and more fair-value upside. Upside is the middle of each "
               "stock's DCF to comps range vs. today's price (base case); values above 150% are drawn at the top "
               "edge. Stars mark stocks you already own.")

    table = pd.DataFrame({
        "Stock": view.index,
        "New score": view["new_score"].round(0).astype(int).values,
        "Change": [f"{c:+.1f}" for c in view["change"]],
        "Sector": view["sector"].values,
        "Beta": view["beta"].round(2).values,
        "Upside": ["n/a" if pd.isna(u) else f"{u:.0%}" for u in view["upside"]],
        "Label": view["label"].values,
        "R²": view["r2"].round(2).values,
        "Sharpe": view["sharpe"].round(2).values,
        "Style": view["style"].values,
        "Name": view["name"].values,
    })
    st.write("Select rows in the table, then add them to your portfolio at 10% each.")
    st.caption("Sharpe is each stock's past monthly return per unit of risk (Lecture 18). Ideas are not ranked "
               "by it, because a high past Sharpe ratio doesn't predict a high future one.")
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="multi-row", key="ideas_table")
    picked = [table.iloc[i]["Stock"] for i in event.selection.rows]
    too_many = len(held) + len(picked) > MAX_STOCKS
    st.button(f"Add {', '.join(picked)} at 10% each" if picked else "Select stocks to add",
              on_click=suggestions.add_to_holdings, args=(picked,), disabled=not picked or too_many,
              type="primary")
    if too_many:
        st.caption(f"That would make more than {MAX_STOCKS} stocks, the most FactorFolio handles.")

with st.expander("How are these ideas picked?"):
    st.write("For every S&P 100 stock you don't own, FactorFolio builds a test portfolio: your stocks scaled down "
             "to 90% plus that stock at 10%. Its monthly return is 0.9 × your portfolio's return + 0.1 × the "
             "stock's return, and the test portfolio gets the same risk score as the homepage (beta, volatility, "
             "diversification and Altman Z health). The best fits are the stocks that move your score closest to "
             "your comfort range; ties go to sectors you don't own yet (spreading out company-specific risk, "
             "Lecture 12) and then to more fair-value upside (Lectures 15 and 16). Ideas are a starting point "
             "for your own research, not recommendations.")
backup = yahoo.snapshot_date()
st.caption("For education only, not investment advice. Betas use the last 60 months of Yahoo Finance prices; "
           f"fair values use the saved analyst forecasts{f' from {backup}' if backup else ''} and a risk-free "
           f"rate of {rf['rf']:.2%}.")
