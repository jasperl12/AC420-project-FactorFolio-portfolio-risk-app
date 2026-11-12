# FactorFolio
A Streamlit app that helps everyday investors with no finance background build stock portfolios
that match their comfort with risk.

Live app: https://ac420-factorfolio-jasperlin.streamlit.app
AC420 A1 Introduction to Financial Analytics (Fall 2026, Prof. Peter Wysocki), individual project by Jasper Lin.

## How to use
1. On the **Portfolio Builder** dashboard, use the control bar at the top: how much you'd invest, for how
   many years, your comfort with risk, and **Edit stocks** for your stocks and weights (up to 15).
2. Read the four KPI cards (risk score, value in N years vs. SPY, Sharpe ratio, fair value) and the
   plain-English line under them; try the bear, base and bull buttons on the growth card.
3. Open any card's "What does this mean?" button for the explanation, and add a suggested stock if your
   score is outside your comfort level.
4. Use the sidebar pages for the math, valuations, more stock ideas and the data behind it all.
5. Use **Text size** (- and +, in 10% steps from 70% to 200%) to make text and charts bigger or smaller, and
   flip the **Dark mode** switch for a dark theme. Both sit next to the title, or in the sidebar on other pages.

## Pages
- **Portfolio Builder (homepage dashboard):** a control bar for the inputs, four KPI cards, a 2 x 2 grid
  (risk score gauge vs. SPY, bear/base/bull growth vs. SPY, allocation by stock and sector, fair-value
  check) with explanations behind "What does this mean?" buttons, and "Balance your risk" suggestions
  with Add buttons when the score is outside your comfort level
- **Risk Deep Dive:** the score part by part, CAPM scatterplot with its OLS line, Sharpe ratios vs. SPY,
  CAPM vs. Fama-French results with p-values, the R-squared risk split, each stock's beta and R-squared, a size and value
  style map, and Altman Z-scores
- **Projections & Valuation:** all three scenarios at once with what-if controls, the worst, most likely
  and best case from a range for beta, DCF values, peer
  comparables, fair-value ranges and implied returns
- **Stock Ideas:** every S&P 100 stock tested as a 10% addition, with filters and Add buttons
- **Data Check:** the hygiene checklist, raw returns, summary statistics, a boxplot of extreme months,
  and where every input comes from and how fresh it is

## How the numbers are made
- **Risk score (0 to 100):** beta (40%), monthly volatility (25%) and the firm-specific share of risk,
  1 - R-squared (20%), each ranked against 66 S&P 100 stocks; plus financial health from Altman Z
  zones (15%). Bands: conservative 0 to 22, moderate 22 to 45, aggressive 45 to 100 (set in the Step 8
  calibration so SPY, at about 25, sits in the moderate band).
- **Regressions:** CAPM and Fama-French 3-factor OLS on the last 60 months of excess returns, merged with
  the factors on year-month (Lectures 12, 13 and 17). Alpha only counts when p < 0.05.
- **Projections:** CAPM expected return, risk-free rate + beta x (market return - risk-free rate), with
  the market at its 25th percentile (bear), risk-free rate + 5.7% (base) or 75th percentile (bull) in
  year 1, then the base return; SPY earns the market return (Lectures 14 and 15). Because beta is an
  estimate, the regression's 95% confidence interval (beta +/- about 2 standard errors) gives a low and a
  high beta; running every scenario with each one gives a worst, most likely and best case.
- **Fair value:** the Lecture 15 & 16 three-part DCF (analyst EPS for this year and next, CAPM cost of
  equity, long-term growth of 2%, 3% or 4%) and peer comparables (median P/E of similar S&P 100 stocks x
  next-year EPS, Lectures 16 & 17). Each stock gets a range from one to the other and a 5% band label:
  looks underpriced, fairly priced or looks overpriced. Never buy or sell.
- **Financial health:** Altman Z = 1.2 T1 + 1.4 T2 + 3.3 T3 + 0.6 T4 + 0.99 T5 (Lecture 10). Financial
  firms, utilities and real estate are left out because the model was built on manufacturing firms.
- **Sharpe ratio:** mean(R - rf) / std(R - rf) on the same 60 months, monthly and x sqrt(12) yearly
  (Lecture 18). Shown for the portfolio, SPY and every stock, but never used in the risk score or to rank
  suggestions, because a high past Sharpe ratio doesn't predict a high future one.
- **Suggestions:** each candidate is tested as 10% of the portfolio (0.9 x your returns + 0.1 x the
  stock's), rescored, and ranked by distance to your comfort band, sectors you don't own, then
  fair-value upside.

## Data
- **Yahoo Finance via yfinance (live):** monthly prices and the 30-year Treasury yield (^TYX); analyst
  EPS forecasts, company info and financial statements when Yahoo answers
- **Saved backups (`data/eps_snapshot.csv`, `data/financials_snapshot.csv`):** Yahoo often refuses
  forecast and statement requests from Streamlit Cloud, so the app falls back to these dated backups,
  saved in Colab, and says so
- **Ken French Data Library:** monthly Fama-French factors and annual market returns since 1927
- **WRDS monthly returns (Lecture 17 class file):** used offline only, to estimate the S&P 100 reference
  betas; no raw WRDS data is stored in this repo

## Testing
- `notebooks/known_answer_tests.ipynb` checks the code against answers from class: Lecture 12 & 13 CAPM
  and Fama-French (MSFT, XOM), the Lecture 17 merge fix, the Lecture 15 Coca-Cola DCF ($85.91), Lecture 16
  drills ($64.42; Amazon comps $202.45 and $140.32), the outline's projection example, Lecture 10's GM
  Altman Z (0.67), the Lecture 18 Sharpe ratio examples (0.19 and 0.208) and the beta range (the
  confidence interval equals beta +/- t x standard error; worst, most likely and best case checked by hand)
- `notebooks/calibration.ipynb` scores SPY, an all-utilities portfolio, three tech stocks and others, and
  checks that a suggestion moves an out-of-band portfolio into or toward its band
- The app was tested with bad tickers, ETFs, short histories, missing weights, more than 15 stocks and
  Yahoo refusing requests: each shows a warning instead of crashing

## Known limitations
- Betas, alphas and volatility come from the last 60 months; the future can differ
- The DCF is very sensitive when the cost of equity is close to long-term growth (low-beta stocks)
- Altman Z is a screening tool built on manufacturing firms, not a verdict
- Projections are scenarios, not predictions, and leave out past alpha by default
- Analyst forecasts can come from a backup up to a week or two old when Yahoo refuses live requests
- The Dark mode switch reloads the page, because Streamlit can't change the theme from Python; your
  inputs come back from a browser cookie that expires after 2 minutes
- Text size changes the page's root font size, so everything Streamlit sizes in rem grows with it, and
  charts are scaled in Python; the size resets when you open the app in a new tab

## Keeping the data fresh
- Every week or so, and before a demo: run `notebooks/yahoo_backup.ipynb` in Colab and upload the two
  CSV files it downloads to `data/`
- Every month or two: replace `data/ff_factors_monthly.csv` with Ken French's latest monthly
  "Fama/French 3 Factors" file (the Data Check page warns when it is getting old)

## Run it on your own computer
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Repo layout
- `streamlit_app.py`: homepage (Portfolio Builder) and page navigation
- `pages/`: Risk Deep Dive, Projections & Valuation, Stock Ideas, Data Check
- `core/`: calculations (`data`, `models`, `scoring`, `projections`, `valuation`, `suggestions`),
  `yahoo.py`, which handles Yahoo lookups that can be refused on Streamlit Cloud, and `theme.py`, the
  light/dark switch, text size control and theme-aware chart colors
- `data/`: small CSV files the app reads (see `data/README.md`)
- `notebooks/`: offline training (`training_wrds`), known-answer tests, calibration and the Yahoo backup

## Code credits
- Adapted from AC420 class notebooks by Prof. Peter Wysocki: Lecture 3 financial statements, Lecture 10
  Altman Z-score, Lecture 12 & 13 CAPM and Fama-French regressions, Lecture 15 & 16 DCF valuation,
  Lecture 16 & 17 comparables, Lecture 17 alpha regressions, Lecture 18 Sharpe ratio (Assignment #5),
  Lecture 5 Plotly charts, and the Lecture 14
  Streamlit starter app (Assignment #4)
- Dashboard layout and the light/dark switch use Streamlit's documented `st.popover`,
  `st.segmented_control`, bordered `st.container`, `st.metric(border=True)`, `st.context` and
  `st.html`; the switch stores the same theme choice as Streamlit's own Settings menu
- Written with AI assistance (Claude by Anthropic); my AI usage log records how AI helped with each part

For education only, not investment advice.
