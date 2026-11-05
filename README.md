# FactorFolio
A Streamlit app that helps beginners build stock portfolios that match their comfort with risk.
Live app: https://ac420-factorfolio-jasperlin.streamlit.app

## How to use
1. Enter an amount, stocks and weights on the homepage.
2. Read the 0 to 100 risk score and try the bear, base and bull buttons.
3. Use the sidebar pages for the math, valuations, stock ideas and data checks.

## Data
- Yahoo Finance via yfinance: prices, the 30-year Treasury yield (^TYX), analyst EPS forecasts, financial statements
  (a dated backup of the EPS forecasts, saved in Colab, is used when Yahoo refuses live requests)
- Ken French Data Library: Fama-French factors
- WRDS monthly stock returns (AC420 Lecture 17 class file): used offline only to
  estimate reference coefficients; no raw WRDS data is stored in this repo

## Repo layout
- `streamlit_app.py`: homepage (Portfolio Builder)
- `pages/`: Risk Deep Dive, Projections & Valuation, Stock Ideas, Data Check
- `core/`: calculations (data, models, valuation, scoring, projections, suggestions)
- `data/`: small CSV files the app reads (see `data/README.md`)
- `notebooks/training_wrds.ipynb`: offline Colab notebook that builds the files in `data/`
- `notebooks/known_answer_tests.ipynb`: checks the code against answers from class
- `notebooks/eps_snapshot.ipynb`: Colab notebook that saves the analyst EPS backup

## Code credits
- Adapted from AC420 class notebooks by Prof. Peter Wysocki: Lecture 12 & 13 CAPM and
  FF3 regressions, Lecture 15 & 16 DCF valuation, Lecture 16 & 17 comparables,
  Lecture 17 alpha regressions, and the Lecture 14 Streamlit starter app (Assignment #4)
- Written with AI assistance (Claude by Anthropic); my AI usage log records how AI helped

For education only, not investment advice.
