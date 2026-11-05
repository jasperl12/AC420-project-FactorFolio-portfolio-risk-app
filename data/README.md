# data folder
Small CSV files the app reads. The first four are built by `notebooks/training_wrds.ipynb` (Step 1).

- sp100_tickers.csv: S&P 100 tickers with sector and industry from yfinance
- sp100_estimates.csv: CAPM and Fama-French estimates for the 66 S&P 100 stocks in the
  Lecture 17 class data (64 months, mid-2021 to 2026); reference values for the risk score
- market_scenarios.csv: bear, median and bull annual market returns, 1927 to 2025
  (25th, 50th and 75th percentiles; 10th and 90th also saved), plus average SMB and HML
- ff_factors_monthly.csv: monthly Fama-French factors (in percent, as posted) from
  Ken French's Data Library; the app divides by 100
- eps_snapshot.csv: a dated backup of Yahoo Finance analyst EPS forecasts (0y and +1y), Yahoo
  beta and forward P/E for the S&P 100, saved in Colab by `notebooks/eps_snapshot.ipynb`. The
  app uses it only when Yahoo refuses live requests from Streamlit Cloud, and shows its date

No raw WRDS data is stored in this repo.
