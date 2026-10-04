# Volatility Forecasting and VaR Backtesting (S&P 500)

Forecasts volatility for 50 large S&P 500 stocks (2018-2024) using EWMA, GARCH(1,1) and a Temporal Fusion Transformer (TFT), then tests each model as a 1-day Value-at-Risk (VaR) model.

## Data
- 50 large-cap S&P 500 stocks plus the VIX index, from Yahoo Finance (yfinance), 2018-2024
- Features: log returns, 20-day realised volatility, lagged volatility, EWMA volatility, volume ratio, VIX, sector, calendar encodings
- Split: train 2018-2021, validation 2022, test 2023-2024

## Models
Random walk, historical average, EWMA, GARCH(1,1), and a TFT (60-day lookback, p10/p50/p90 quantile outputs, trained jointly across all 50 stocks).

## Results

### 1-day VaR backtest (2023-2024, Kupiec POF test at 5% significance)

| Model | 95% VaR breach rate | 95% stocks passing | 99% VaR breach rate | 99% stocks passing |
|---|---|---|---|---|
| EWMA | 5.06% | 49/50 | 2.17% | 21/50 |
| GARCH(1,1) | 3.48% | 31/50 | 1.29% | 45/50 |
| TFT (p50) | 5.13% | 49/50 | 2.24% | 19/50 |

Each day's forecast uses only information available before that day, and all models are scored on the same stock-days. VaR assumes normally distributed returns with zero mean.

- TFT and EWMA are well calibrated at 95% but breach about twice as often as expected at 99%, consistent with fat-tailed returns.
- GARCH(1,1) is the most reliable at 99% but too conservative at 95% (too few breaches also fails the Kupiec test).

### Interval calibration
- TFT p10-p90 coverage on the test set: 76.9% (target 80%)
- After split conformal calibration on 2022 validation residuals: 84.6%

## Known limitations and next steps
- The RMSE table in the baseline notebook is not like-for-like: the baselines are scored on same-day realised volatility from 2022, the TFT on 5-day-ahead volatility from 2023, and the GARCH baseline there reuses its forecast between refits. The VaR backtest above corrects all three.
- The TFT target is realised volatility shifted 5 days ahead. Because past target values enter the encoder, the model is effectively a short-horizon forecaster.
- Next steps: Student-t VaR for fatter tails, and the Christoffersen test for clustered breaches.

## Repository structure
- `notebooks/Volatility_prediction.ipynb`: data download, feature engineering, baseline models
- `notebooks/TFT_with_VaR.ipynb`: TFT training and evaluation, conformal calibration, and the 1-day VaR backtest (Kupiec test). Main results are here.
- `notebooks/volatility_prediction.py`, `notebooks/tft.py`: script versions of the pipeline that produce the files the app loads
- `backend/`: FastAPI service with endpoints for tickers, GARCH and TFT forecasts, and a Claude-powered chat about the forecasts
- `frontend/`: React (Vite) dashboard with a ticker selector, forecast chart and chat panel

## Running the app
The trained checkpoint and data files are not committed (too large). Run `volatility_prediction.py` then `tft.py` to generate them, copy them into `backend/data/`, and set `ANTHROPIC_API_KEY` in `backend/.env`. Then:

    cd backend && pip install -r requirements.txt && uvicorn main:app --reload
    cd frontend && npm install && npm run dev
pandas, yfinance, arch (GARCH), pytorch-forecasting (TFT), PyTorch Lightning, scipy


