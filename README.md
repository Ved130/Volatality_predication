# Volatility Forecasting with TFT and GARCH

machine learning project that forecasts stock market volatility using Temporal Fusion Transformer (TFT) and GARCH(1,1) models across 50 S&P 500 stocks.


## Results

| Model | RMSE | MAE | Coverage |
|---|---|---|---|
| Random Walk | 0.0240 | 0.0107 | — |
| Historical Average | 0.0795 | 0.0602 | — |
| EWMA | 0.0373 | 0.0269 | — |
| GARCH(1,1) | 0.0751 | 0.0558 | — |
| TFT (p50) | 0.0602 | 0.0394 | 76.9% |
| TFT + Conformal | 0.0602 | 0.0394 | 79.9% |

TFT outperforms GARCH(1,1) on both RMSE and MAE. Conformal prediction calibration corrects the uncertainty intervals from 76.9% to 79.9% coverage against a nominal 80% target.

## Project Structure
## What This Project Does

### Problem
Predicting how much a stock price will move over the next 5 trading days — used for options pricing, risk management, and portfolio construction.

### Approach
Five models compared in order of complexity:
1. Random Walk
2. Historical Average
3. EWMA (RiskMetrics)
4. GARCH(1,1)
5. Temporal Fusion Transformer

### Why TFT over GARCH
GARCH only uses a single stock return history. TFT additionally uses VIX, volume ratio, lag volatility, EWMA, cyclic time encodings, and static per-ticker features (sector, market cap). TFT learns these jointly across all 50 stocks.

### Uncertainty Quantification
TFT outputs p10/p50/p90 quantiles directly. Conformal prediction calibration on the validation set corrects interval width to achieve 79.9% coverage on the test set against a nominal 80% target.

## Data
- **Universe**: 50 S&P 500 stocks
- **Period**: 2018-2024
- **Source**: Yahoo Finance via yfinance
- **Target**: 20-day realized volatility shifted forward 5 days
- **Split**: Train 2018-2021 | Val 2022 | Test 2023-2024

## Tech Stack
- **Data**: yfinance, pandas, pyarrow
- **Models**: arch (GARCH), pytorch-forecasting (TFT), pytorch-lightning

## Author
Ved Pashine
