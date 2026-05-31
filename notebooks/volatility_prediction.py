# =============================================================================
# volatility_prediction.py
# Data pipeline, feature engineering, and baseline models
# Run this first — produces features.parquet and baseline_results.parquet
# =============================================================================

import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# CONFIG — change DATA_DIR to wherever you want to store the output files
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B",
    "LLY",  "AVGO", "JPM",  "TSLA", "UNH",  "V",     "XOM",  "MA",
    "JNJ",  "PG",   "HD",   "MRK",  "COST", "ABBV",  "CVX",  "CRM",
    "BAC",  "NFLX", "AMD",  "PEP",  "KO",   "TMO",   "ADBE", "WMT",
    "MCD",  "CSCO", "ACN",  "ABT",  "LIN",  "DHR",   "TXN",  "QCOM",
    "NKE",  "PM",   "INTU", "CAT",  "MS",   "GS",    "AMGN", "SPGI",
    "BLK",  "RTX",  "T"
]

HORIZON    = 5          # days ahead to predict
SPLIT_DATE = "2022-01-01"


# =============================================================================
# STEP 1 — Download OHLCV data
# =============================================================================

def download_data():
    print(f"Downloading {len(TICKERS)} tickers (2018-2024)...")
    raw = yf.download(
        TICKERS,
        start="2018-01-01",
        end="2024-12-31",
        auto_adjust=True,
        progress=True,
    )

    print("Downloading VIX...")
    vix_raw = yf.download(
        "^VIX",
        start="2018-01-01",
        end="2024-12-31",
        auto_adjust=True,
        progress=False,
    )

    print(f"Raw shape: {raw.shape}")
    return raw, vix_raw


# =============================================================================
# STEP 2 — Reshape to long format
# =============================================================================

def reshape_to_long(raw):
    frames = []
    for ticker in TICKERS:
        try:
            df_t = raw.xs(ticker, axis=1, level=1).copy()
            df_t["ticker"] = ticker
            df_t.index.name = "date"
            frames.append(df_t)
        except KeyError:
            print(f"  Skipping {ticker} — no data")

    df = pd.concat(frames).reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df.dropna(subset=["close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Shape: {df.shape}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# =============================================================================
# STEP 3 — Add company names and sectors from Wikipedia
# =============================================================================

def add_metadata(df):
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)
        table = pd.read_html(response)[0]
        meta = table[["Symbol", "Security", "GICS Sector"]].copy()
        meta.columns = ["ticker", "company", "sector"]
        meta["ticker"] = meta["ticker"].str.replace(".", "-", regex=False)
        df = df.merge(meta, on="ticker", how="left")
        print(f"Missing company names: {df['company'].isna().sum()}")
    except Exception as e:
        print(f"Wikipedia scrape failed ({e}) — adding placeholder metadata")
        df["company"] = df["ticker"]
        df["sector"]  = "Unknown"
    return df


# =============================================================================
# STEP 4 — Compute features
# =============================================================================

def compute_features(df, vix_raw):
    # Clean VIX
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw[["Close"]].rename(columns={"Close": "vix"}).reset_index()
    vix.columns = [c.lower() for c in vix.columns]
    vix["date"] = pd.to_datetime(vix["date"])

    # Log returns
    df["log_ret"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Realized volatility — 20-day rolling, annualized
    df["real_vol"] = df.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )

    # Target — future realized vol shifted forward by HORIZON
    df["target"] = df.groupby("ticker")["real_vol"].transform(
        lambda x: x.shift(-HORIZON)
    )

    # Lag features
    for lag in [1, 5, 20]:
        df[f"vol_lag{lag}"] = df.groupby("ticker")["real_vol"].transform(
            lambda x, l=lag: x.shift(l)
        )

    # EWMA volatility
    df["ewma_vol"] = df.groupby("ticker")["log_ret"].transform(
        lambda x: x.pow(2).ewm(span=20, adjust=False).mean().apply(np.sqrt) * np.sqrt(252)
    )

    # Cyclic time encodings
    df["day_sin"]   = np.sin(2 * np.pi * df["date"].dt.dayofweek / 5)
    df["day_cos"]   = np.cos(2 * np.pi * df["date"].dt.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Volume ratio
    df["volume_ratio"] = df.groupby("ticker")["volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )

    # Merge VIX
    df = df.merge(vix, on="date", how="left")
    df["vix"] = df["vix"].ffill()

    # Cap bucket — static covariate for TFT
    avg_price  = df.groupby("ticker")["close"].mean()
    bucket_map = pd.cut(avg_price, bins=3, labels=["small", "mid", "large"]).to_dict()
    df["cap_bucket"] = df["ticker"].map(bucket_map)

    # time_idx — required by pytorch-forecasting
    df["time_idx"] = df.groupby("ticker").cumcount()

    print(f"Features computed. Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    return df


# =============================================================================
# STEP 5 — Clean and save features.parquet
# =============================================================================

def clean_and_save(df):
    df_clean = df.dropna(subset=[
        "target", "vol_lag1", "vol_lag5", "vol_lag20", "vix", "ewma_vol"
    ]).copy().reset_index(drop=True)

    print(f"Rows before clean: {len(df):,}")
    print(f"Rows after clean:  {len(df_clean):,}")
    print(f"Tickers retained:  {df_clean['ticker'].nunique()}")

    out_path = DATA_DIR / "features.parquet"
    df_clean.to_parquet(out_path, index=False, compression="snappy")
    print(f"Saved -> {out_path}")
    return df_clean


# =============================================================================
# STEP 6 — Baseline evaluation across all tickers
# =============================================================================

def rmse(actual, pred):
    return round(np.sqrt(mean_squared_error(actual, pred)), 4)

def mae(actual, pred):
    return round(mean_absolute_error(actual, pred), 4)


def run_baselines(df_clean):
    all_results = []

    for ticker in df_clean["ticker"].unique():
        tkr  = df_clean[df_clean["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        test = tkr[tkr["date"] >= SPLIT_DATE].copy()

        if len(test) < 50:
            continue

        test["rw_pred"]       = test["real_vol"].shift(1)
        rolling_mean          = tkr.set_index("date")["real_vol"].rolling(60).mean()
        test["hist_avg_pred"] = rolling_mean.reindex(test["date"].values).values
        test["ewma_pred"]     = test["ewma_vol"]

        test = test.dropna(subset=["rw_pred", "hist_avg_pred", "ewma_pred", "real_vol"])
        if len(test) < 10:
            continue

        actual = test["real_vol"]
        all_results.append({
            "ticker":    ticker,
            "rw_rmse":   rmse(actual, test["rw_pred"]),
            "rw_mae":    mae(actual,  test["rw_pred"]),
            "hist_rmse": rmse(actual, test["hist_avg_pred"]),
            "hist_mae":  mae(actual,  test["hist_avg_pred"]),
            "ewma_rmse": rmse(actual, test["ewma_pred"]),
            "ewma_mae":  mae(actual,  test["ewma_pred"]),
        })

    baseline_results = pd.DataFrame(all_results)
    print(f"Tickers evaluated: {len(baseline_results)}")
    return baseline_results


# =============================================================================
# STEP 7 — GARCH rolling evaluation across all tickers
# =============================================================================

def run_garch(df_clean):
    garch_results = []
    tickers       = df_clean["ticker"].unique()

    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] {ticker}", end=" ... ")

        tkr  = df_clean[df_clean["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        test = tkr[tkr["date"] >= SPLIT_DATE].copy()

        if len(test) < 50:
            print("skipped (too few rows)")
            continue

        returns_all  = tkr["log_ret"] * 100
        test_indices = test.index.tolist()
        garch_preds  = []
        fitted_res   = None

        for count, idx in enumerate(test_indices):
            history = returns_all.iloc[:tkr.index.get_loc(idx)]
            if len(history) < 100:
                garch_preds.append(np.nan)
                continue
            if count % 20 == 0 or fitted_res is None:
                try:
                    m          = arch_model(history, vol="Garch", p=1, q=1, dist="normal")
                    fitted_res = m.fit(disp="off", show_warning=False)
                except Exception:
                    garch_preds.append(np.nan)
                    continue
            try:
                fc  = fitted_res.forecast(horizon=1, reindex=False)
                vol = np.sqrt(fc.variance.values[-1, 0]) * np.sqrt(252) / 100
                garch_preds.append(vol)
            except Exception:
                garch_preds.append(np.nan)

        test             = test.copy()
        test["garch_pred"] = garch_preds
        test             = test.dropna(subset=["garch_pred", "real_vol"])

        if len(test) < 10:
            print("skipped (too few valid preds)")
            continue

        garch_results.append({
            "ticker":     ticker,
            "garch_rmse": rmse(test["real_vol"], test["garch_pred"]),
            "garch_mae":  mae(test["real_vol"],  test["garch_pred"]),
        })
        print(f"RMSE={garch_results[-1]['garch_rmse']}")

    return pd.DataFrame(garch_results)


# =============================================================================
# STEP 8 — Merge and print summary table
# =============================================================================

def build_summary(baseline_results, garch_df, df_clean):
    summary  = baseline_results.merge(garch_df, on="ticker", how="inner")
    mean_row = summary.drop(columns="ticker").mean().round(4)

    print("\n=== MEAN ACROSS ALL TICKERS ===")
    print(f"{'Model':<18} {'RMSE':>8} {'MAE':>8}")
    print("-" * 36)
    print(f"{'Random Walk':<18} {mean_row['rw_rmse']:>8} {mean_row['rw_mae']:>8}")
    print(f"{'Hist Average':<18} {mean_row['hist_rmse']:>8} {mean_row['hist_mae']:>8}")
    print(f"{'EWMA':<18} {mean_row['ewma_rmse']:>8} {mean_row['ewma_mae']:>8}")
    print(f"{'GARCH(1,1)':<18} {mean_row['garch_rmse']:>8} {mean_row['garch_mae']:>8}")

    out_path = DATA_DIR / "baseline_results.parquet"
    summary.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    return summary


# =============================================================================
# STEP 9 — Plot best and worst GARCH performers
# =============================================================================

def plot_garch_results(summary, df_clean):
    best3  = summary.nsmallest(3, "garch_rmse")["ticker"].tolist()
    worst3 = summary.nlargest(3,  "garch_rmse")["ticker"].tolist()
    show   = best3 + worst3

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes      = axes.flatten()

    for ax, ticker in zip(axes, show):
        tkr  = df_clean[df_clean["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        test = tkr[tkr["date"] >= SPLIT_DATE].copy()

        returns_all = tkr["log_ret"] * 100
        preds, fitted_res = [], None

        for count, idx in enumerate(test.index.tolist()):
            history = returns_all.iloc[:tkr.index.get_loc(idx)]
            if len(history) < 100:
                preds.append(np.nan)
                continue
            if count % 20 == 0 or fitted_res is None:
                try:
                    m          = arch_model(history, vol="Garch", p=1, q=1, dist="normal")
                    fitted_res = m.fit(disp="off", show_warning=False)
                except Exception:
                    preds.append(np.nan)
                    continue
            try:
                fc  = fitted_res.forecast(horizon=1, reindex=False)
                vol = np.sqrt(fc.variance.values[-1, 0]) * np.sqrt(252) / 100
                preds.append(vol)
            except Exception:
                preds.append(np.nan)

        test               = test.copy()
        test["garch_pred"] = preds
        test               = test.dropna(subset=["garch_pred"])
        label              = "BEST" if ticker in best3 else "WORST"

        ax.plot(test["date"], test["real_vol"],   label="Actual",     color="black",     lw=1)
        ax.plot(test["date"], test["garch_pred"], label="GARCH(1,1)", color="steelblue", lw=1, ls="--")
        ax.set_title(f"{ticker} ({label})")
        ax.set_ylabel("Ann. Vol")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("GARCH(1,1) — Best and Worst Performing Tickers", fontsize=14)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "garch_best_worst.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {DATA_DIR / 'garch_best_worst.png'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Step 1 — Download
    raw, vix_raw = download_data()

    # Step 2 — Reshape
    df = reshape_to_long(raw)

    # Step 3 — Metadata
    df = add_metadata(df)

    # Step 4 — Features
    df = compute_features(df, vix_raw)

    # Step 5 — Clean and save
    df_clean = clean_and_save(df)

    # Step 6 — Baselines
    baseline_results = run_baselines(df_clean)

    # Step 7 — GARCH (takes 10-15 min)
    garch_df = run_garch(df_clean)

    # Step 8 — Summary table
    summary = build_summary(baseline_results, garch_df, df_clean)

    # Step 9 — Plot
    plot_garch_results(summary, df_clean)

    print("\nAll done. Files saved to:", DATA_DIR)
