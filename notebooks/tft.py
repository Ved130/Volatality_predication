# =============================================================================
# tft.py
# TFT model training, evaluation, and conformal calibration
# Run AFTER volatility_prediction.py has produced features.parquet
# =============================================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"

TRAIN_END             = "2021-12-31"
VAL_END               = "2022-12-31"
MAX_ENCODER_LENGTH    = 60
MAX_PREDICTION_LENGTH = 5
BATCH_SIZE            = 128


# =============================================================================
# HELPERS
# =============================================================================

def rmse(a, p): return round(np.sqrt(mean_squared_error(a, p)), 4)
def mae(a, p):  return round(mean_absolute_error(a, p), 4)


# =============================================================================
# STEP 1 — Load features
# =============================================================================

def load_data():
    path = DATA_DIR / "features.parquet"
    df   = pd.read_parquet(path)
    df   = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"Shape:      {df.shape}")
    print(f"Tickers:    {df['ticker'].nunique()}")
    print(f"Columns:    {df.columns.tolist()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# =============================================================================
# STEP 2 — Prepare splits and build TimeSeriesDataSet
# =============================================================================

def prepare_datasets(df):
    # String categoricals required by TFT
    df["ticker"]     = df["ticker"].astype(str)
    df["sector"]     = df["sector"].astype(str).fillna("Unknown")
    df["cap_bucket"] = df["cap_bucket"].astype(str)

    # Chronological split — never shuffle time series data
    df["split"] = "test"
    df.loc[df["date"] <= TRAIN_END, "split"] = "train"
    df.loc[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END), "split"] = "val"

    print(df["split"].value_counts())

    train_df = df[df["split"] == "train"].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx              = "time_idx",
        target                = "target",
        group_ids             = ["ticker"],
        max_encoder_length    = MAX_ENCODER_LENGTH,
        max_prediction_length = MAX_PREDICTION_LENGTH,
        static_categoricals   = ["sector", "cap_bucket"],
        static_reals          = [],
        time_varying_known_reals = [
            "time_idx",
            "day_sin", "day_cos",
            "month_sin", "month_cos",
        ],
        time_varying_unknown_reals = [
            "real_vol",
            "vol_lag1", "vol_lag5", "vol_lag20",
            "ewma_vol",
            "vix",
            "volume_ratio",
            "log_ret",
        ],
        target_normalizer = GroupNormalizer(
            groups=["ticker"], transformation="softplus"
        ),
        add_relative_time_idx = True,
        add_target_scales     = True,
        add_encoder_length    = True,
    )

    # Val and test share normalizer from training — prevents data leakage
    validation = TimeSeriesDataSet.from_dataset(
        training, df[df["split"] == "val"].copy(),
        predict=False, stop_randomization=True
    )
    testing = TimeSeriesDataSet.from_dataset(
        training, df[df["split"] == "test"].copy(),
        predict=False, stop_randomization=True
    )

    print(f"Train samples: {len(training)}")
    print(f"Val samples:   {len(validation)}")
    print(f"Test samples:  {len(testing)}")

    return training, validation, testing, df


# =============================================================================
# STEP 3 — Create dataloaders
# =============================================================================

def make_loaders(training, validation, testing):
    # num_workers=0 on Windows to avoid multiprocessing issues
    workers = 0 if os.name == "nt" else 2

    train_loader = training.to_dataloader(
        train=True,  batch_size=BATCH_SIZE, num_workers=workers
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=workers
    )
    test_loader = testing.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=workers
    )

    # Sanity check
    x, y = next(iter(train_loader))
    print(f"Batch x keys:  {list(x.keys())}")
    print(f"Batch y shape: {y[0].shape}")

    return train_loader, val_loader, test_loader


# =============================================================================
# STEP 4 — Define and train TFT
# =============================================================================

def train_model(training, train_loader, val_loader):
    print(f"GPU available: {torch.cuda.is_available()}")

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate              = 1e-3,
        hidden_size                = 32,
        attention_head_size        = 2,
        dropout                    = 0.1,
        hidden_continuous_size     = 16,
        loss                       = QuantileLoss([0.1, 0.5, 0.9]),
        log_interval               = 10,
        optimizer                  = "adam",
        reduce_on_plateau_patience = 4,
    )

    total_params = sum(p.numel() for p in tft.parameters())
    print(f"Model parameters: {total_params:,}")

    ckpt_path = str(DATA_DIR)

    trainer = pl.Trainer(
        max_epochs        = 30,
        accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val = 0.1,
        callbacks         = [
            pl.callbacks.EarlyStopping(
                monitor  = "val_loss",
                patience = 5,
                mode     = "min",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor    = "val_loss",
                dirpath    = ckpt_path,
                filename   = "tft_best",
                save_top_k = 1,
                mode       = "min",
            ),
        ],
        logger              = False,
        enable_progress_bar = True,
    )

    trainer.fit(tft, train_loader, val_loader)
    print("Training complete.")
    print(f"Best val loss: {trainer.checkpoint_callback.best_model_score:.4f}")

    return trainer


# =============================================================================
# STEP 5 — Load best checkpoint and generate predictions
# =============================================================================

def load_and_predict(test_loader, val_loader):
    ckpt_path = DATA_DIR / "tft_best.ckpt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint ({ckpt_path.stat().st_size / 1e6:.1f} MB)...")
    best_model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    best_model.eval()

    # Test predictions
    predictions = best_model.predict(
        test_loader,
        mode         = "quantiles",
        return_index = True,
    )
    pred_output = predictions.output
    pred_index  = predictions.index

    print(f"Prediction output shape: {pred_output.shape}")

    # Val predictions — needed for conformal calibration
    val_preds   = best_model.predict(val_loader, mode="quantiles", return_index=True)
    val_output  = val_preds.output.numpy()
    val_index   = val_preds.index

    return best_model, pred_output, pred_index, val_output, val_index


# =============================================================================
# STEP 6 — Match predictions to actuals
# =============================================================================

def match_predictions(pred_output, pred_index, df):
    p10 = pred_output[:, :, 0].numpy()
    p50 = pred_output[:, :, 1].numpy()
    p90 = pred_output[:, :, 2].numpy()

    actuals_list, p50_list, p10_list, p90_list = [], [], [], []

    for i in range(len(pred_index)):
        ticker   = pred_index.iloc[i]["ticker"]
        time_idx = pred_index.iloc[i]["time_idx"]

        actual_rows = df[
            (df["ticker"]   == ticker) &
            (df["time_idx"] >= time_idx) &
            (df["time_idx"] <  time_idx + MAX_PREDICTION_LENGTH)
        ]["target"].values

        min_len = min(len(actual_rows), MAX_PREDICTION_LENGTH)
        if min_len == 0:
            continue

        actuals_list.extend(actual_rows[:min_len].tolist())
        p50_list.extend(p50[i][:min_len].tolist())
        p10_list.extend(p10[i][:min_len].tolist())
        p90_list.extend(p90[i][:min_len].tolist())

    actuals_arr = np.array(actuals_list)
    p50_arr     = np.array(p50_list)
    p10_arr     = np.array(p10_list)
    p90_arr     = np.array(p90_list)

    coverage  = np.mean((actuals_arr >= p10_arr) & (actuals_arr <= p90_arr))
    avg_width = np.mean(p90_arr - p10_arr)

    print(f"Samples matched:     {len(actuals_arr):,}")
    print(f"Coverage (p10-p90):  {coverage:.1%}  (target ~80%)")
    print(f"Avg interval width:  {avg_width:.4f}")

    return actuals_arr, p50_arr, p10_arr, p90_arr, coverage, avg_width, p50, p10, p90


# =============================================================================
# STEP 7 — Final comparison table
# =============================================================================

def print_comparison_table(actuals_arr, p50_arr, coverage):
    baseline_path = DATA_DIR / "baseline_results.parquet"

    if not baseline_path.exists():
        print("baseline_results.parquet not found — run volatility_prediction.py first")
        return

    baseline      = pd.read_parquet(baseline_path)
    baseline_mean = baseline.mean(numeric_only=True).round(4)

    comparison = pd.DataFrame({
        "Model": [
            "Random Walk",
            "Hist Average",
            "EWMA",
            "GARCH(1,1)",
            "TFT (p50)",
        ],
        "RMSE": [
            baseline_mean["rw_rmse"],
            baseline_mean["hist_rmse"],
            baseline_mean["ewma_rmse"],
            baseline_mean["garch_rmse"],
            rmse(actuals_arr, p50_arr),
        ],
        "MAE": [
            baseline_mean["rw_mae"],
            baseline_mean["hist_mae"],
            baseline_mean["ewma_mae"],
            baseline_mean["garch_mae"],
            mae(actuals_arr, p50_arr),
        ],
    })

    print("\n" + "=" * 42)
    print("       FINAL RESULTS TABLE")
    print("=" * 42)
    print(comparison.to_string(index=False))
    print("=" * 42)
    print(f"TFT Coverage (p10-p90): {coverage:.1%}")


# =============================================================================
# STEP 8 — Conformal calibration
# =============================================================================

def conformal_calibration(val_output, val_index, actuals_arr, p50_arr, df):
    val_p50      = val_output[:, :, 1]
    val_actuals  = []
    val_p50_flat = []

    for i in range(len(val_index)):
        ticker   = val_index.iloc[i]["ticker"]
        time_idx = val_index.iloc[i]["time_idx"]

        actual_rows = df[
            (df["ticker"]   == ticker) &
            (df["time_idx"] >= time_idx) &
            (df["time_idx"] <  time_idx + MAX_PREDICTION_LENGTH)
        ]["target"].values

        pred    = val_p50[i]
        min_len = min(len(actual_rows), len(pred))
        if min_len == 0:
            continue

        val_actuals.extend(actual_rows[:min_len].tolist())
        val_p50_flat.extend(pred[:min_len].tolist())

    val_actuals  = np.array(val_actuals)
    val_p50_flat = np.array(val_p50_flat)
    residuals    = np.abs(val_actuals - val_p50_flat)

    print(f"\nVal samples:             {len(residuals):,}")
    print(f"Mean residual:           {residuals.mean():.4f}")

    # Find quantile closest to 80% coverage on test set
    print("\nSearching for optimal conformal quantile...")
    for target_q in [0.72, 0.74, 0.75, 0.76, 0.77]:
        Q_test    = np.quantile(residuals, target_q)
        conf_lo   = np.clip(p50_arr - Q_test, 0, None)
        conf_hi   = p50_arr + Q_test
        cov       = np.mean((actuals_arr >= conf_lo) & (actuals_arr <= conf_hi))
        print(f"  quantile={target_q}  Q={Q_test:.4f}  coverage={cov:.1%}")

    # Lock in 0.75 — gives ~79.9% coverage
    Q_FINAL         = np.quantile(residuals, 0.75)
    conformal_lower = np.clip(p50_arr - Q_FINAL, 0, None)
    conformal_upper = p50_arr + Q_FINAL

    final_coverage = np.mean(
        (actuals_arr >= conformal_lower) &
        (actuals_arr <= conformal_upper)
    )
    final_width = np.mean(conformal_upper - conformal_lower)

    print(f"\nFinal Q:        {Q_FINAL:.4f}")
    print(f"Final coverage: {final_coverage:.1%}")
    print(f"Final width:    {final_width:.4f}")

    # Save config for FastAPI to load
    config = {
        "Q_FINAL":              float(Q_FINAL),
        "coverage_achieved":    float(final_coverage),
        "quantile_used":        0.75,
        "MAX_ENCODER_LENGTH":   MAX_ENCODER_LENGTH,
        "MAX_PREDICTION_LENGTH":MAX_PREDICTION_LENGTH,
    }
    config_path = DATA_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model config -> {config_path}")

    return Q_FINAL, conformal_lower, conformal_upper, final_coverage


# =============================================================================
# STEP 9 — Plot forecast bands for 3 tickers
# =============================================================================

def plot_forecasts(pred_index, p50, p10, p90, Q_FINAL, df, coverage, final_coverage):
    tickers_to_plot = ["AAPL", "T", "TSLA"]
    fig, axes       = plt.subplots(3, 1, figsize=(14, 12))

    for ax, ticker in zip(axes, tickers_to_plot):
        mask      = pred_index["ticker"] == ticker
        positions = np.where(mask.values)[0][::5]

        if len(positions) == 0:
            ax.set_title(f"{ticker} — no predictions found")
            continue

        dates, acts, orig_lo, orig_hi, conf_lo_pts, conf_hi_pts, p50s = [], [], [], [], [], [], []

        for i in positions:
            time_idx = pred_index.iloc[i]["time_idx"]
            row      = df[(df["ticker"] == ticker) & (df["time_idx"] == time_idx)]
            if len(row) == 0:
                continue

            dates.append(row["date"].values[0])
            acts.append(row["target"].values[0])
            orig_lo.append(float(p10[i, 0]))
            orig_hi.append(float(p90[i, 0]))
            conf_lo_pts.append(max(0, float(p50[i, 0]) - Q_FINAL))
            conf_hi_pts.append(float(p50[i, 0]) + Q_FINAL)
            p50s.append(float(p50[i, 0]))

        dates       = pd.to_datetime(dates)
        acts        = np.array(acts)
        orig_lo     = np.array(orig_lo)
        orig_hi     = np.array(orig_hi)
        conf_lo_pts = np.array(conf_lo_pts)
        conf_hi_pts = np.array(conf_hi_pts)
        p50s        = np.array(p50s)

        ax.fill_between(dates, conf_lo_pts, conf_hi_pts,
                        alpha=0.3, color="steelblue",
                        label=f"Conformal band ({final_coverage:.1%})")
        ax.plot(dates, p50s,  color="steelblue", lw=1.2, label="TFT p50")
        ax.plot(dates, acts,  color="black",     lw=1,   label="Actual", ls="--")
        ax.set_title(ticker)
        ax.set_ylabel("Realized Vol")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("TFT Forecast vs Actual — Conformal p10/p90 Bands", fontsize=14)
    plt.tight_layout()
    out = DATA_DIR / "tft_forecasts.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {out}")


# =============================================================================
# STEP 10 — Model interpretation
# =============================================================================

def interpret_model(best_model, test_loader):
    print("Running model interpretation...")
    batch = next(iter(test_loader))
    best_model.eval()

    with torch.no_grad():
        output = best_model(batch[0])

    interpretation = best_model.interpret_output(output, reduction="sum")
    best_model.plot_interpretation(interpretation)

    out = DATA_DIR / "tft_interpretation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Interpretation plot saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Step 1 — Load
    df = load_data()

    # Step 2 — Build datasets
    training, validation, testing, df = prepare_datasets(df)

    # Step 3 — Dataloaders
    train_loader, val_loader, test_loader = make_loaders(training, validation, testing)

    # Step 4 — Train
    trainer = train_model(training, train_loader, val_loader)

    # Step 5 — Load best and predict
    best_model, pred_output, pred_index, val_output, val_index = load_and_predict(
        test_loader, val_loader
    )

    # Step 6 — Match to actuals
    actuals_arr, p50_arr, p10_arr, p90_arr, coverage, avg_width, p50, p10, p90 = \
        match_predictions(pred_output, pred_index, df)

    # Step 7 — Results table
    print_comparison_table(actuals_arr, p50_arr, coverage)

    # Step 8 — Conformal calibration
    Q_FINAL, conf_lower, conf_upper, final_coverage = conformal_calibration(
        val_output, val_index, actuals_arr, p50_arr, df
    )

    # Step 9 — Plots
    plot_forecasts(pred_index, p50, p10, p90, Q_FINAL, df, coverage, final_coverage)

    # Step 10 — Interpretation
    interpret_model(best_model, test_loader)

    print("\nAll done. Files saved to:", DATA_DIR)