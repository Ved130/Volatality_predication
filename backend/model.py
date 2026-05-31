# =============================================================================
# model.py
# TFT model loading and inference
# =============================================================================

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from data import load_features, load_config

DATA_DIR = Path(__file__).parent / "data"

MAX_ENCODER_LENGTH    = 60
MAX_PREDICTION_LENGTH = 5

# Cache model in memory — load once at startup
_model   = None
_dataset = None
_config  = None


def get_model_and_dataset():
    global _model, _dataset, _config

    if _model is not None:
        return _model, _dataset, _config

    print("Loading TFT model...")
    _config = load_config()

    df = load_features()
    df["ticker"]     = df["ticker"].astype(str)
    df["sector"]     = df["sector"].astype(str).fillna("Unknown")
    df["cap_bucket"] = df["cap_bucket"].astype(str)

    train_df = df[df["date"] <= "2021-12-31"].copy()

    _dataset = TimeSeriesDataSet(
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

    _model = TemporalFusionTransformer.load_from_checkpoint(
        str(DATA_DIR / "tft_best.ckpt")
    )
    _model.eval()
    print("TFT model loaded.")

    return _model, _dataset, _config


def run_tft_forecast(ticker: str, horizon: int = 5) -> dict:
    """
    Runs TFT inference for a single ticker.
    Returns p10, p50, p90 with conformal correction applied.
    """
    model, dataset, config = get_model_and_dataset()
    Q_FINAL = config["Q_FINAL"]

    df = load_features()
    df["ticker"]     = df["ticker"].astype(str)
    df["sector"]     = df["sector"].astype(str).fillna("Unknown")
    df["cap_bucket"] = df["cap_bucket"].astype(str)

    tkr = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    if len(tkr) == 0:
        raise ValueError(f"Ticker {ticker} not found")

    # Use last MAX_ENCODER_LENGTH rows as encoder input
    encoder_data = tkr.tail(MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH).copy()

    predict_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        encoder_data,
        predict=True,
        stop_randomization=True,
    )

    loader = predict_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    with torch.no_grad():
        preds = model.predict(loader, mode="quantiles", return_index=True)

    output = preds.output.numpy()  # (1, horizon, 3)
    p10    = output[0, :horizon, 0].tolist()
    p50    = output[0, :horizon, 1].tolist()
    p90    = output[0, :horizon, 2].tolist()

    # Apply conformal correction
    p50_arr        = np.array(p50)
    conf_lower     = np.clip(p50_arr - Q_FINAL, 0, None).tolist()
    conf_upper     = (p50_arr + Q_FINAL).tolist()

    return {
        "ticker":          ticker,
        "model":           "TFT",
        "horizon":         horizon,
        "p50":             p50,
        "p10_native":      p10,
        "p90_native":      p90,
        "conf_lower":      conf_lower,
        "conf_upper":      conf_upper,
        "Q_FINAL":         Q_FINAL,
    }