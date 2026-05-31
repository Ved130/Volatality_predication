import json
import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model

DATA_DIR = Path(__file__).parent / "data"


def load_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features.parquet")


def load_config() -> dict:
    with open(DATA_DIR / "model_config.json") as f:
        return json.load(f)


def get_ticker_list() -> list[dict]:
    df = load_features()
    meta = (
        df[["ticker", "company", "sector"]]
        .drop_duplicates()
        .sort_values("ticker")
    )
    return meta.to_dict(orient="records")


def get_ticker_data(ticker: str) -> pd.DataFrame:
    df = load_features()
    return df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)


def get_recent_context(ticker: str, days: int = 30) -> dict:
    """
    Returns recent stats for a ticker.
    Used to inject context into Claude chat prompt.
    """
    tkr = get_ticker_data(ticker)

    if len(tkr) == 0:
        raise ValueError(f"Ticker {ticker} not found")

    recent   = tkr.tail(days)
    last_row = tkr.iloc[-1]

    recent_vol  = float(last_row["real_vol"])
    avg_vol_30d = float(recent["real_vol"].mean())
    return_5d   = float(tkr["log_ret"].iloc[-5:].sum())

    return {
        "ticker":          ticker,
        "company":         str(last_row.get("company", ticker)),
        "sector":          str(last_row.get("sector", "Unknown")),
        "current_vol":     round(recent_vol, 4),
        "avg_vol_30d":     round(avg_vol_30d, 4),
        "return_5d":       round(return_5d, 4),
        "vol_regime":      "high" if recent_vol > avg_vol_30d * 1.3 else "normal",
    }


def run_garch_forecast(ticker: str, horizon: int = 5) -> dict:
    """
    Fits GARCH(1,1) on full history and forecasts `horizon` steps ahead.
    """
    tkr     = get_ticker_data(ticker)
    returns = tkr["log_ret"].dropna() * 100  # scale for numerical stability

    model  = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    result = model.fit(disp="off", show_warning=False)
    params = result.params

    fc           = result.forecast(horizon=horizon, reindex=False)
    forecast_var = fc.variance.values[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252) / 100

    upper = forecast_vol * 1.645
    lower = np.clip(forecast_vol / 1.645, 0, None)

    return {
        "ticker":          ticker,
        "model":           "GARCH(1,1)",
        "horizon":         horizon,
        "forecast_vol":    forecast_vol.tolist(),
        "upper_90":        upper.tolist(),
        "lower_90":        lower.tolist(),
        "alpha":           round(float(params["alpha[1]"]), 4),
        "beta":            round(float(params["beta[1]"]), 4),
        "alpha_plus_beta": round(float(params["alpha[1]"] + params["beta[1]"]), 4),
    }