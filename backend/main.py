# =============================================================================
# main.py
# FastAPI application — volatility forecasting API
# =============================================================================

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

from data  import get_ticker_list, get_recent_context, run_garch_forecast
from model import run_tft_forecast

load_dotenv()

app = FastAPI(title="Volatility Forecasting API", version="1.0.0")

# CORS — allows React frontend on localhost:5173 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# =============================================================================
# Request / Response models
# =============================================================================

class ForecastRequest(BaseModel):
    ticker:  str
    horizon: int = 5


class ChatRequest(BaseModel):
    ticker:  str
    message: str
    history: list[dict] = []


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tickers")
def list_tickers():
    """Returns all available tickers with company name and sector."""
    try:
        return get_ticker_list()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/garch")
def forecast_garch(req: ForecastRequest):
    """
    Runs GARCH(1,1) forecast for a ticker.
    Returns point forecast + 90% uncertainty interval.
    """
    try:
        result  = run_garch_forecast(req.ticker, req.horizon)
        context = get_recent_context(req.ticker)
        return {**result, **context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/tft")
def forecast_tft(req: ForecastRequest):
    """
    Runs TFT forecast for a ticker.
    Returns p10/p50/p90 with conformal calibration applied.
    """
    try:
        result  = run_tft_forecast(req.ticker, req.horizon)
        context = get_recent_context(req.ticker)
        return {**result, **context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    """
    AI chat assistant powered by Claude.
    Injects live stock context into every message.
    """
    try:
        context = get_recent_context(req.ticker)

        system_prompt = f"""You are a financial analyst assistant specialising in volatility analysis.

Current data for {context['company']} ({context['ticker']}):
- Sector: {context['sector']}
- Current realized volatility: {context['current_vol']:.1%}
- 30-day average volatility: {context['avg_vol_30d']:.1%}
- 5-day return: {context['return_5d']:.2%}
- Volatility regime: {context['vol_regime']}

Answer questions about this stock's volatility clearly and concisely.
When you don't have enough data to answer confidently, say so."""

        messages = req.history + [{"role": "user", "content": req.message}]

        response = claude_client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 500,
            system     = system_prompt,
            messages   = messages,
        )

        return {"response": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))