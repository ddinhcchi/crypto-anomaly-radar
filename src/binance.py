"""Binance public REST client — no API key required for klines / 24hr ticker."""
import httpx
import pandas as pd

BINANCE_HOST = "https://api.binance.com"


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    """Fetch up to 1000 most recent candles for a spot symbol."""
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(
            f"{BINANCE_HOST}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
        )
        resp.raise_for_status()
        raw = resp.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pct_return"] = df["close"].pct_change() * 100.0
    return df


def fetch_24h_change(symbols: list[str]) -> pd.DataFrame:
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{BINANCE_HOST}/api/v3/ticker/24hr")
        resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df = df[df["symbol"].isin(symbols)].copy()
    for c in ("priceChangePercent", "lastPrice", "volume", "quoteVolume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["symbol", "lastPrice", "priceChangePercent", "quoteVolume"]]
