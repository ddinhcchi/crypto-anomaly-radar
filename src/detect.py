"""Anomaly detection: rolling z-score + IsolationForest + ATR range expansion.

Three complementary signals, deliberately overlapping:

- **z-score** — fast, interpretable, great for single-feature return spikes
- **IsolationForest** — catches joint (return, volume) regime shifts
- **ATR range** — flags candles whose high-low range exceeds k × ATR, i.e.
  wick spikes that close near the open and so leave z-score quiet

Each detector adds its own boolean column; the UI overlays them so you can
see what each contributes and where they agree.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def add_zscore(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Adds rolling mean/std and z-score over `pct_return`."""
    out = df.copy()
    rolling = out["pct_return"].rolling(window=window, min_periods=max(10, window // 4))
    out["return_mean"] = rolling.mean()
    out["return_std"] = rolling.std(ddof=0)
    safe_std = out["return_std"].replace(0, np.nan)
    out["zscore"] = (out["pct_return"] - out["return_mean"]) / safe_std
    return out


def add_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    # log1p compresses the long right tail of spot volume so the forest learns
    # "this candle is loud relative to recent activity" instead of "this candle
    # is loud in absolute terms" — see README for the regression-to-volume
    # failure mode that motivates this.
    features = pd.DataFrame(
        {
            "ret": out["pct_return"].fillna(0.0),
            "log_vol": np.log1p(out["volume"].fillna(0.0)),
        }
    )
    if len(features) < 20:
        out["if_score"] = 0.0
        out["if_anomaly"] = False
        return out

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    model.fit(features)
    out["if_score"] = -model.score_samples(features)  # higher = more anomalous
    out["if_anomaly"] = model.predict(features) == -1
    return out


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's True Range and its rolling mean (ATR).

    TR = max(high-low, |high - prev_close|, |low - prev_close|). Using prev
    close (not prev high/low) makes TR correctly account for overnight gaps
    — in crypto, a 1-minute gap during low-liquidity hours is the equivalent.
    """
    out = df.copy()
    high, low, close = out["high"], out["low"], out["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["tr"] = tr
    out["atr"] = tr.rolling(window=window, min_periods=max(3, window // 3)).mean()
    return out


def detect_anomalies(
    df: pd.DataFrame,
    zscore_window: int = 60,
    zscore_threshold: float = 3.0,
    if_contamination: float = 0.02,
    atr_window: int = 14,
    atr_multiplier: float = 2.5,
) -> pd.DataFrame:
    out = add_zscore(df, window=zscore_window)
    out = add_isolation_forest(out, contamination=if_contamination)
    out = add_atr(out, window=atr_window)

    out["zscore_anomaly"] = out["zscore"].abs() >= zscore_threshold
    # ATR anomaly: this candle's true range is `multiplier` × the recent ATR
    out["atr_anomaly"] = (out["tr"] >= atr_multiplier * out["atr"]).fillna(False)
    # "Both" used to mean (z & iso); now any two-of-three agreement is more
    # informative, so we flag where the count of active signals is >= 2.
    signals = out[["zscore_anomaly", "if_anomaly", "atr_anomaly"]].astype(int)
    out["signal_count"] = signals.sum(axis=1)
    out["both_anomaly"] = out["signal_count"] >= 2
    return out
