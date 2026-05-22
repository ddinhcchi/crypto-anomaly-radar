"""Anomaly detection: rolling z-score + IsolationForest on (return, log-volume).

Z-score is fast, interpretable and great for single-feature spikes.
IsolationForest catches joint anomalies (e.g. small price move on huge volume).
Both are returned so the UI can highlight where they agree vs. diverge.
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


def detect_anomalies(
    df: pd.DataFrame,
    zscore_window: int = 60,
    zscore_threshold: float = 3.0,
    if_contamination: float = 0.02,
) -> pd.DataFrame:
    out = add_zscore(df, window=zscore_window)
    out = add_isolation_forest(out, contamination=if_contamination)
    out["zscore_anomaly"] = out["zscore"].abs() >= zscore_threshold
    out["both_anomaly"] = out["zscore_anomaly"] & out["if_anomaly"]
    return out
