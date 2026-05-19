import numpy as np
import pandas as pd

from src.detect import detect_anomalies


def _synthetic(n: int = 300, spike_at: int = 250, spike_pct: float = 8.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.002, n))
    close[spike_at] = close[spike_at - 1] * (1 + spike_pct / 100.0)
    times = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": times,
            "close_time": times,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(80, 120, n),
        }
    )
    df["pct_return"] = df["close"].pct_change() * 100.0
    return df


def test_zscore_flags_injected_spike():
    df = _synthetic(spike_at=250, spike_pct=8.0)
    out = detect_anomalies(df, zscore_window=60, zscore_threshold=3.0)
    assert out.loc[250, "zscore_anomaly"], "8% spike should be flagged by z-score"


def test_quiet_series_has_few_anomalies():
    df = _synthetic(n=300, spike_at=10, spike_pct=0.0)  # no real spike
    out = detect_anomalies(df, zscore_window=60, zscore_threshold=3.5)
    # quiet random walk shouldn't trigger many z-score hits
    assert out["zscore_anomaly"].sum() <= 5


def test_isolation_forest_is_well_calibrated():
    df = _synthetic(n=300, spike_at=200, spike_pct=10.0)
    out = detect_anomalies(df, if_contamination=0.02)
    # the injected 10% spike should be among the highest if_score points
    assert out.loc[200, "if_score"] >= out["if_score"].quantile(0.95)
