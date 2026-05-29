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


def test_atr_catches_wick_spike_that_zscore_misses():
    """A wick spike: huge intraperiod range but close == open, so pct_return
    is 0 and z-score stays calm. ATR should still flag it."""
    df = _synthetic(n=300, spike_at=10, spike_pct=0.0)  # no return spike anywhere
    # craft a wick on candle 200: open == close, but high spikes 8%
    df.loc[200, "high"] = df.loc[200, "open"] * 1.08
    df.loc[200, "low"] = df.loc[200, "open"] * 0.99
    out = detect_anomalies(df, atr_window=14, atr_multiplier=2.5)
    assert out.loc[200, "atr_anomaly"], "wick should be flagged by atr"
    assert not out.loc[200, "zscore_anomaly"], "zscore should stay calm on a wick"


def test_signal_count_and_both_flag():
    df = _synthetic(spike_at=250, spike_pct=8.0)
    out = detect_anomalies(df)
    # signal_count is 0..3
    assert out["signal_count"].between(0, 3).all()
    # both_anomaly == True iff at least two detectors agree
    assert (out["both_anomaly"] == (out["signal_count"] >= 2)).all()
