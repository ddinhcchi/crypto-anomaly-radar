# 📈 Crypto Anomaly Radar

Real-time anomaly detection on Binance spot candles. Two detectors run side by side — **rolling z-score** for fast spike detection, **IsolationForest** for joint return/volume regimes — and the chart highlights where they agree. Optional per-symbol Telegram alerts.

![demo](demo/demo.gif)

> Demo GIF placeholder: load BTC + ETH + SOL on 1-minute candles, walk the slider to a recent volatility burst, expand the "Recent anomalies" table.

---

## Why this project

A z-score alone is too jumpy on volatile markets — every news headline trips it. A pure ML anomaly detector hides its reasoning behind an opaque score. Showing **both, side by side** is the honest answer: a real signal is one where a fast, interpretable rule and an unsupervised learner agree. This is also how risk desks actually triage trade alerts.

---

## Features

- **Live Binance spot data** via the public `/api/v3/klines` endpoint (no API key needed)
- **Two detectors**:
  - Rolling z-score of % returns (window + threshold sliders)
  - IsolationForest on (return, log-volume) joint feature space
- **Three highlight tiers**: z-score only, IsolationForest only, both agree
- **Candle chart** (Plotly) with anomaly markers + a z-score sub-panel with threshold lines
- **24h universe overview** table
- **Optional Telegram alerts** with per-symbol cooldown
- **Auto-refresh** every 30 s for paper-trading-style monitoring

---

## Quick start

```bash
git clone <this-repo>
cd crypto-anomaly-radar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optional — Telegram alerts only
streamlit run app.py
```

Open <http://localhost:8501>. Default universe is BTC/ETH/SOL/BNB on 1-minute candles.

---

## How the detectors work

### Rolling z-score

Computed over `pct_return = close.pct_change()`:

```
z = (return - rolling_mean) / rolling_std
anomaly = |z| >= threshold
```

Window and threshold are sidebar sliders. The lower z-score sub-chart draws horizontal lines at ±threshold so the threshold's effect is visually obvious.

### IsolationForest

Trained on the same lookback window with two features:

- `pct_return` (signed, raw)
- `log1p(volume)` (handles long-tail volume distribution)

#### Why `log1p(volume)` and not raw volume

Spot volume is heavily right-skewed — a quiet minute might trade 5 BTC while a news minute trades 5,000. Feed raw volume to IsolationForest and *every* high-volume candle gets flagged as an outlier just because the distribution is fat-tailed, not because anything unusual is happening relative to recent activity. `log1p` compresses that tail so the forest is sensitive to *deviations from the typical regime* instead of absolute magnitude:

| What you'd see with raw volume | What you see with `log1p(volume)` |
|---|---|
| Every London-open / NY-open candle flagged | Only the *unusually* loud London-open candles flagged |
| Hard to tune `contamination` — fraction of "outliers" shifts with overall vol | `contamination` stays meaningful across market regimes |
| Score correlates almost perfectly with volume itself | Score correlates with the *joint* (return, volume) surprise |

`log1p` (i.e. `log(1 + x)`) instead of plain `log` because spot volume can legitimately be zero on some illiquid micro-cap candles, and `log(0)` is `-inf`.

#### Why two features and not just return

Most pure-price spikes already get caught by z-score. The wins from IsolationForest are anomalies where price barely moved but volume exploded (or vice versa) — e.g., a stealth accumulation candle. Those won't trip a z-score but will trip the forest. The chart marks these in purple so you can see what each detector contributes.

### Sample run on BTC/USDT, 1-minute, 500 candles

> Numbers from a fresh fetch at the time this README was written.

| Detector | Hits |
|---|---:|
| Z-score (\|z\| ≥ 3, window 60) | 4 |
| IsolationForest (contamination 2%) | 10 |
| Both agree | 2 |

The "both agree" set is what you'd actually wake someone up for.

---

## Code layout

| File | Responsibility |
|---|---|
| [`src/binance.py`](src/binance.py) | Public REST client: klines + 24h ticker |
| [`src/detect.py`](src/detect.py) | Z-score + IsolationForest + the agreement flag |
| [`src/alert.py`](src/alert.py) | Telegram client with per-symbol cooldown |
| [`src/config.py`](src/config.py) | `.env`-driven settings |
| [`app.py`](app.py) | Streamlit UI, charts, controls, auto-refresh |
| [`tests/test_detect.py`](tests/test_detect.py) | Synthetic spike tests for both detectors |

---

## Tests

```bash
pytest tests/ -v
# 3 passed: injected spike caught by z-score, quiet series stays quiet,
#           IsolationForest ranks the injected spike in top 5%
```

---

## Run with Docker

```bash
docker build -t crypto-anomaly-radar .
docker run --rm -p 8501:8501 --env-file .env crypto-anomaly-radar
```

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT` | Default universe (override in UI) |
| `INTERVAL` | `1m` | Candle granularity |
| `ZSCORE_WINDOW` | 60 | Rolling baseline length |
| `ZSCORE_THRESHOLD` | 3.0 | \|z\| cutoff |
| `IF_CONTAMINATION` | 0.02 | IsolationForest outlier fraction |
| `TELEGRAM_BOT_TOKEN` | — | Optional alerts |
| `TELEGRAM_CHAT_ID` | — | Optional alerts |
| `ALERT_COOLDOWN_SECONDS` | 300 | Per-symbol throttle |

---

## Roadmap

- Backtest mode: feed historical CSVs and grid-search threshold / contamination
- Multi-timeframe agreement (1m spike + 5m baseline)
- Per-symbol model retrained on its own volatility profile, not global contamination
- Webhook sink in addition to Telegram (Discord, Slack, custom)

---

## License

MIT
