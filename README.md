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
- **Three complementary detectors**:
  - Rolling z-score of % returns (window + threshold sliders)
  - IsolationForest on (return, log-volume) joint feature space
  - ATR (Average True Range) range expansion — catches wick spikes that close near the open
- **Agreement-weighted signal**: a candle highlighted in red when **≥2 of 3 detectors agree** — fewer false positives than any single signal
- **Per-detector chart markers** so you can see what each contributes (orange = z, purple × = iso, teal ▲ = atr-only, red ★ = ≥2 agree)
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

### ATR range expansion

`TR = max(high - low, |high - prev_close|, |low - prev_close|)`. ATR is just `TR` rolling-meaned over `ATR_WINDOW` candles (default 14, Wilder's choice). A candle is flagged when `TR ≥ multiplier × ATR` (default 2.5×).

Why add ATR when we already have z-score and IsoForest? Because **wick spikes**. A candle that opens at 100, prints a 108 high, and closes back at 100 has:

- pct_return = 0 → z-score stays quiet
- volume might be average → IsoForest stays quiet
- but TR = 8% → ATR detector lights up

These are real events in crypto — liquidations, flash dumps, fat-finger wicks. ATR is the only one of the three that catches them.

### Agreement is the gold signal

Each detector alone has a false-positive rate you can tune (threshold, contamination, multiplier). Agreement across detectors compounds: the joint false-positive rate roughly multiplies. The UI shows a red star when **≥ 2 of 3 detectors agree** — that's what you'd actually wake an on-call engineer for.

### Sample run on BTC/USDT, 1-minute, 500 candles

> Numbers from a fresh fetch at the time this README was written.

| Detector | Hits |
|---|---:|
| Z-score (\|z\| ≥ 3, window 60) | 4 |
| IsolationForest (contamination 2%) | 10 |
| ATR (multiplier 2.5×, window 14) | 6 |
| **≥ 2 detectors agree** | **2** |

The agreement set is the small, high-confidence subset of any single signal.

---

## Code layout

| File | Responsibility |
|---|---|
| [`src/binance.py`](src/binance.py) | Public REST client: klines + 24h ticker |
| [`src/detect.py`](src/detect.py) | Z-score + IsolationForest + ATR + agreement flag |
| [`src/alert.py`](src/alert.py) | Telegram client with per-symbol cooldown |
| [`src/config.py`](src/config.py) | `.env`-driven settings |
| [`app.py`](app.py) | Streamlit UI, charts, controls, auto-refresh |
| [`tests/test_detect.py`](tests/test_detect.py) | Synthetic spike tests for both detectors |

---

## Tests

```bash
pytest tests/ -v
# 5 passed: z-score catches injected spike, quiet series stays quiet,
#           IsolationForest ranks the spike top 5%, ATR catches wick
#           spikes that z-score misses, signal_count integrity check
```

---

## Security — secret scanning

The repo wires up [`gitleaks`](https://github.com/gitleaks/gitleaks) via [`pre-commit`](https://pre-commit.com) so a Telegram bot token (or any other credential) can never end up in a commit by accident. The hook runs locally on every `git commit` and blocks the commit if anything matches.

```bash
brew install gitleaks pre-commit   # macOS — Linux users: pipx install both
pre-commit install                  # installs the .git/hooks/pre-commit shim
pre-commit run --all-files          # scan everything already in the index
gitleaks detect --source . --verbose # scan the full git history
```

[`.gitleaks.toml`](.gitleaks.toml) extends the default ruleset with two allowlists:

- `.env.example` and `README.md` — they intentionally contain placeholder credentials
- `api.binance.com` — public REST endpoint that requires no authentication, so the hostname appearing in code is not a secret

This radar only uses Binance's *public* market-data endpoints — no signed requests, no API key needed. The only credential at risk is the optional Telegram bot token in `.env`, which is gitignored.

For deployments, also enable [GitHub Push Protection](https://docs.github.com/en/code-security/secret-scanning/push-protection-for-repositories-and-organizations) as a second line of defence.

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
| `ATR_WINDOW` | 14 | Rolling window for ATR |
| `ATR_MULTIPLIER` | 2.5 | TR ≥ multiplier × ATR → flagged |
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
