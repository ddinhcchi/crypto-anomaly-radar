import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.alert import TelegramAlerter
from src.binance import fetch_24h_change, fetch_klines
from src.config import settings
from src.detect import detect_anomalies

st.set_page_config(page_title="Crypto Anomaly Radar", page_icon="📈", layout="wide")
st.title("📈 Crypto Anomaly Radar")
st.caption(
    "Rolling z-score + IsolationForest on Binance spot candles. "
    "Highlights statistically unusual returns and volume regimes; "
    "optional Telegram alerts with per-symbol cooldown."
)

with st.sidebar:
    st.subheader("Universe")
    symbols_input = st.text_area(
        "Symbols (one per line, Binance spot)",
        value="\n".join(settings.symbols),
        height=120,
    )
    symbols = [s.strip().upper() for s in symbols_input.splitlines() if s.strip()]
    interval = st.selectbox(
        "Candle interval",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        index=["1m", "5m", "15m", "1h", "4h", "1d"].index(settings.interval),
    )
    lookback = st.slider("Lookback candles", 100, 1000, 500, 50)

    st.subheader("Detection")
    zwin = st.slider("Z-score rolling window", 10, 200, settings.zscore_window, 5)
    zth = st.slider("Z-score threshold", 1.5, 6.0, settings.zscore_threshold, 0.1)
    contam = st.slider("IsolationForest contamination", 0.005, 0.10, settings.if_contamination, 0.005)
    atr_win = st.slider("ATR window", 5, 60, settings.atr_window, 1)
    atr_mult = st.slider("ATR multiplier", 1.5, 5.0, settings.atr_multiplier, 0.1)

    st.subheader("Alerts")
    enable_alert = st.toggle(
        "Send Telegram alert on most recent anomaly",
        value=False,
        help="Per-symbol cooldown prevents repeated pings for the same spike.",
    )
    st.caption(
        "Configured ✅" if settings.telegram_bot_token and settings.telegram_chat_id
        else "Set TELEGRAM_BOT_TOKEN + CHAT_ID in .env to enable"
    )

    st.subheader("Refresh")
    auto_refresh = st.toggle("Auto-refresh every 30s", value=False)


@st.cache_data(ttl=20)
def cached_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    return fetch_klines(symbol=symbol, interval=interval, limit=limit)


@st.cache_data(ttl=20)
def cached_24h(symbols_tuple: tuple[str, ...]) -> pd.DataFrame:
    return fetch_24h_change(list(symbols_tuple))


alerter = TelegramAlerter(
    bot_token=settings.telegram_bot_token,
    chat_id=settings.telegram_chat_id,
    cooldown_seconds=settings.alert_cooldown_seconds,
)

# --- Universe overview ----
if not symbols:
    st.info("Add at least one Binance spot symbol in the sidebar (e.g. `BTCUSDT`).")
    st.stop()

try:
    overview = cached_24h(tuple(symbols))
    st.subheader("24h overview")
    if overview.empty:
        st.warning(
            "Binance returned no rows for the requested symbols. "
            "Check spelling (`BTCUSDT` not `BTC/USDT`) or whether they were delisted."
        )
    else:
        st.dataframe(
            overview.rename(
                columns={
                    "lastPrice": "Last",
                    "priceChangePercent": "24h %",
                    "quoteVolume": "Quote Volume",
                }
            ).set_index("symbol"),
            use_container_width=True,
        )
except Exception as exc:
    st.warning(f"24h ticker fetch failed: {exc}")

# --- Per-symbol charts ----
for symbol in symbols:
    st.markdown(f"### {symbol}")
    try:
        raw = cached_klines(symbol, interval, lookback)
    except Exception as exc:
        st.error(f"{symbol}: {exc}")
        continue

    df = detect_anomalies(
        raw,
        zscore_window=zwin,
        zscore_threshold=zth,
        if_contamination=contam,
        atr_window=atr_win,
        atr_multiplier=atr_mult,
    )
    n_z = int(df["zscore_anomaly"].sum())
    n_if = int(df["if_anomaly"].sum())
    n_atr = int(df["atr_anomaly"].sum())
    n_both = int(df["both_anomaly"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Candles", len(df))
    c2.metric("Z-score hits", n_z)
    c3.metric("IsoForest hits", n_if)
    c4.metric("ATR hits", n_atr)
    c5.metric("≥2 signals agree", n_both)

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            showlegend=False,
        )
    )
    z_hits = df[df["zscore_anomaly"]]
    if not z_hits.empty:
        price_fig.add_trace(
            go.Scatter(
                x=z_hits["open_time"],
                y=z_hits["close"],
                mode="markers",
                marker=dict(color="orange", size=10, symbol="diamond"),
                name="z-score",
            )
        )
    if_hits = df[df["if_anomaly"] & ~df["zscore_anomaly"]]
    if not if_hits.empty:
        price_fig.add_trace(
            go.Scatter(
                x=if_hits["open_time"],
                y=if_hits["close"],
                mode="markers",
                marker=dict(color="purple", size=9, symbol="x"),
                name="iso-forest only",
            )
        )
    atr_only_hits = df[df["atr_anomaly"] & ~df["zscore_anomaly"] & ~df["if_anomaly"]]
    if not atr_only_hits.empty:
        price_fig.add_trace(
            go.Scatter(
                x=atr_only_hits["open_time"],
                y=atr_only_hits["high"],
                mode="markers",
                marker=dict(color="teal", size=10, symbol="triangle-up"),
                name="atr range only",
            )
        )
    both_hits = df[df["both_anomaly"]]
    if not both_hits.empty:
        price_fig.add_trace(
            go.Scatter(
                x=both_hits["open_time"],
                y=both_hits["close"],
                mode="markers",
                marker=dict(color="red", size=12, symbol="star"),
                name="both",
            )
        )
    price_fig.update_layout(
        height=380,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(price_fig, use_container_width=True)

    z_fig = go.Figure()
    z_fig.add_trace(
        go.Scatter(x=df["open_time"], y=df["zscore"], name="z-score", line=dict(width=1))
    )
    z_fig.add_hline(y=zth, line_dash="dash", line_color="red", annotation_text=f"+{zth}")
    z_fig.add_hline(y=-zth, line_dash="dash", line_color="red", annotation_text=f"-{zth}")
    z_fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(z_fig, use_container_width=True)

    if enable_alert and n_both > 0 and alerter.is_configured:
        latest = df[df["both_anomaly"]].iloc[-1]
        if alerter.send(symbol, latest):
            st.success(f"Telegram alert sent for {symbol} (cooldown {settings.alert_cooldown_seconds}s).")

    with st.expander(f"Recent anomalies — {symbol}"):
        any_hit = df["if_anomaly"] | df["zscore_anomaly"] | df["atr_anomaly"]
        recent = df[any_hit].tail(15)[
            [
                "open_time", "close", "pct_return", "zscore",
                "volume", "if_score", "tr", "atr",
                "signal_count", "both_anomaly",
            ]
        ]
        st.dataframe(recent.set_index("open_time"), use_container_width=True)

if auto_refresh:
    import time as _time

    _time.sleep(30)
    st.rerun()
