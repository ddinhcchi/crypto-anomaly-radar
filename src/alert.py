import time

import pandas as pd
import requests

TELEGRAM_API = "https://api.telegram.org"


class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str, cooldown_seconds: int = 300):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self._last_sent_at: dict[str, float] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token) and bool(self.chat_id)

    def _can_send(self, symbol: str) -> bool:
        if not self.is_configured:
            return False
        now = time.time()
        last = self._last_sent_at.get(symbol, 0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_sent_at[symbol] = now
        return True

    def send(self, symbol: str, row: pd.Series) -> bool:
        if not self._can_send(symbol):
            return False
        when = pd.to_datetime(row["close_time"]).strftime("%Y-%m-%d %H:%M UTC")
        text = (
            f"🚨 *Anomaly detected* — `{symbol}`\n"
            f"Return: *{row['pct_return']:+.2f}%*  |  Z-score: *{row['zscore']:+.2f}*\n"
            f"Price: `{row['close']}`  Volume: `{row['volume']:.2f}`\n"
            f"Time: {when}"
        )
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
            return resp.ok
        except requests.RequestException:
            return False
