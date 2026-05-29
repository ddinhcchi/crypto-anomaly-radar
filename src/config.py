import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _split_csv(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass
class Settings:
    symbols: list[str]
    interval: str = os.getenv("INTERVAL", "1m")
    zscore_window: int = int(os.getenv("ZSCORE_WINDOW", "60"))
    zscore_threshold: float = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
    if_contamination: float = float(os.getenv("IF_CONTAMINATION", "0.02"))
    atr_window: int = int(os.getenv("ATR_WINDOW", "14"))
    atr_multiplier: float = float(os.getenv("ATR_MULTIPLIER", "2.5"))
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    alert_cooldown_seconds: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))


settings = Settings(symbols=_split_csv(os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")))
