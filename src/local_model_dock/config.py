import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Config:
    budget_gb: float
    auto_download: bool
    families: tuple[str, ...]
    api_key: str = field(repr=False)
    telegram_token: str = field(repr=False)
    telegram_user_id: str
    telegram_chat_id: str

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_user_id and self.telegram_chat_id)


def load_config(path: Path = Path(".env"), environ: Mapping[str, str] | None = None) -> Config:
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError("Invalid .env line")
            values[key.strip()] = value.strip().strip("\"'")
    values.update(os.environ if environ is None else environ)
    budget = float(values.get("MODEL_BUDGET_GB", "20"))
    if not 0 < budget < float("inf"):
        raise ValueError("MODEL_BUDGET_GB must be positive and finite")
    auto = values.get("AUTO_DOWNLOAD", "true").lower()
    if auto not in ("true", "false"):
        raise ValueError("AUTO_DOWNLOAD must be true or false")
    families = tuple(name.strip().lower() for name in values.get("MODEL_FAMILIES", "deepseek,qwen").split(",") if name.strip())
    return Config(budget, auto == "true", families, values.get("API_KEY", ""),
                  values.get("TELEGRAM_BOT_TOKEN", ""), values.get("TELEGRAM_USER_ID", ""),
                  values.get("TELEGRAM_CHAT_ID", ""))
