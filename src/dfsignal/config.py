"""Application configuration loaded from environment and YAML files."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe local defaults."""

    app_env: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="DFSIGNAL_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one project YAML configuration file."""

    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return payload
