from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@lru_cache
def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("PV_CONFIG") or ROOT / "config" / "config.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
