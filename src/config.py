"""
ArthaMart — Configuration loader.
Loads config.yaml and exposes it as a nested dict with dot-access.
"""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "arthamart.db"

_config: dict | None = None


def load_config() -> dict:
    global _config
    if _config is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def get(key_path: str, default=None):
    """Get a nested config value using dot notation, e.g. get('ocr.similarity_threshold')."""
    cfg = load_config()
    node = cfg
    for part in key_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
