# -*- coding: utf-8 -*-
"""Load steering-wheel insertion configuration."""

from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config(path=None):
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(config_path)
    return data


def cfg_get(config, *keys, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


CONFIG = load_config()


def relative_path(config, *keys, default):
    value = cfg_get(config, *keys, default=default)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path
