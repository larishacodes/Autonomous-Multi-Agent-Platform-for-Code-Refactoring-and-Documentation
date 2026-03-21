"""
core/config_validator.py — Config Schema Validator

Called by pipeline.py (and optionally main.py) at startup.
Raises ConfigValidationError with a precise message so bad config.json
entries fail loudly rather than silently using defaults.

Usage
-----
    from core.config_validator import validate_config
    validate_config(config_dict_or_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when config.json is missing required fields or has wrong types."""
    pass


# Required fields with their expected types and constraints
_SCHEMA: dict[str, dict[str, Any]] = {
    "models.refactor_adapter_path": {"type": str,  "required": True},
    "models.doc_adapter_path":      {"type": str,  "required": True},
    "models.max_input_length":      {"type": int,  "required": False, "min": 1},
    "models.max_output_length":     {"type": int,  "required": False, "min": 1},
    "pipeline.output_dir":          {"type": str,  "required": True},
}


def _get_nested(d: dict, dotted_key: str) -> tuple[bool, Any]:
    """Traverse a nested dict by dotted key. Returns (found, value)."""
    parts = dotted_key.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def validate_config(config: dict | Path | str) -> dict:
    """
    Validate a config dict or path to config.json.

    Parameters
    ----------
    config  Either a dict (already loaded) or a Path/str to a JSON file.

    Returns
    -------
    The validated config dict (loaded from file if a path was given).

    Raises
    ------
    ConfigValidationError   For missing fields, wrong types, out-of-range values.
    FileNotFoundError        If a path is given but the file does not exist.
    json.JSONDecodeError     If the file contains invalid JSON.
    """
    if config is None:
        raise ConfigValidationError("Config must not be None.")

    # ── Load from file if a path was given ───────────────────────────────
    if isinstance(config, (str, Path)):
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Config file contains invalid JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigValidationError("Config must be a dict or a path to a JSON file.")

    errors: list[str] = []

    # ── Check top-level sections ──────────────────────────────────────────
    if "models" not in config:
        errors.append("Missing required section: 'models'")
    if "pipeline" not in config:
        errors.append("Missing required section: 'pipeline'")

    if errors:
        raise ConfigValidationError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )

    # ── Validate each field ───────────────────────────────────────────────
    for dotted_key, rules in _SCHEMA.items():
        found, value = _get_nested(config, dotted_key)

        if not found:
            if rules["required"]:
                errors.append(f"Missing required field: '{dotted_key}'")
            continue

        expected_type = rules["type"]
        if not isinstance(value, expected_type):
            errors.append(
                f"Field '{dotted_key}' must be {expected_type.__name__}, "
                f"got {type(value).__name__} ({value!r})"
            )
            continue

        if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
            errors.append(
                f"Field '{dotted_key}' must be >= {rules['min']}, got {value}"
            )

    if errors:
        raise ConfigValidationError(
            "Config validation failed:\n  " + "\n  ".join(errors)
        )

    logger.debug("Config validation passed.")
    return config