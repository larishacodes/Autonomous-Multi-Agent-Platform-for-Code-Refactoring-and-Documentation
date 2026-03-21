"""
tests/test_config.py — Config schema validation

These tests verify that bad config.json entries fail loudly at startup
rather than silently using defaults that cause confusing runtime errors.

Research: fail-fast validation at startup is the standard pattern —
validate all configuration before any model is loaded, not mid-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configvalidator import validate_config, ConfigValidationError


# =============================================================================
# Valid configurations — must not raise
# =============================================================================

class TestValidConfig:

    def test_minimal_valid_config(self, minimal_config):
        """A minimal valid config must pass validation without raising."""
        validate_config(minimal_config)

    def test_full_valid_config(self):
        cfg = {
            "models": {
                "refactor_adapter_path": "models/refactor_agent_final",
                "doc_adapter_path":      "models/doc_agent_final",
                "max_input_length":      512,
                "max_output_length":     256,
            },
            "pipeline": {
                "output_dir": "outputs",
            },
            "dacos": {"path": ""},
        }
        validate_config(cfg)  # must not raise

    def test_optional_dacos_path_can_be_empty(self):
        cfg = {
            "models": {
                "refactor_adapter_path": "models/r",
                "doc_adapter_path": "models/d",
            },
            "pipeline": {"output_dir": "outputs"},
            "dacos": {"path": ""},
        }
        validate_config(cfg)

    def test_optional_dacos_section_can_be_absent(self):
        cfg = {
            "models": {
                "refactor_adapter_path": "models/r",
                "doc_adapter_path": "models/d",
            },
            "pipeline": {"output_dir": "outputs"},
        }
        validate_config(cfg)


# =============================================================================
# Missing required fields — must raise ConfigValidationError
# =============================================================================

class TestMissingFields:

    def test_missing_models_section(self):
        cfg = {"pipeline": {"output_dir": "outputs"}}
        with pytest.raises(ConfigValidationError, match="models"):
            validate_config(cfg)

    def test_missing_refactor_adapter_path(self):
        cfg = {
            "models": {"doc_adapter_path": "models/d"},
            "pipeline": {"output_dir": "outputs"},
        }
        with pytest.raises(ConfigValidationError, match="refactor_adapter_path"):
            validate_config(cfg)

    def test_missing_doc_adapter_path(self):
        cfg = {
            "models": {"refactor_adapter_path": "models/r"},
            "pipeline": {"output_dir": "outputs"},
        }
        with pytest.raises(ConfigValidationError, match="doc_adapter_path"):
            validate_config(cfg)

    def test_missing_pipeline_section(self):
        cfg = {
            "models": {
                "refactor_adapter_path": "models/r",
                "doc_adapter_path": "models/d",
            }
        }
        with pytest.raises(ConfigValidationError, match="pipeline"):
            validate_config(cfg)

    def test_missing_output_dir(self):
        cfg = {
            "models": {
                "refactor_adapter_path": "models/r",
                "doc_adapter_path": "models/d",
            },
            "pipeline": {},
        }
        with pytest.raises(ConfigValidationError, match="output_dir"):
            validate_config(cfg)

    def test_empty_config_raises(self):
        with pytest.raises(ConfigValidationError):
            validate_config({})

    def test_none_config_raises(self):
        with pytest.raises((ConfigValidationError, TypeError)):
            validate_config(None)


# =============================================================================
# Wrong types — must raise ConfigValidationError
# =============================================================================

class TestWrongTypes:

    def _base(self):
        return {
            "models": {
                "refactor_adapter_path": "models/r",
                "doc_adapter_path": "models/d",
                "max_input_length": 512,
                "max_output_length": 256,
            },
            "pipeline": {"output_dir": "outputs"},
        }

    def test_max_input_length_must_be_int(self):
        cfg = self._base()
        cfg["models"]["max_input_length"] = "five-hundred"
        with pytest.raises(ConfigValidationError, match="max_input_length"):
            validate_config(cfg)

    def test_max_output_length_must_be_int(self):
        cfg = self._base()
        cfg["models"]["max_output_length"] = 3.14
        with pytest.raises(ConfigValidationError, match="max_output_length"):
            validate_config(cfg)

    def test_adapter_path_must_be_string(self):
        cfg = self._base()
        cfg["models"]["refactor_adapter_path"] = 42
        with pytest.raises(ConfigValidationError, match="refactor_adapter_path"):
            validate_config(cfg)

    def test_output_dir_must_be_string(self):
        cfg = self._base()
        cfg["pipeline"]["output_dir"] = ["outputs"]
        with pytest.raises(ConfigValidationError, match="output_dir"):
            validate_config(cfg)


# =============================================================================
# Out-of-range values — must raise ConfigValidationError
# =============================================================================

class TestOutOfRange:

    def _base(self):
        return {
            "models": {
                "refactor_adapter_path": "models/r",
                "doc_adapter_path": "models/d",
                "max_input_length": 512,
                "max_output_length": 256,
            },
            "pipeline": {"output_dir": "outputs"},
        }

    def test_max_input_length_must_be_positive(self):
        cfg = self._base()
        cfg["models"]["max_input_length"] = 0
        with pytest.raises(ConfigValidationError, match="max_input_length"):
            validate_config(cfg)

    def test_max_output_length_must_be_positive(self):
        cfg = self._base()
        cfg["models"]["max_output_length"] = -1
        with pytest.raises(ConfigValidationError, match="max_output_length"):
            validate_config(cfg)


# =============================================================================
# File-based validation — reading from disk
# =============================================================================

class TestConfigFile:

    def test_reads_valid_config_from_file(self, config_path, minimal_config):
        """validate_config should accept a path and read it."""
        validate_config(config_path)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        with pytest.raises((ConfigValidationError, FileNotFoundError)):
            validate_config(missing)

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not json }", encoding="utf-8")
        with pytest.raises((ConfigValidationError, json.JSONDecodeError)):
            validate_config(bad)