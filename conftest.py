"""
tests/conftest.py — Shared pytest fixtures

Research (LangChain docs, 2025): unit tests use in-memory fakes to assert
exact behaviour quickly and deterministically.  Integration tests use real
component calls with mocked LLM boundaries.

Fixture layers
--------------
1. sample_java     — the raw fixture source code (no I/O)
2. parsed          — JavaParser output (pure Python, no models)
3. repo_state      — RepoState built from parsed output (no models)
4. symbol_index    — SymbolIndex from build_symbol_index (no models)
5. mock_agent      — callable stub that returns a canned dict
6. mock_refactor_agent / mock_doc_agent — domain-specific stubs
7. minimal_config  — a valid config.json dict
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Fixture file paths ────────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "tests" / "fixtures"
SAMPLE_JAVA = Path(__file__).parent / "tests" / "fixtures" / "Sample.java"


# ── 1. Raw source ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_java_source() -> str:
    """Read the Sample.java fixture once per test session."""
    return SAMPLE_JAVA.read_text(encoding="utf-8")


# ── 2. Parsed output ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def parsed(sample_java_source) -> dict:
    """
    Run JavaParser on the fixture.  Scoped to session — parsing is
    deterministic and expensive to repeat.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from parser.java_parser import JavaParser
    return JavaParser().parse(sample_java_source, file_path="Sample.java")


# ── 3. RepoState ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def repo_state(parsed, sample_java_source):
    """Build an initial RepoState from the parsed fixture."""
    from pipeline import _build_function_units, _build_class_units
    from core.state import create_repo_state

    functions = _build_function_units(parsed)
    classes   = _build_class_units(parsed)

    return create_repo_state(
        raw_code=sample_java_source,
        classes=classes,
        functions=functions,
        imports=parsed.get("imports", []),
        metadata={"parser_used": parsed.get("parser_used"), "mode": "both"},
    )


# ── 4. SymbolIndex ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def symbol_index(parsed):
    """Build a SymbolIndex from the parsed fixture (no embeddings)."""
    from core.hybrid_retriever import build_symbol_index
    return build_symbol_index(parsed)


# ── 5. Generic mock agent ─────────────────────────────────────────────────────

@pytest.fixture
def mock_agent():
    """
    A callable stub that accepts any args and returns a generic success dict.
    Research: mock the LLM boundary, not the orchestration logic.
    """
    agent = MagicMock()
    agent.run.return_value = {
        "refactored_code": "// refactored\npublic class OrderService {}",
        "documentation":   "## OrderService\nProcesses orders.",
        "used_model":       True,
        "confidence":       0.85,
    }
    return agent


# ── 6. Domain-specific agent stubs ───────────────────────────────────────────

@pytest.fixture
def mock_refactor_agent():
    """Stub for RefactorAgent — returns valid Java and high confidence."""
    agent = MagicMock()
    agent.run.return_value = {
        "refactored_code": "public class OrderService { /* refactored */ }",
        "used_model":       True,
        "confidence":       0.82,
    }
    return agent


@pytest.fixture
def mock_doc_agent():
    """Stub for DocAgent — returns markdown documentation."""
    agent = MagicMock()
    agent.run.return_value = {
        "documentation": "## OrderService\n\nProcesses customer orders.",
        "used_model":     True,
        "confidence":     0.88,
    }
    return agent


@pytest.fixture
def mock_llm():
    """LangChain-compatible LLM stub for rag_query tests."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="The processOrder method handles order lifecycle.")
    return llm


# ── 7. Config fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config(tmp_path) -> dict:
    """A valid minimal config.json dict written to a temp directory."""
    cfg = {
        "models": {
            "refactor_adapter_path": "models/refactor_agent_final",
            "doc_adapter_path":      "models/doc_agent_final",
            "max_input_length":      512,
            "max_output_length":     256,
        },
        "pipeline": {
            "output_dir": str(tmp_path / "outputs"),
        },
        "dacos": {"path": ""},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


@pytest.fixture
def config_path(tmp_path, minimal_config) -> Path:
    """Path to the written config.json in tmp_path."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(minimal_config), encoding="utf-8")
    return path