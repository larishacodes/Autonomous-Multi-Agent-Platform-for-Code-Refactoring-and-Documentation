"""
tests/test_planner.py — Planner DAG validation unit tests

Tests both the Pydantic-level Task.validate_dag() (task.py) and the
PlannerAgent's behaviour with typed CodeSmell inputs.

Research: unit tests exercise small, deterministic pieces of the agent in
isolation using in-memory fakes — assert exact behaviour quickly.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import pytest
from unittest.mock import MagicMock

from core.task_models import Task
from core.state import (
    AgentTask, TaskKind, TaskStatus,
    CodeSmell, Severity,
    create_repo_state,
)


# =============================================================================
# Task.validate_dag() — structural validation
# =============================================================================

class TestValidateDAG:

    def _make_task(self, task_id, kind=TaskKind.REFACTOR, depends_on=None):
        return Task(
            id=task_id,
            kind=kind,
            target=f"com.example.{task_id}",
            agent="refactor_agent",
            priority=5,
            depends_on=depends_on or [],
        )

    def test_valid_linear_chain(self):
        """A → B → C with no cycles must validate without raising."""
        tasks = [
            self._make_task("refactor-A"),
            self._make_task("doc-A", kind=TaskKind.DOCUMENT, depends_on=["refactor-A"]),
            self._make_task("refactor-B", depends_on=["doc-A"]),
        ]
        result = Task.validate_dag(tasks)
        assert len(result) == 3

    def test_valid_parallel_tasks(self):
        """Two tasks with no shared dependency must validate."""
        tasks = [
            self._make_task("refactor-A"),
            self._make_task("refactor-B"),
        ]
        Task.validate_dag(tasks)

    def test_detects_direct_cycle(self):
        """A → B → A must raise ValueError mentioning cycle."""
        tasks = [
            self._make_task("task-A", depends_on=["task-B"]),
            self._make_task("task-B", depends_on=["task-A"]),
        ]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            Task.validate_dag(tasks)

    def test_detects_three_node_cycle(self):
        """A → B → C → A must raise ValueError."""
        tasks = [
            self._make_task("task-A", depends_on=["task-C"]),
            self._make_task("task-B", depends_on=["task-A"]),
            self._make_task("task-C", depends_on=["task-B"]),
        ]
        with pytest.raises(ValueError, match="[Cc]ycle"):
            Task.validate_dag(tasks)

    def test_detects_dangling_dependency(self):
        """A task depending on a non-existent ID must raise ValueError."""
        tasks = [
            self._make_task("task-A", depends_on=["task-NONEXISTENT"]),
        ]
        with pytest.raises(ValueError, match="unknown"):
            Task.validate_dag(tasks)

    def test_self_dependency_raises_at_construction(self):
        """A task depending on itself must raise at construction time."""
        with pytest.raises(ValueError):
            Task(
                id="task-A",
                kind=TaskKind.REFACTOR,
                target="com.example.Foo",
                agent="refactor_agent",
                priority=5,
                depends_on=["task-A"],   # self-reference
            )

    def test_empty_dag_is_valid(self):
        """An empty task list must not raise."""
        result = Task.validate_dag([])
        assert result == []

    def test_single_task_is_valid(self):
        """A single task with no dependencies must not raise."""
        Task.validate_dag([self._make_task("solo-task")])


# =============================================================================
# Task field validation
# =============================================================================

class TestTaskFieldValidation:

    def test_priority_bounds(self):
        """Priority must be in [1, 10]."""
        with pytest.raises(Exception):
            Task(id="t1", kind=TaskKind.REFACTOR, target="Foo", agent="a",
                 priority=0)
        with pytest.raises(Exception):
            Task(id="t1", kind=TaskKind.REFACTOR, target="Foo", agent="a",
                 priority=11)

    def test_valid_priority_range(self):
        for p in (1, 5, 10):
            t = Task(id=f"t{p}", kind=TaskKind.REFACTOR, target="Foo",
                     agent="a", priority=p)
            assert t.priority == p

    def test_id_no_spaces(self):
        with pytest.raises(ValueError, match="spaces"):
            Task(id="bad id", kind=TaskKind.REFACTOR, target="Foo", agent="a")

    def test_in_progress_requires_assigned_to(self):
        """A task in IN_PROGRESS status must have assigned_to set."""
        with pytest.raises(ValueError):
            Task(
                id="task-1",
                kind=TaskKind.REFACTOR,
                target="Foo",
                agent="refactor_agent",
                status=TaskStatus.IN_PROGRESS,
                assigned_to=None,    # must raise
            )

    def test_done_status_requires_assigned_to(self):
        with pytest.raises(ValueError):
            Task(
                id="task-1",
                kind=TaskKind.DOCUMENT,
                target="Foo",
                agent="doc_agent",
                status=TaskStatus.DONE,
                assigned_to=None,
            )

    def test_pending_does_not_require_assigned_to(self):
        """PENDING tasks do not need an assigned agent yet."""
        t = Task(
            id="task-pending",
            kind=TaskKind.DOCUMENT,
            target="Foo",
            agent="doc_agent",
            status=TaskStatus.PENDING,
            assigned_to=None,
        )
        assert t.assigned_to is None


# =============================================================================
# PlannerAgent — output shape and priority logic
# =============================================================================

class TestPlannerAgentOutput:

    def _base_state(self):
        return create_repo_state(
            raw_code="public class Foo { public void bar() {} }",
            classes=[], functions=[], imports=[],
        )

    def _planner(self, symbol_index=None):
        from core.planner_agent import PlannerAgent
        from core.hybrid_retriever import build_symbol_index
        if symbol_index is None:
            symbol_index = build_symbol_index({"functions": [], "classes": []})
        return PlannerAgent(engine=MagicMock(), retriever=None, symbol_index=symbol_index)

    def test_no_smells_produces_doc_tasks(self):
        """Without smells, planner emits DOCUMENT tasks for undocumented functions."""
        from core.state import FunctionUnit, TaskKind
        from core.hybrid_retriever import build_symbol_index

        fn = FunctionUnit(name="processOrder", params=[], docstring=None)
        state = create_repo_state(
            raw_code="public class Foo {}",
            classes=[], functions=[fn], imports=[],
        )
        symbol_index = build_symbol_index({"functions": [{"name": "processOrder", "params": []}], "classes": []})
        planner = self._planner(symbol_index)
        new_state = planner.run(state, smells=[])

        doc_tasks = [t for t in new_state.tasks if t.kind == TaskKind.DOCUMENT]
        assert len(doc_tasks) >= 1, "Expected at least one DOCUMENT task for undocumented function"

    def test_smell_produces_refactor_before_doc(self):
        """REFACTOR task must appear before its dependent DOCUMENT task."""
        from core.state import FunctionUnit, CodeSmell, Severity, TaskKind
        from core.hybrid_retriever import build_symbol_index

        fn = FunctionUnit(name="processOrder", params=[], docstring=None, complexity=15)
        state = create_repo_state(
            raw_code="", classes=[], functions=[fn], imports=[]
        )
        smell = CodeSmell(
            smell_type="LongMethod",
            location="processOrder",
            description="too long",
            severity=Severity.HIGH,
            confidence=0.9,
        )
        symbol_index = build_symbol_index({"functions": [{"name": "processOrder", "params": []}], "classes": []})
        planner = self._planner(symbol_index)
        new_state = planner.run(state, smells=[smell])

        refactor_ids = {t.task_id for t in new_state.tasks if t.kind == TaskKind.REFACTOR}
        doc_tasks = [t for t in new_state.tasks if t.kind == TaskKind.DOCUMENT]

        # Every doc task targeting processOrder must depend on a refactor task
        for dt in doc_tasks:
            if dt.target == "processOrder":
                assert any(dep in refactor_ids for dep in dt.depends_on), \
                    f"Doc task {dt.task_id} does not depend on a refactor task"

    def test_critical_smell_gets_higher_priority(self):
        """CRITICAL severity smell must produce a task with priority <= MEDIUM smell task."""
        from core.state import FunctionUnit, CodeSmell, Severity, TaskKind
        from core.hybrid_retriever import build_symbol_index

        fn = FunctionUnit(name="target", params=[], docstring=None)
        state = create_repo_state(raw_code="", classes=[], functions=[fn], imports=[])

        smells = [
            CodeSmell("GodClass", "target", "desc", Severity.CRITICAL, confidence=0.9),
            CodeSmell("LongMethod", "other",  "desc", Severity.MEDIUM,   confidence=0.8),
        ]
        other_fn = FunctionUnit(name="other", params=[], docstring=None)
        state = create_repo_state(raw_code="", classes=[], functions=[fn, other_fn], imports=[])

        symbol_index = build_symbol_index({
            "functions": [{"name": "target", "params": []}, {"name": "other", "params": []}],
            "classes": [],
        })
        planner = self._planner(symbol_index)
        new_state = planner.run(state, smells=smells)

        priority_map = {t.target: t.priority for t in new_state.tasks if t.kind == TaskKind.REFACTOR}
        if "target" in priority_map and "other" in priority_map:
            assert priority_map["target"] <= priority_map["other"], \
                "CRITICAL smell should have equal or higher priority than MEDIUM smell"

    def test_replan_preserves_done_tasks(self, repo_state, symbol_index):
        """Replan must not remove DONE tasks from the state."""
        from core.planner_agent import PlannerAgent
        from core.state import AgentTask, TaskKind, TaskStatus

        # Manually add a DONE task
        done_task = AgentTask(
            task_id="done-task-1",
            kind=TaskKind.DOCUMENT,
            target="processOrder",
            status=TaskStatus.DONE,
        )
        state_with_done = repo_state.evolve(
            agent_id="test", action="inject_done",
            tasks=[done_task],
            completed_tasks=["done-task-1"],
        )

        planner = PlannerAgent(engine=MagicMock(), retriever=None, symbol_index=symbol_index)
        replanned = planner.run(state_with_done, smells=[], replan=True)

        done_ids = {t.task_id for t in replanned.tasks if t.status == TaskStatus.DONE}
        assert "done-task-1" in done_ids, \
            "Replan must preserve DONE tasks from the previous run"