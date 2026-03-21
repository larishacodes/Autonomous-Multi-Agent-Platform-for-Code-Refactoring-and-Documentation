from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Layer 3 — Orchestration: Supervisor
# =============================================================================

class TestSupervisorOrchestration:

    def _make_planner_state(self, repo_state, symbol_index):
        from core.planner_agent import PlannerAgent
        from core.state import CodeSmell, Severity

        smell = CodeSmell(
            smell_type="LongMethod",
            location="processOrder",
            description="too long",
            severity=Severity.MEDIUM,
            confidence=0.85,
        )

        planner = PlannerAgent(
            engine=MagicMock(),
            retriever=None,
            symbol_index=symbol_index,
        )

        return planner.run(repo_state, smells=[smell])

    def test_supervisor_completes_tasks(
        self, repo_state, symbol_index, mock_refactor_agent, mock_doc_agent
    ):
        from core.supervisor import SupervisorAgent

        planner_state = self._make_planner_state(repo_state, symbol_index)

        # ✅ correct mocking
        mock_planner = MagicMock()
        mock_planner.run.return_value = planner_state

        supervisor = SupervisorAgent(
            refactor_agent=mock_refactor_agent,
            doc_agent=mock_doc_agent,
            planner_agent=mock_planner,
        )

        final_state = supervisor.run(planner_state)

        assert len(final_state.completed_tasks) > 0

    def test_supervisor_writes_results(
        self, repo_state, symbol_index, mock_refactor_agent, mock_doc_agent
    ):
        from core.supervisor import SupervisorAgent

        planner_state = self._make_planner_state(repo_state, symbol_index)

        mock_planner = MagicMock()
        mock_planner.run.return_value = planner_state

        supervisor = SupervisorAgent(
            refactor_agent=mock_refactor_agent,
            doc_agent=mock_doc_agent,
            planner_agent=mock_planner,
        )

        final_state = supervisor.run(planner_state)

        assert (
            len(final_state.refactor_results) > 0
            or len(final_state.documentation_results) > 0
        )

    def test_supervisor_provenance(
        self, repo_state, symbol_index, mock_refactor_agent, mock_doc_agent
    ):
        from core.supervisor import SupervisorAgent

        planner_state = self._make_planner_state(repo_state, symbol_index)

        mock_planner = MagicMock()
        mock_planner.run.return_value = planner_state

        supervisor = SupervisorAgent(
            refactor_agent=mock_refactor_agent,
            doc_agent=mock_doc_agent,
            planner_agent=mock_planner,
        )

        final_state = supervisor.run(planner_state)

        actions = [e.action for e in final_state.provenance_log]

        assert any("task_started" in a for a in actions)
        assert any("task_done" in a or "task_failed" in a for a in actions)

    def test_failed_task_cancels_dependents(self, repo_state, symbol_index):
        from core.supervisor import SupervisorAgent
        from core.state import TaskKind

        # ❌ failing refactor agent
        failing_refactor = MagicMock()
        failing_refactor.run.side_effect = ValueError("failure")

        mock_doc = MagicMock()
        mock_doc.run.return_value = {
            "documentation": "doc",
            "confidence": 0.8,
        }

        planner_state = self._make_planner_state(repo_state, symbol_index)

        mock_planner = MagicMock()
        mock_planner.run.return_value = planner_state

        supervisor = SupervisorAgent(
            refactor_agent=failing_refactor,
            doc_agent=mock_doc,
            planner_agent=mock_planner,
        )

        final_state = supervisor.run(planner_state)

        completed = set(final_state.completed_tasks)

        refactor_tasks = [
            t for t in planner_state.tasks if t.kind == TaskKind.REFACTOR
        ]

        for task in refactor_tasks:
            assert task.task_id not in completed


# =============================================================================
# Layer 3 — Evaluator
# =============================================================================

class TestEvaluatorAgent:

    def _make_refactor_verdict(self, confidence=0.8):
        from core.evaluator import AgentVerdict

        return AgentVerdict(
            confidence=confidence,
            status="ACCEPTED" if confidence >= 0.75 else "REJECTED",
            accepted=confidence >= 0.75,
            conditional=False,
            needs_replan=confidence < 0.65,
            needs_human=False,
            raw={},
        )

    def _make_doc_verdict(self, confidence=0.85):
        from core.evaluator import AgentVerdict

        return AgentVerdict(
            confidence=confidence,
            status="ACCEPTED",
            accepted=True,
            conditional=False,
            needs_replan=False,
            needs_human=False,
            raw={},
        )

    def test_evaluator_writes_scores(self, repo_state, tmp_path):
        from core.evaluator import EvaluatorAgent
        from core.state import RefactorResult, DocumentationResult

        state = repo_state.evolve(
            agent_id="test",
            action="inject",
            refactor_results=[
                RefactorResult(
                    task_id="r1",
                    target_name="processOrder",
                    success=True,
                    changes="code",
                    confidence=0.8,
                )
            ],
            documentation_results=[
                DocumentationResult(
                    task_id="d1",
                    target_name="processOrder",
                    docstring="doc",
                    confidence=0.85,
                )
            ],
        )

        evaluator = EvaluatorAgent(run_dir=tmp_path)

        evaluator._evaluate_refactor = MagicMock(
            return_value=self._make_refactor_verdict(0.8)
        )
        evaluator._evaluate_doc = MagicMock(
            return_value=self._make_doc_verdict(0.85)
        )

        final_state, summary = evaluator.run(state)

        scores = {}
        if summary.refactor:
            scores["refactor/confidence"] = summary.refactor.confidence
        if summary.doc:
            scores["doc/confidence"] = summary.doc.confidence

        assert "refactor/confidence" in final_state.evaluation_scores
        assert "doc/confidence" in final_state.evaluation_scores


# =============================================================================
# Smoke test
# =============================================================================

@pytest.mark.integration
class TestPipelineSmoke:

    def test_pipeline_runs(self, sample_java_source, tmp_path, minimal_config):

        with patch("pipeline.RefactorAgent") as MockRA, \
             patch("pipeline.DocAgent") as MockDA, \
             patch("pipeline.build_symbol_index"), \
             patch("pipeline.PlannerAgent") as MockPlanner, \
             patch("pipeline.SupervisorAgent") as MockSupervisor, \
             patch("pipeline.EvaluatorAgent") as MockEval:

            MockRA.return_value.run.return_value = {
                "refactored_code": "class A {}",
                "confidence": 0.8,
            }

            MockDA.return_value.run.return_value = {
                "documentation": "docs",
                "confidence": 0.85,
            }

            from core.state import create_repo_state
            empty_state = create_repo_state(
                raw_code=sample_java_source,
                classes=[],
                functions=[],
                imports=[],
            )

            MockPlanner.return_value.run.return_value = empty_state
            MockSupervisor.return_value.run.return_value = empty_state

            from core.evaluator import EvaluationSummary
            MockEval.return_value.run.return_value = (
                empty_state,
                EvaluationSummary(),
            )

            with patch("pipeline._load_config", return_value=minimal_config):
                from pipeline import Pipeline

                pipeline = Pipeline()
                pipeline.output_dir = tmp_path

                result = pipeline.run(sample_java_source)

        assert result.get("success") is True