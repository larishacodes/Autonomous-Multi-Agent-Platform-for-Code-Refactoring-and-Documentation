# pipeline.py — Block 4: Agent Orchestration
#
# Flow:
#   input.java
#     → [Block 2] parser/java_parser.py          → parsed metrics
#     → [Block 3] prompt_engine/                 → smell report + 2 prompts
#     → [Block 4] core/repostate.py              → RepoState (NEW)
#     → [Block 4] core.planner_agent.py               → task DAG (NEW)
#     → [Block 5] supervisor.py                  → dispatch + retry (NEW)
#         → agents/refactor_agent.py             → refactored Java
#         → agents/doc_agent.py                  → documentation
#     → [Block 6] evaluator_agent.py             → scores → RepoState (NEW)
#         → evaluator/refactor_evaluator         → refactor scores
#         → evaluator/doc_evaluator              → doc scores
#     → outputs/run_TIMESTAMP/                   → all saved files
#
# Output files (unchanged):
#   parsed_analysis.json          parser metrics per method
#   smell_report.txt              detected smells with reasons
#   refactoring_plan.txt          action plan per smell
#   PROMPT_refactor_agent.txt     exact prompt sent to Refactor Agent
#   PROMPT_doc_agent.txt          exact prompt sent to Doc Agent
#   refactored_code.java          Refactor Agent output
#   documentation.md              Doc Agent output
#   EVALUATION_refactor.txt       refactor evaluation (human-readable)
#   EVALUATION_refactor.json      refactor evaluation (machine-readable)
#   EVALUATION_doc.txt            doc evaluation (human-readable)
#   EVALUATION_doc.json           doc evaluation (machine-readable)
#   summary.json                  one-page summary of all scores
#   provenance_log.json           NEW: full agent-level audit trail

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

from core import state
from parser.java_parser import JavaParser
from prompt_engine.prompting_engine import PromptingEngine
from prompt_engine.smell_detector import SmellDetector
from agents.refactor_agent import RefactorAgent
from agents.doc_agent import DocAgent
from evaluator.refactor_evaluator import save_evaluation_report
from evaluator.doc_evaluator import save_doc_evaluation_report

# ── NEW: agent layer imports ──────────────────────────────────────────────────
from core.state import (
    create_repo_state,
    CodeSmell,
    Severity,
    FunctionUnit,
    ClassUnit,
)
from core.planner_agent import PlannerAgent
from core.supervisor import SupervisorAgent
from core.evaluator import EvaluatorAgent

# ── CHANGE 1 (of 6): hybrid retriever needed by PlannerAgent ─────────────────
# If your hybrid_retrieve / symbol_index setup lives elsewhere, adjust imports.
from core.hybrid_retriever import hybrid_retrieve, build_symbol_index


def _load_config() -> dict:
    cfg = ROOT / "config.json"
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _severity_from_str(s: str) -> Severity:
    """Map SmellDetector severity strings to typed Severity enum."""
    return {
        "critical": Severity.CRITICAL,
        "high":     Severity.HIGH,
        "medium":   Severity.MEDIUM,
        "low":      Severity.LOW,
    }.get(s.lower(), Severity.MEDIUM)


def _build_function_units(parsed: dict) -> list[FunctionUnit]:
    """Convert parser output dicts to typed FunctionUnit instances."""
    units = []
    for f in parsed.get("functions", []):
        units.append(FunctionUnit(
            name=f.get("name", "unknown"),
            params=f.get("params", []),
            docstring=f.get("docstring"),
            file_path=f.get("file_path", ""),
            start_line=f.get("start_line", 0),
            end_line=f.get("end_line", 0),
            calls=f.get("calls", []),
            called_by=f.get("called_by", []),
            complexity=f.get("complexity", 0),
            loc=f.get("loc", 0),
            cohesion_score=f.get("cohesion_score", 1.0),
        ))
    return units


def _build_class_units(parsed: dict) -> list[ClassUnit]:
    """Convert parser output dicts to typed ClassUnit instances."""
    units = []
    for c in parsed.get("classes", []):
        units.append(ClassUnit(
            name=c.get("name", "unknown"),
            methods=[],      # populated separately if needed
            docstring=c.get("docstring"),
            file_path=c.get("file_path", ""),
            superclass=c.get("superclass"),
            interfaces=c.get("interfaces", []),
            is_abstract=c.get("is_abstract", False),
            lcom=c.get("lcom", 0.0),
            instability=c.get("instability", 0.0),
        ))
    return units


def _build_code_smells(smells: list[dict]) -> list[CodeSmell]:
    """
    Convert SmellDetector raw dicts to typed CodeSmell instances.

    # CHANGE 2: replaces the isinstance(smell, dict) duck-typing in the
    # old PlannerAgent.  All downstream code now works with typed objects.
    """
    result = []
    for s in smells:
        result.append(CodeSmell(
            smell_type=s.get("name", "unknown"),
            location=s.get("function", "unknown"),
            description=(
                f"{s.get('metric', '')}={s.get('value', '')} "
                f"(threshold={s.get('threshold', '')})"
            ),
            severity=_severity_from_str(s.get("severity", "medium")),
            agent_id="smell_detector",
            confidence=s.get("confidence", 1.0),
            reasoning=s.get("reason", ""),
        ))
    return result


class Pipeline:

    def __init__(self):
        config = _load_config()

        self.refactor_adapter = str(ROOT / config.get("models", {})
                                    .get("refactor_adapter_path", "models/refactor_agent_final"))
        self.doc_adapter      = str(ROOT / config.get("models", {})
                                    .get("doc_adapter_path", "models/doc_agent_final"))
        self.output_dir       = ROOT / config.get("pipeline", {}).get("output_dir", "outputs")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        dacos_path = config.get("dacos", {}).get("path", "") or None
        max_in     = config.get("models", {}).get("max_input_length",  512)
        max_out    = config.get("models", {}).get("max_output_length", 256)

        self.parser         = JavaParser()
        self.engine         = PromptingEngine(model_type="codet5p-770m",
                                              dacos_folder=dacos_path)
        self.detector       = SmellDetector(dacos_folder=dacos_path)

        # Leaf agents (unchanged — Supervisor calls these internally)
        self.refactor_agent = RefactorAgent(adapter_path=self.refactor_adapter,
                                            max_input_length=max_in,
                                            max_output_length=max_out)
        self.doc_agent      = DocAgent(adapter_path=self.doc_adapter,
                                       max_input_length=max_in,
                                       max_output_length=256)

        # ── CHANGE 3: build retriever + agent layer ───────────────────────
        # symbol_index is built lazily per-run in run() once we have parsed
        # output; the retriever itself is stateless.
        self.retriever = None          # set in run() after parsing
        self.symbol_index = None       # set in run() after parsing

    def run(self, source_code: str, mode: str = "both") -> dict:

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{ts}"
        run_dir.mkdir(exist_ok=True)

        result = {"timestamp": ts, "run_dir": str(run_dir),
                  "mode": mode, "success": False}

        # ── BLOCK 2: Parse ────────────────────────────────────────────────
        logger.info("[Block 2]  Parsing ...")
        parsed = self.parser.parse(source_code)
        parsed["original_code"] = source_code

        if not parsed.get("parse_success", True):
            logger.error("Parse failed: %s", parsed.get("error"))
            return result

        logger.info(
            "           %d method(s)  %d class(es)  parser=%s",
            len(parsed["functions"]),
            len(parsed["classes"]),
            parsed["parser_used"],
        )

        (run_dir / "parsed_analysis.json").write_text(
            json.dumps({k: v for k, v in parsed.items() if k != "original_code"},
                       indent=2, default=str), encoding="utf-8")

        # ── BLOCK 3: Prompting Engine ─────────────────────────────────────
        logger.info("[Block 3]  Detecting smells + generating prompts ...")
        raw_smells  = self.detector.detect_smells(parsed)
        report      = self.detector.generate_report(parsed)
        plan        = self.engine.generate_refactoring_plan(parsed)
        prompts     = self.engine.generate_prompts(source_code, parsed, "both")

        refactor_prompt = prompts.get("refactor_prompt", "")
        doc_prompt      = prompts.get("documentation_prompt", "")

        logger.info("           %d smell(s) detected", len(raw_smells))
        for s in raw_smells:
            logger.info(
                "              [%s] %s — %s  (%s=%s, threshold=%s)",
                s["severity"], s["name"], s["function"],
                s["metric"], s["value"], s["threshold"],
            )

        (run_dir / "smell_report.txt").write_text(report, encoding="utf-8")
        (run_dir / "refactoring_plan.txt").write_text(plan, encoding="utf-8")
        (run_dir / "PROMPT_refactor_agent.txt").write_text(refactor_prompt, encoding="utf-8")
        (run_dir / "PROMPT_doc_agent.txt").write_text(doc_prompt, encoding="utf-8")
        result["smells"] = [{"name": s["name"], "severity": s["severity"],
                              "function": s["function"]} for s in raw_smells]

        # ── BLOCK 4: Build RepoState + Task DAG ──────────────────────────
        # CHANGE 4: construct RepoState and run the Planner
        logger.info("[Block 4]  Building RepoState + task DAG ...")

        functions   = _build_function_units(parsed)
        classes     = _build_class_units(parsed)
        code_smells = _build_code_smells(raw_smells)

        repo_state = create_repo_state(
            raw_code=source_code,
            classes=classes,
            functions=functions,
            imports=parsed.get("imports", []),
            metadata={
                "parser_used": parsed.get("parser_used"),
                "total_loc":   parsed.get("total_loc"),
                "mode":        mode,
            },
        )
        # Attach typed smells to state
        repo_state = repo_state.evolve(
            agent_id="pipeline",
            action="smells_attached",
            summary=f"{len(code_smells)} smell(s) from SmellDetector",
            smells=code_smells,
        )

        # Build symbol index from parsed functions (used by PlannerAgent retrieval)
        self.symbol_index = build_symbol_index(parsed)

        planner   = PlannerAgent(
            engine=self.engine,
            retriever=self.retriever,
            symbol_index=self.symbol_index,
        )
        repo_state = planner.run(repo_state, code_smells)
        logger.info("           %d task(s) in DAG", len(repo_state.tasks))

        # ── BLOCK 5: Supervisor dispatches tasks ──────────────────────────
        # CHANGE 5: replace inline Block 5a/b + 6a/b with Supervisor call.
        # EvaluatorAgent is wired inside the Supervisor's feedback loop;
        # we also run it explicitly here so pipeline.py can read scores.
        if mode in ("refactor", "both", "document"):
            logger.info("[Block 5]  Supervisor dispatching tasks ...")

            supervisor = SupervisorAgent(
                refactor_agent=self.refactor_agent,
                doc_agent=self.doc_agent,
                planner_agent=planner,
            )
            repo_state = supervisor.run(repo_state)

        # ── BLOCK 6: Evaluator writes scores into RepoState ──────────────
        logger.info("[Block 6]  Evaluating results ...")
        evaluator  = EvaluatorAgent(run_dir=run_dir)
        repo_state, eval_summary = evaluator.run(repo_state)
        print("AFTER EVAL:", repo_state.evaluation_scores)

        # ── Extract outputs from RepoState for file saving ────────────────
        # CHANGE 6: read from RepoState instead of local variables.
        if repo_state.refactor_results:
            primary_refactor = repo_state.refactor_results[0]
            refactored_code  = primary_refactor.changes
            (run_dir / "refactored_code.java").write_text(
                refactored_code, encoding="utf-8"
            )
            result["refactored_code"]     = refactored_code
            result["refactor_used_model"] = True   # model was used if result exists

            if eval_summary.refactor:
                re = eval_summary.refactor
                result["refactor_evaluation"] = {
                    "confidence_score":  re.confidence,
                    "confidence_status": re.status,
                    "ast_valid":         bool(re.raw.get("ast_validity", {}).get("valid")),
                    "style_score":       re.raw.get("style_metrics", {}).get("score", 0.0),
                    "codebleu":          re.raw.get("codebleu_score", 0.0),
                    "semantic_score":    re.raw.get("semantic_preservation", {}).get("score", 0.0),
                    "improvement_score": re.raw.get("improvement", {}).get("score", 0.0),
                    "loc_original":      re.raw.get("improvement", {}).get("loc_original", 0),
                    "loc_refactored":    re.raw.get("improvement", {}).get("loc_refactored", 0),
                }

        if repo_state.documentation_results:
            primary_doc   = repo_state.documentation_results[0]
            documentation = primary_doc.docstring
            (run_dir / "documentation.md").write_text(
                documentation, encoding="utf-8"
            )
            result["documentation"]  = documentation
            result["doc_used_model"] = True

            if eval_summary.doc:
                de = eval_summary.doc
                result["doc_evaluation"] = {
                    "confidence_score":   de.confidence,
                    "confidence_status":  de.status,
                    "coverage_score":     de.raw.get("coverage", {}).get("score", 0.0),
                    "completeness_score": de.raw.get("completeness", {}).get("score", 0.0),
                }

        # ── BLOCK 7: Final Result + Save ──────────────────────────────────
        result["success"] = True
        self._save_summary(result, run_dir, parsed, repo_state)
        self._print_final(result, run_dir)

        # NEW: save provenance log for audit trail
        provenance = [
            {"version": e.version, "agent": e.agent_id,
             "action": e.action, "summary": e.summary}
            for e in repo_state.provenance_log
        ]
        with open(run_dir / "provenance_log.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

        return result

    def _save_summary(self, result, run_dir, parsed, repo_state: state):
        summary = {
            "timestamp": result["timestamp"],
            "mode":      result["mode"],
            "input": {
                "parser":  parsed.get("parser_used"),
                "loc":     parsed.get("total_loc"),
                "methods": len(parsed.get("functions", [])),
                "classes": len(parsed.get("classes", [])),
                "smells":  result.get("smells", []),
            },
            "refactor":          result.get("refactor_evaluation"),
            "doc":               result.get("doc_evaluation"),
            "evaluation_scores": repo_state.evaluation_scores,
            "tasks_completed":   len(repo_state.completed_tasks),
            "tasks_total":       len(repo_state.tasks),
            "state_version":     repo_state.version,
            "files":             sorted(p.name for p in run_dir.iterdir()),
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _print_final(self, result, run_dir):
        print("\n" + "=" * 60)
        print("  FINAL RESULT")
        print("=" * 60)

        if "refactor_evaluation" in result:
            re  = result["refactor_evaluation"]
            loc = f"{re['loc_original']} → {re['loc_refactored']}"
            v   = ("ACCEPTED"              if re["confidence_score"] >= 0.75
                   else "CONDITIONALLY ACCEPTED" if re["confidence_score"] >= 0.65
                   else "REJECTED")
            print(f"\n  REFACTORING")
            print(f"     Confidence  : {re['confidence_score']:.3f}  {re['confidence_status']}")
            print(f"     AST Valid   : {'YES' if re['ast_valid'] else 'NO'}")
            print(f"     Style       : {re['style_score']:.3f}")
            print(f"     CodeBLEU    : {re['codebleu']:.3f}")
            print(f"     Semantic    : {re['semantic_score']:.3f}")
            print(f"     Improvement : {re['improvement_score']:.3f}")
            print(f"     LOC change  : {loc}")
            print(f"     Verdict     : {v}")

        if "doc_evaluation" in result:
            de = result["doc_evaluation"]
            v  = ("ACCEPTED"              if de["confidence_score"] >= 0.75
                  else "CONDITIONALLY ACCEPTED" if de["confidence_score"] >= 0.65
                  else "REJECTED")
            print(f"\n  DOCUMENTATION")
            print(f"     Confidence   : {de['confidence_score']:.3f}  {de['confidence_status']}")
            print(f"     Coverage     : {de['coverage_score']:.3f}")
            print(f"     Completeness : {de['completeness_score']:.3f}")
            print(f"     Verdict      : {v}")

        print(f"\n  Outputs : {run_dir}")
        print(f"\n  Files:")
        descs = {
            "parsed_analysis.json":      "Parser metrics (LOC, params, conditionals)",
            "smell_report.txt":          "Detected smells with reasons",
            "refactoring_plan.txt":      "Action plan per smell",
            "PROMPT_refactor_agent.txt": "Exact prompt sent to Refactor Agent",
            "PROMPT_doc_agent.txt":      "Exact prompt sent to Doc Agent",
            "refactored_code.java":      "Refactor Agent output",
            "documentation.md":          "Doc Agent output",
            "EVALUATION_refactor.txt":   "Refactor evaluation (human-readable)",
            "EVALUATION_refactor.json":  "Refactor scores (machine-readable)",
            "EVALUATION_doc.txt":        "Doc evaluation (human-readable)",
            "EVALUATION_doc.json":       "Doc scores (machine-readable)",
            "summary.json":              "One-page summary of all scores",
            "provenance_log.json":       "Agent-level audit trail (NEW)",
        }
        for f in sorted(run_dir.iterdir()):
            desc = descs.get(f.name, "")
            print(f"    {f.name:<40} {desc}")

        print("=" * 60)