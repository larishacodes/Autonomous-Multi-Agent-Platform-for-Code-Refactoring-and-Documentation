# pipeline.py — Block 4: Agent Orchestration
#
# Flow:
#   input.java
#     → [Block 2] parser/java_parser.py          → parsed metrics
#     → [Block 3] prompt_engine/                 → smell report + 2 prompts
#     → [Block 5a] agents/refactor_agent.py      → refactored Java
#     → [Block 5b] agents/doc_agent.py           → documentation
#     → [Block 6a] evaluator/refactor_evaluator  → refactor scores
#     → [Block 6b] evaluator/doc_evaluator       → doc scores
#     → outputs/run_TIMESTAMP/                   → all saved files
#
# Output files (clearly named):
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

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# All third-party noise is suppressed in main.py before this file loads.
# This logger only shows warnings and errors from our own code.
logger = logging.getLogger(__name__)

from parser.java_parser import JavaParser
from prompt_engine.prompting_engine import PromptingEngine
from prompt_engine.smell_detector import SmellDetector
from agents.refactor_agent import RefactorAgent
from agents.doc_agent import DocAgent
from evaluator.refactor_evaluator import analyze_refactoring, save_evaluation_report
from evaluator.doc_evaluator import evaluate_documentation, save_doc_evaluation_report


def _load_config() -> dict:
    cfg = ROOT / "config.json"
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    return {}


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
        self.refactor_agent = RefactorAgent(adapter_path=self.refactor_adapter,
                                            max_input_length=max_in,
                                            max_output_length=max_out)
        self.doc_agent      = DocAgent(adapter_path=self.doc_adapter,
                                       max_input_length=max_in,
                                       max_output_length=256)

    def run(self, source_code: str, mode: str = "both") -> dict:

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{ts}"
        run_dir.mkdir(exist_ok=True)

        result = {"timestamp": ts, "run_dir": str(run_dir),
                  "mode": mode, "success": False}

        # ── BLOCK 2: Parse ────────────────────────────────────────────────
        print("\n[Block 2]  Parsing ...")
        parsed = self.parser.parse(source_code)
        parsed["original_code"] = source_code

        if not parsed.get("parse_success", True):
            logger.error(f"Parse failed: {parsed.get('error')}")
            return result

        print(f"           ✅ {len(parsed['functions'])} method(s)  "
              f"{len(parsed['classes'])} class(es)  "
              f"parser={parsed['parser_used']}")

        (run_dir / "parsed_analysis.json").write_text(
            json.dumps({k: v for k, v in parsed.items() if k != "original_code"},
                       indent=2, default=str), encoding="utf-8")

        # ── BLOCK 3: Prompting Engine ─────────────────────────────────────
        print("\n[Block 3]  Detecting smells + generating prompts ...")
        smells  = self.detector.detect_smells(parsed)
        report  = self.detector.generate_report(parsed)
        plan    = self.engine.generate_refactoring_plan(parsed)
        prompts = self.engine.generate_prompts(source_code, parsed, "both")

        refactor_prompt = prompts.get("refactor_prompt", "")
        doc_prompt      = prompts.get("documentation_prompt", "")

        print(f"           ✅ {len(smells)} smell(s) detected")
        for s in smells:
            icon = "🔴" if s["severity"] == "critical" else \
                   "🟡" if s["severity"] in ("high", "medium") else "🟢"
            print(f"              {icon} [{s['severity']:8}] "
                  f"{s['name']}  —  {s['function']}  "
                  f"({s['metric']}={s['value']}, threshold={s['threshold']})")

        tok = len(refactor_prompt) // 4
        tok_note = "✅ fits" if tok <= 400 else "⚠ tight" if tok <= 480 else "❌ over limit"
        print(f"\n           Refactor prompt : {len(refactor_prompt)} chars  "
              f"~{tok} tokens  {tok_note}")
        print(f"           Doc prompt     : {len(doc_prompt)} chars")

        (run_dir / "smell_report.txt").write_text(report, encoding="utf-8")
        (run_dir / "refactoring_plan.txt").write_text(plan, encoding="utf-8")
        (run_dir / "PROMPT_refactor_agent.txt").write_text(refactor_prompt, encoding="utf-8")
        (run_dir / "PROMPT_doc_agent.txt").write_text(doc_prompt, encoding="utf-8")
        result["smells"] = [{"name": s["name"], "severity": s["severity"],
                              "function": s["function"]} for s in smells]

        # ── BLOCK 5a: Refactor Agent ──────────────────────────────────────
        if mode in ("refactor", "both"):
            print("\n[Block 5a] Refactor Agent  (may take 1-2 min on CPU) ...")
            r_out           = self.refactor_agent.run(refactor_prompt, source_code)
            refactored_code = r_out["refactored_code"]
            r_model_used    = r_out["used_model"]

            if r_model_used:
                print("           ✅ Model produced valid Java")
            else:
                print("           ⚠  Model fallback — original code returned")

            (run_dir / "refactored_code.java").write_text(refactored_code, encoding="utf-8")
            result["refactored_code"]     = refactored_code
            result["refactor_used_model"] = r_model_used

            # ── BLOCK 6a: Refactor Evaluator ──────────────────────────────
            print("\n[Block 6a] Evaluating refactored code ...")
            r_eval = analyze_refactoring(source_code, refactored_code)
            r_conf = r_eval["confidence"]["score"]
            r_stat = r_eval["confidence"]["status"]

            save_evaluation_report(r_eval, source_code, refactored_code,
                                   str(run_dir / "EVALUATION_refactor.txt"))
            with open(run_dir / "EVALUATION_refactor.json", "w", encoding="utf-8") as f:
                json.dump(r_eval, f, indent=2, default=str)

            print(f"           ✅ Confidence: {r_conf:.3f}  {r_stat}")
            result["refactor_evaluation"] = {
                "confidence_score":  r_conf,
                "confidence_status": r_stat,
                "ast_valid":         r_eval["ast_validity"]["valid"],
                "style_score":       r_eval["style_metrics"]["score"],
                "codebleu":          r_eval["codebleu_score"],
                "semantic_score":    r_eval["semantic_preservation"]["score"],
                "improvement_score": r_eval["improvement"]["score"],
                "loc_original":      r_eval["improvement"]["loc_original"],
                "loc_refactored":    r_eval["improvement"]["loc_refactored"],
            }

        # ── BLOCK 5b: Doc Agent ───────────────────────────────────────────
        if mode in ("document", "both"):
            print("\n[Block 5b] Doc Agent ...")
            d_out         = self.doc_agent.run(doc_prompt, parsed)
            documentation = d_out["documentation"]
            d_model_used  = d_out["used_model"]

            if d_model_used:
                print("           ✅ Model generated documentation")
            else:
                print("           ℹ  Template fallback  "
                      "(train Doc Agent on Kaggle for AI descriptions)")

            (run_dir / "documentation.md").write_text(documentation, encoding="utf-8")
            result["documentation"]  = documentation
            result["doc_used_model"] = d_model_used

            # ── BLOCK 6b: Doc Evaluator ────────────────────────────────────
            print("\n[Block 6b] Evaluating documentation ...")
            d_eval = evaluate_documentation(documentation, parsed)
            d_conf = d_eval["confidence"]["score"]
            d_stat = d_eval["confidence"]["status"]

            save_doc_evaluation_report(d_eval, documentation,
                                       str(run_dir / "EVALUATION_doc.txt"))
            with open(run_dir / "EVALUATION_doc.json", "w", encoding="utf-8") as f:
                json.dump(d_eval, f, indent=2, default=str)

            print(f"           ✅ Confidence: {d_conf:.3f}  {d_stat}")
            result["doc_evaluation"] = {
                "confidence_score":   d_conf,
                "confidence_status":  d_stat,
                "coverage_score":     d_eval["coverage"]["score"],
                "completeness_score": d_eval["completeness"]["score"],
            }

        # ── BLOCK 7: Final Result + Save ──────────────────────────────────
        result["success"] = True
        self._save_summary(result, run_dir, parsed)
        self._print_final(result, run_dir)
        return result

    def _save_summary(self, result, run_dir, parsed):
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
            "refactor": result.get("refactor_evaluation"),
            "doc":      result.get("doc_evaluation"),
            "files":    sorted(p.name for p in run_dir.iterdir()),
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
            v   = ("✅ ACCEPTED" if re["confidence_score"] >= 0.75
                   else "⚠  CONDITIONALLY ACCEPTED" if re["confidence_score"] >= 0.65
                   else "❌ REJECTED")
            print(f"\n  🔧 REFACTORING")
            print(f"     Confidence  : {re['confidence_score']:.3f}  {re['confidence_status']}")
            print(f"     AST Valid   : {'✅ YES' if re['ast_valid'] else '❌ NO'}")
            print(f"     Style       : {re['style_score']:.3f}")
            print(f"     CodeBLEU    : {re['codebleu']:.3f}")
            print(f"     Semantic    : {re['semantic_score']:.3f}")
            print(f"     Improvement : {re['improvement_score']:.3f}")
            print(f"     LOC change  : {loc}")
            print(f"     Verdict     : {v}")

        if "doc_evaluation" in result:
            de = result["doc_evaluation"]
            v  = ("✅ ACCEPTED" if de["confidence_score"] >= 0.75
                  else "⚠  CONDITIONALLY ACCEPTED" if de["confidence_score"] >= 0.65
                  else "❌ REJECTED")
            print(f"\n  📄 DOCUMENTATION")
            print(f"     Confidence   : {de['confidence_score']:.3f}  {de['confidence_status']}")
            print(f"     Coverage     : {de['coverage_score']:.3f}")
            print(f"     Completeness : {de['completeness_score']:.3f}")
            print(f"     Verdict      : {v}")

        print(f"\n  📁 Outputs : {run_dir}")
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
        }
        for f in sorted(run_dir.iterdir()):
            desc = descs.get(f.name, "")
            print(f"    {f.name:<40} {desc}")

        print("=" * 60)
