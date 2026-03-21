"""
fix_evaluator_final.py
----------------------
Rewrites core/evaluator.py run() by finding the method boundary
using line-by-line scanning instead of exact string matching.

Run from your project root:
    python fix_evaluator_final.py
"""
from pathlib import Path
import ast, sys

ROOT   = Path(__file__).parent
TARGET = ROOT / "core" / "evaluator.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found"); sys.exit(1)

lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)

# Find the run() method start and its return statement end
run_start = None
run_end   = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("def run(self") and "repo_state" in line:
        run_start = i
    if run_start is not None and stripped == "return new_state, summary":
        run_end = i
        break

if run_start is None or run_end is None:
    print(f"ERROR: could not find run() method (start={run_start} end={run_end})")
    print("File lines 1-30:")
    for i, l in enumerate(lines[:30], 1):
        print(f"  {i:3}: {l}", end="")
    sys.exit(1)

print(f"Found run() at lines {run_start+1}–{run_end+1}")

# Detect indentation from the def line
indent = len(lines[run_start]) - len(lines[run_start].lstrip())
ind = " " * indent  # method-level indent (usually 4 spaces)
body = " " * (indent + 4)  # body indent (usually 8 spaces)

NEW_BODY = f'''\
{ind}def run(self, repo_state):
{body}"""
{body}Evaluate all completed results in repo_state.
{body}Returns (new_state, summary) with evaluation_scores populated.
{body}"""
{body}scores = dict(repo_state.evaluation_scores)
{body}summary = EvaluationSummary()

{body}# Evaluate refactor results
{body}if repo_state.refactor_results:
{body}    refactor_verdict = self._evaluate_refactor(repo_state, scores)
{body}    summary.refactor = refactor_verdict
{body}    raw = refactor_verdict.raw
{body}    scores["refactor/confidence"]     = refactor_verdict.confidence
{body}    scores["refactor/ast_valid"]      = float(raw.get("ast_validity", {{}}).get("valid", False))
{body}    scores["refactor/style"]          = raw.get("style_metrics", {{}}).get("score", 0.0)
{body}    scores["refactor/codebleu"]       = raw.get("codebleu_score", 0.0)
{body}    scores["refactor/semantic"]       = raw.get("semantic_preservation", {{}}).get("score", 0.0)
{body}    scores["refactor/improvement"]    = raw.get("improvement", {{}}).get("score", 0.0)
{body}    scores["refactor/loc_original"]   = raw.get("improvement", {{}}).get("loc_original", 0)
{body}    scores["refactor/loc_refactored"] = raw.get("improvement", {{}}).get("loc_refactored", 0)
{body}    logger.info("Refactor evaluation: confidence=%.3f", refactor_verdict.confidence)

{body}# Evaluate documentation results
{body}if repo_state.documentation_results:
{body}    doc_verdict = self._evaluate_doc(repo_state, scores)
{body}    summary.doc = doc_verdict
{body}    raw = doc_verdict.raw
{body}    scores["doc/confidence"]   = doc_verdict.confidence
{body}    scores["doc/coverage"]     = raw.get("coverage", {{}}).get("score", 0.0)
{body}    scores["doc/completeness"] = raw.get("completeness", {{}}).get("score", 0.0)
{body}    logger.info("Doc evaluation: confidence=%.3f", doc_verdict.confidence)

{body}if summary.any_needs_human:
{body}    logger.warning("One or more results in the borderline band — consider human review.")

{body}parts = []
{body}if summary.refactor:
{body}    parts.append(f"refactor_conf={{summary.refactor.confidence:.3f}}")
{body}if summary.doc:
{body}    parts.append(f"doc_conf={{summary.doc.confidence:.3f}}")

{body}new_state = repo_state.evolve(
{body}    agent_id=AGENT_ID,
{body}    action="evaluation_complete",
{body}    summary=" ".join(parts) or "partial evaluation",
{body}    evaluation_scores=scores,
{body})
{body}return new_state, summary
'''

# Replace lines from run_start to run_end (inclusive)
new_lines = lines[:run_start] + [NEW_BODY] + lines[run_end + 1:]
new_content = "".join(new_lines)

try:
    ast.parse(new_content)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR at line {e.lineno}: {e.msg}")
    ctx = new_content.splitlines()
    for i, l in enumerate(ctx[max(0,e.lineno-4):e.lineno+2], max(1,e.lineno-3)):
        print(f"  {i:4}: {l}")
    sys.exit(1)

TARGET.write_text(new_content, encoding="utf-8")
print(f"SAVED {TARGET}")
print("\nscores lines now in run():")
for line in new_content.splitlines():
    if 'scores["' in line:
        print(" ", line.strip())