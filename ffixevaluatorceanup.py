"""
fix_evaluator_cleanup.py
------------------------
Removes the scores["..."] mutation lines from _evaluate_refactor
and _evaluate_doc in core/evaluator.py since run() now owns them.

Run from your project root:
    python fix_evaluator_cleanup.py
"""
from pathlib import Path
import ast, sys

ROOT   = Path(__file__).parent
TARGET = ROOT / "core" / "evaluator.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found"); sys.exit(1)

lines  = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)
output = []
skip   = False

# Keys written by the sub-methods that run() now owns
OWNED_KEYS = {
    '"refactor/confidence"', '"refactor/ast_valid"', '"refactor/style"',
    '"refactor/codebleu"',   '"refactor/semantic"',  '"refactor/improvement"',
    '"refactor/loc_original"', '"refactor/loc_refactored"',
    '"doc/confidence"', '"doc/coverage"', '"doc/completeness"',
}

# We only remove these lines when we are INSIDE _evaluate_refactor or _evaluate_doc
# (not inside run(), which we want to keep)
inside_sub_method = False
inside_run        = False

for line in lines:
    stripped = line.strip()

    # Track which method we are in
    if stripped.startswith("def run(self"):
        inside_run        = True
        inside_sub_method = False
    elif stripped.startswith("def _evaluate_refactor(") or stripped.startswith("def _evaluate_doc("):
        inside_sub_method = True
        inside_run        = False
    elif stripped.startswith("def ") and inside_sub_method:
        # Entered a new method — leave sub-method scope
        inside_sub_method = False

    # Drop scores mutation lines that are inside sub-methods
    if inside_sub_method and not inside_run:
        # Check if this line is a scores["..."] = ... assignment
        is_scores_line = (
            stripped.startswith("scores[") and
            "=" in stripped and
            any(k in stripped for k in OWNED_KEYS)
        )
        if is_scores_line:
            # Skip this line
            continue

    output.append(line)

new_content = "".join(output)

try:
    ast.parse(new_content)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR at line {e.lineno}: {e.msg}")
    sys.exit(1)

TARGET.write_text(new_content, encoding="utf-8")
print(f"SAVED {TARGET}")

# Count scores lines remaining — should be exactly 11 (8 refactor + 3 doc)
score_lines = [l.strip() for l in new_content.splitlines() if 'scores["' in l and "=" in l]
print(f"\n{len(score_lines)} scores lines remaining (expected 11):")
for l in score_lines:
    print(" ", l)