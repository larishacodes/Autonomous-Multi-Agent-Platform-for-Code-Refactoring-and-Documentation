"""
fix_task_models_v2.py
---------------------
Fixes core/task_models.py properly:

1. Moves  from __future__ import annotations  back to line 1
2. Places the sys.path injection after it (not before it)
3. Changes  from core.state import  to  from state import
4. Removes any duplicate/broken injection blocks

Run from your project root:
    python fix_task_models_v2.py
"""

from pathlib import Path
import re
import sys

ROOT   = Path(__file__).parent
TARGET = ROOT / "core" / "task_models.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found"); sys.exit(1)

content = TARGET.read_text(encoding="utf-8")

# ── Step 1: strip ALL existing sys.path injection blocks ─────────────────────
# They may appear multiple times and in the wrong place
INJECTION_PATTERN = re.compile(
    r"#.*sys\.path fix.*\n"
    r"import sys as _sys.*?\n"
    r"_root = .*?\n"
    r"if _root not in _sys\.path:\n"
    r"    _sys\.path\.insert\(0, _root\)\n"
    r"\n?",
    re.DOTALL
)
content = INJECTION_PATTERN.sub("", content)

# Also strip the two-line variant without the comment header
INJECTION_PATTERN2 = re.compile(
    r"import sys as _sys,? .*?\n"
    r"_root = _os\.path\.dirname.*?\n"
    r"if _root not in _sys\.path:\n"
    r"    _sys\.path\.insert\(0, _root\)\n"
    r"\n?",
    re.DOTALL
)
content = INJECTION_PATTERN2.sub("", content)

# ── Step 2: fix the import — from core.state → from state ────────────────────
content = content.replace("from core.state import", "from state import")

# ── Step 3: ensure from __future__ is at the very top ───────────────────────
# Remove it from wherever it is now
future_line = "from __future__ import annotations\n"
content = content.replace(future_line, "")
content = content.lstrip("\n")

# ── Step 4: build the correct file ───────────────────────────────────────────
INJECTION = (
    "import sys as _sys, os as _os\n"
    "_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))\n"
    "if _root not in _sys.path:\n"
    "    _sys.path.insert(0, _root)\n"
    "\n"
)

new_content = future_line + "\n" + INJECTION + content

TARGET.write_text(new_content, encoding="utf-8")
print("SAVED. New file top:\n")
print(new_content[:500])

# ── Syntax check ─────────────────────────────────────────────────────────────
import ast
try:
    ast.parse(new_content)
    print("\nSYNTAX OK")
except SyntaxError as e:
    print(f"\nSYNTAX ERROR at line {e.lineno}: {e.msg}")