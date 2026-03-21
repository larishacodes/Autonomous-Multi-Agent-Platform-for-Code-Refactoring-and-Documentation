"""
fix_core_state_imports.py
-------------------------
Fixes ALL files in core/ that import from core.state (wrong) instead of
from state (correct), and ensures the sys.path injection is present and
correctly placed after any __future__ import.

Run from your project root:
    python fix_core_state_imports.py
"""

from pathlib import Path
import ast
import re
import sys

ROOT = Path(__file__).parent

INJECTION = (
    "import sys as _sys, os as _os\n"
    "_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))\n"
    "if _root not in _sys.path:\n"
    "    _sys.path.insert(0, _root)\n"
    "\n"
)

CORE_FILES = sorted((ROOT / "core").glob("*.py"))


def fix_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    original = content

    # Fix 1: from core.state import → from state import
    content = content.replace("from core.state import", "from state import")

    # Fix 2: strip ALL existing injection blocks (any variant)
    patterns = [
        re.compile(
            r"#[^\n]*sys\.path[^\n]*\n"
            r"import sys as _sys[^\n]*\n"
            r"[^\n]*_os[^\n]*\n"
            r"[^\n]*_root[^\n]*\n"
            r"if _root[^\n]*\n"
            r"    _sys\.path[^\n]*\n\n?",
        ),
        re.compile(
            r"import sys as _sys[^\n]*\n"
            r"[^\n]*_os[^\n]*\n"
            r"_root[^\n]*\n"
            r"if _root[^\n]*\n"
            r"    _sys\.path[^\n]*\n\n?",
        ),
        re.compile(
            r"import sys as _sys, os as _os\n"
            r"_root = _os\.path\.dirname\(_os\.path\.dirname\(_os\.path\.abspath\(__file__\)\)\)\n"
            r"if _root not in _sys\.path:\n"
            r"    _sys\.path\.insert\(0, _root\)\n\n?",
        ),
    ]
    for pat in patterns:
        content = pat.sub("", content)

    # Fix 3: ensure __future__ is first, injection comes right after
    future_line = "from __future__ import annotations\n"
    has_future = future_line in content
    if has_future:
        content = content.replace(future_line, "")

    content = content.lstrip("\n")

    # Only inject if file imports from state
    needs_injection = "from state import" in content or "import state" in content

    if has_future:
        new_content = future_line + "\n" + (INJECTION if needs_injection else "") + content
    else:
        new_content = (INJECTION if needs_injection else "") + content

    # Fix 4: remove any blank lines that got doubled up
    while "\n\n\n" in new_content:
        new_content = new_content.replace("\n\n\n", "\n\n")

    if new_content == original:
        # Check syntax anyway
        try:
            ast.parse(new_content)
            print(f"  OK (no changes)       {path.name}")
        except SyntaxError as e:
            print(f"  SYNTAX ERROR (unchanged) {path.name}: line {e.lineno} {e.msg}")
        return

    # Syntax check before saving
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR — NOT SAVED  {path.name}: line {e.lineno} {e.msg}")
        print(f"  Content start: {repr(new_content[:300])}")
        return

    path.write_text(new_content, encoding="utf-8")
    changes = []
    if "from core.state import" in original:
        changes.append("fixed core.state→state")
    if needs_injection and INJECTION not in original:
        changes.append("injected sys.path")
    if has_future and not original.startswith(future_line):
        changes.append("moved __future__ to top")
    print(f"  FIXED [{', '.join(changes) or 'cleaned up'}]  {path.name}")


print("\n── Fixing core/ files ──────────────────────────────────────────")
for f in CORE_FILES:
    if f.name.startswith("__"):
        continue
    fix_file(f)

print("\n── Syntax check all core/ files ────────────────────────────────")
all_ok = True
for f in CORE_FILES:
    if f.name.startswith("__"):
        continue
    try:
        ast.parse(f.read_text(encoding="utf-8"))
        print(f"  OK  {f.name}")
    except SyntaxError as e:
        print(f"  FAIL  {f.name}  line {e.lineno}: {e.msg}")
        all_ok = False

print("\n" + ("ALL SYNTAX OK" if all_ok else "SYNTAX ERRORS — check above"))
print("\nRun: python -m pytest tests/testpipeline.py -v -k 'not integration'")