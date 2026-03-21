"""
fix_imports.py  (v2)
--------------------
Run from your project root:

    python fix_imports.py

What it does
------------
Every file inside core/ that does  `from state import ...`  will fail
with ModuleNotFoundError because Python looks for state.py relative to
core/, not the project root.

This script inserts a sys.path root-injection block at the top of each
affected file so the import resolves correctly regardless of where Python
is invoked from.

Safe to run multiple times — skips files that already have the injection.
"""

from pathlib import Path

ROOT = Path(__file__).parent

PATH_BLOCK = (
    "import sys as _sys, os as _os\n"
    "_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))\n"
    "if _root not in _sys.path:\n"
    "    _sys.path.insert(0, _root)\n"
    "\n"
)

CORE_FILES = [
    "core/task_models.py",
    "core/planner_agent.py",
    "core/supervisor.py",
    "core/evaluator.py",
    "core/evaluator_agent.py",
    "core/document_builder.py",
    "core/hybrid_retriever.py",
    "core/rag_query.py",
    "core/config_validator.py",
]


def inject_path_block(path):
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    if "_root = _os.path.dirname" in content:
        print(f"  SKIP (already patched)  {path.name}")
        return
    if "from state import" not in content and "import state" not in content:
        print(f"  SKIP (no state import)  {path.name}")
        return

    lines = content.splitlines(keepends=True)
    insert_at = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                    continue
                in_docstring = True
                continue
        else:
            if docstring_char in stripped:
                in_docstring = False
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("from __future__"):
            insert_at = i + 1
            continue
        insert_at = i
        break

    lines.insert(insert_at, PATH_BLOCK)
    path.write_text("".join(lines), encoding="utf-8")
    print(f"  PATCHED  {path.name}")


print("\n── Injecting sys.path fix into core/ files ────────────────────")
for rel in CORE_FILES:
    inject_path_block(ROOT / rel)

print("\n── Fixing remaining repostate references ───────────────────────")
for rel in CORE_FILES + ["tests/testpipeline.py", "tests/testplanner.py"]:
    path = ROOT / rel
    if not path.exists():
        continue
    content = path.read_text(encoding="utf-8")
    if "repostate" in content:
        fixed = content.replace("from repostate import", "from state import")
        fixed = fixed.replace("import repostate", "import state")
        path.write_text(fixed, encoding="utf-8")
        print(f"  FIXED  {path.name}")

print("\nDone.")
print("Run: python -m pytest tests/testpipeline.py -v -k 'not integration'")