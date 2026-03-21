"""
fix_main_guard.py
-----------------
Two fixes:
1. Adds TRANSFORMERS_OFFLINE=1 env var at the top of main.py so the
   safetensors background thread never tries to contact Hugging Face Hub.
2. Ensures if __name__ == "__main__": guard exists to stop double-run.

Run from your project root:
    python fix_main_guard.py
"""
from pathlib import Path
import ast, sys

ROOT   = Path(__file__).parent
TARGET = ROOT / "main.py"

if not TARGET.exists():
    print(f"ERROR: main.py not found"); sys.exit(1)

content = TARGET.read_text(encoding="utf-8")
original = content

# ── Fix 1: add offline env var at very top (after any __future__ import) ─────
OFFLINE_BLOCK = (
    "import os as _os\n"
    "# Prevent transformers from contacting HuggingFace Hub during model load\n"
    "_os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')\n"
    "_os.environ.setdefault('HF_HUB_OFFLINE', '1')\n"
    "\n"
)

if "TRANSFORMERS_OFFLINE" not in content:
    lines = content.splitlines(keepends=True)
    # Insert after __future__ line if present, else at top
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("from __future__"):
            insert_at = i + 1
            break
    lines.insert(insert_at, OFFLINE_BLOCK)
    content = "".join(lines)
    print("FIXED: added TRANSFORMERS_OFFLINE=1 env var")
else:
    print("OK: TRANSFORMERS_OFFLINE already set")

# ── Fix 2: ensure if __name__ == "__main__": guard ────────────────────────────
lines = content.splitlines()
last_lines = "\n".join(lines[-10:])

if '__name__ == "__main__"' not in last_lines and "__name__ == '__main__'" not in last_lines:
    # Find bare main() call at module level and wrap it
    new_lines = []
    found = False
    for line in lines:
        stripped = line.strip()
        # Detect a bare top-level call to main() or the entry block
        if not found and stripped in ("main()", "sys.exit(main())"):
            new_lines.append('\nif __name__ == "__main__":\n')
            new_lines.append("    " + line.lstrip() + "\n")
            found = True
        else:
            new_lines.append(line + "\n" if not line.endswith("\n") else line)
    if found:
        content = "".join(new_lines)
        print("FIXED: wrapped main() in if __name__ guard")
    else:
        # Append the guard at the end
        content = content.rstrip() + '\n\nif __name__ == "__main__":\n    main()\n'
        print("FIXED: appended if __name__ guard at end of file")
else:
    print("OK: if __name__ guard already present")

# ── Syntax check ──────────────────────────────────────────────────────────────
try:
    ast.parse(content)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR at line {e.lineno}: {e.msg}")
    sys.exit(1)

TARGET.write_text(content, encoding="utf-8")
print(f"SAVED {TARGET}")
print("\nRun: python main.py --file tests/fixtures/Sample.java --mode both")