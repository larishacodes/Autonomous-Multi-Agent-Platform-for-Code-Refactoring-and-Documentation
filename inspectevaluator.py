"""
inspect_evaluator.py
--------------------
Shows the run() method of your local evaluator so we can see
exactly how scores are written.

Run from your project root:
    python inspect_evaluator.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).parent

# Find the evaluator file — could be core/evaluator.py or core/evaluator_agent.py
candidates = [
    ROOT / "core" / "evaluator.py",
    ROOT / "core" / "evaluator_agent.py",
]

for path in candidates:
    if path.exists():
        content = path.read_text(encoding="utf-8")
        print(f"\n=== {path} ===")
        # Show the run() method
        match = re.search(r'def run\(self.*?(?=\n    def |\Z)', content, re.DOTALL)
        if match:
            print(match.group(0)[:3000])
        else:
            print("run() not found — showing first 2000 chars:")
            print(content[:2000])