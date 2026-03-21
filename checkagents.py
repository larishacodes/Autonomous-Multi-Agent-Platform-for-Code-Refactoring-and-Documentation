"""
check_agents.py
---------------
Shows the from_pretrained calls in your agent files so we can
add use_safetensors=False to stop the hang.

Run from your project root:
    python check_agents.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).parent
candidates = [
    "agents/refactor_agent.py",
    "agents/doc_agent.py",
    "core/refactor_agent.py",
    "core/doc_agent.py",
    "refactor_agent.py",
    "doc_agent.py",
]

for rel in candidates:
    path = ROOT / rel
    if not path.exists():
        continue
    content = path.read_text(encoding="utf-8")
    print(f"\n=== {rel} ===")
    for i, line in enumerate(content.splitlines(), 1):
        if any(k in line for k in [
            "from_pretrained", "AutoModel", "T5For", "use_safetensors",
            "tie_word_embeddings", "adapter_path", "model_path"
        ]):
            print(f"  {i:4}: {line.rstrip()}")