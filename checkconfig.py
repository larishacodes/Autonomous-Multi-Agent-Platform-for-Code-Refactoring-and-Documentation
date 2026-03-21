"""
check_config.py
---------------
Shows your current config.json and fixes the two issues:
1. DACOS path pointing to wrong user
2. Adds use_safetensors=False hint

Run from your project root:
    python check_config.py
"""
from pathlib import Path
import json, sys

ROOT   = Path(__file__).parent
CONFIG = ROOT / "config.json"

if not CONFIG.exists():
    print("ERROR: config.json not found")
    sys.exit(1)

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
print("=== Current config.json ===")
print(json.dumps(cfg, indent=2))

print("\n=== Fixes to apply ===")

# Fix 1: DACOS path
dacos_path = cfg.get("dacos", {}).get("path", "")
if dacos_path and "Administrator" in dacos_path:
    print(f"  DACOS path wrong user: {dacos_path}")
    print("  → Setting dacos.path to empty string (disables DACOS, stops warning)")
    cfg.setdefault("dacos", {})["path"] = ""

# Fix 2: add use_safetensors flag
if "models" in cfg:
    if "use_safetensors" not in cfg["models"]:
        cfg["models"]["use_safetensors"] = False
        print("  → Added models.use_safetensors = false")

CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("\n=== Updated config.json ===")
print(json.dumps(cfg, indent=2))
print("\nDone. Run: python main.py --file tests/fixtures/Sample.java --mode both")