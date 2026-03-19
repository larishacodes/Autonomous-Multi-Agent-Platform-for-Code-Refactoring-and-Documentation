#!/usr/bin/env python3
"""
main.py — Entry point for the Autonomous Multi-Agent Platform.

Usage:
    python main.py                                     # SimpleTest.java, mode=both
    python main.py --file OrderProcessor.java          # custom input
    python main.py --file SimpleTest.java --mode refactor
    python main.py --file SimpleTest.java --mode document
    python main.py --file SimpleTest.java --mode both
"""

# ── Silence all third-party noise BEFORE any imports ─────────────────────────
import warnings
warnings.filterwarnings("ignore")          # PEFT missing keys, transformers tied weights

import logging
# Only show ERROR from third-party libraries
for _lib in ["httpx", "httpcore", "urllib3", "huggingface_hub",
             "transformers", "peft", "torch", "safetensors",
             "transformers.safetensors_conversion",
             "huggingface_hub.utils._http"]:
    logging.getLogger(_lib).setLevel(logging.ERROR)

# Our own logger — WARNING level (only problems, not routine steps)
logging.basicConfig(
    level=logging.WARNING,
    format="  ⚠  %(message)s",
)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Agent Platform — Java"
    )
    parser.add_argument("--file", "-f", default="SimpleTest.java",
                        help="Java file to process (default: SimpleTest.java)")
    parser.add_argument("--mode", "-m",
                        choices=["refactor", "document", "both"],
                        default="both",
                        help="Pipeline mode (default: both)")
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("=" * 60)
    print("  Autonomous Multi-Agent Platform")
    print("  Java Code Refactoring + Documentation")
    print("=" * 60)
    print(f"  File : {args.file}")
    print(f"  Mode : {args.mode}")

    input_path = ROOT / args.file
    if not input_path.exists():
        print(f"\n  ❌ File not found: {args.file}")
        print("     Put your Java file in the project root.")
        sys.exit(1)

    source_code = input_path.read_text(encoding="utf-8")
    if not source_code.strip():
        print(f"\n  ❌ File is empty: {args.file}")
        sys.exit(1)

    print(f"  Lines: {len(source_code.splitlines())}")

    from pipeline import Pipeline
    pipeline = Pipeline()
    result   = pipeline.run(source_code, mode=args.mode)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
