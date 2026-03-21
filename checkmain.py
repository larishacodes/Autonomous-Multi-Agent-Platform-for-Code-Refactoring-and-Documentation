"""
check_main.py
-------------
Shows your main.py's argument parser so we know the exact flags to use.

Run from your project root:
    python check_main.py
"""
from pathlib import Path
import re

content = (Path(__file__).parent / "main.py").read_text(encoding="utf-8")

# Show the argparse section
match = re.search(r'(add_argument.*?)(?=def |\Z)', content, re.DOTALL)
# Just show lines with add_argument or ArgumentParser
for line in content.splitlines():
    stripped = line.strip()
    if any(k in stripped for k in [
        'ArgumentParser', 'add_argument', 'parse_args',
        '--file', '--mode', '--ask', '--query', 'positional',
        'source', 'input'
    ]):
        print(line)