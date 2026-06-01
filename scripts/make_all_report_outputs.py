#!/usr/bin/env python
"""Regenerate all report figures and table snippets from data/results/*.csv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"Running {script}")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)


def main() -> None:
    run("plot_report_single_column.py")
    run("plot_feedback_additions.py")


if __name__ == "__main__":
    main()
