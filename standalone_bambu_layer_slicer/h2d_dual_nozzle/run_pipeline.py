#!/usr/bin/env python3
"""Run the shared H2D pipeline with this folder's right-nozzle configuration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SHARED_PIPELINE = HERE.parent / "common" / "run_pipeline.py"


def main() -> int:
    command = [sys.executable, str(SHARED_PIPELINE), *sys.argv[1:]]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
