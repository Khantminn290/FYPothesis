"""Deployment-only entrypoint for the public, read-only showcase."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


# This assignment is deliberate rather than setdefault: even a mistakenly
# configured hosting secret cannot enable the paid subprocess launch path.
os.environ["FYPOTHESIS_ENABLE_REAL_RUNS"] = "0"

ROOT = Path(__file__).resolve().parent.parent
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
