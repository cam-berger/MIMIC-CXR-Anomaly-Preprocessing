"""Paths and the setup record shared by v3's commands (docs/V3_PLAN.md)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from v2.common import load_dotenv


def default_out_dir() -> Path:
    """`V3_OUT`, else `$OUTPUT_PATH/v3`, else ./output/v3."""
    load_dotenv()
    if os.environ.get("V3_OUT"):
        return Path(os.environ["V3_OUT"])
    return Path(os.environ.get("OUTPUT_PATH", "output")) / "v3"


def default_bench_dir() -> Path:
    """M5's benchmark: `BENCH_OUT`, else `$OUTPUT_PATH/v2_bench` (as v2/scripts/run.sh)."""
    load_dotenv()
    if os.environ.get("BENCH_OUT"):
        return Path(os.environ["BENCH_OUT"])
    return Path(os.environ.get("OUTPUT_PATH", "output")) / "v2_bench"


def setup_path(out_dir: Path) -> Path:
    return Path(out_dir) / "v3_setup.json"


def load_setup(out_dir: Path) -> dict:
    path = setup_path(out_dir)
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m v3.setup` first.")
    return json.loads(path.read_text())
