"""
Make v3's output dir from M5's benchmark (docs/V3_PLAN.md §4).

    python -m v3.setup                 # BENCH_OUT -> V3_OUT (default $OUTPUT_PATH/v2_bench -> $OUTPUT_PATH/v3)

V3_OUT gets symbolic links to the benchmark's manifest, frozen classes, audit, context table and images, and to the
v2 runs v3 compares with. Then v2.stack fits the val combiner of the image run with the indication and with all
three text sources, the like-for-like baselines for jev_est_indication and jev_est_text. Nothing in BENCH_OUT is
written, and v2's evaluation reads V3_OUT exactly as it reads the benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

from v2 import stack
from v2.common import read_predictions, setup_logging

from .common import default_bench_dir, default_out_dir, setup_path

logger = logging.getLogger("v3.setup")

REQUIRED = ("manifest.parquet", "measurable_classes.json", "audit.json", "context.parquet")
OPTIONAL = ("context_audit.md", "images")
IMAGE_RUN = "rad-dino_518_ft_attn_s0"   # stage 1: image only, chosen on val in M7a and G7
INDICATION_RUN = "rad-dino_518_ctx_indication"
TEXT_RUN = "rad-dino_518_ctx_text"     # v2's TF-IDF text-only arm with all three sources


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        raise SystemExit(f"{link} exists and is not a link to {target}: remove it or choose another --out")
    link.symlink_to(target)


def setup(bench: Path, out: Path, image_run: str = IMAGE_RUN, indication_run: str = INDICATION_RUN,
          text_run: str = TEXT_RUN, n_boot: int = 1000) -> dict:
    bench, out = Path(bench).resolve(), Path(out).absolute()
    if out.resolve() == bench:
        raise SystemExit("V3_OUT must differ from BENCH_OUT: v3 never writes into the benchmark")
    missing = [name for name in REQUIRED if not (bench / name).exists()]
    if missing:
        raise SystemExit(f"{bench} lacks {', '.join(missing)}: build the benchmark first (v2 `run.sh bench`)")
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED, *OPTIONAL):
        if (bench / name).exists():
            _link(bench / name, out / name)
    for run in (image_run, indication_run, text_run):
        for suffix in (".parquet", ".json"):
            src = bench / "predictions" / f"{run}{suffix}"
            if not src.exists():
                raise SystemExit(f"{src} not found")
            _link(src, out / "predictions" / f"{run}{suffix}")
    arms = {run: read_predictions(out / "predictions" / f"{run}.parquet")[1].get("arm") for run in
            (indication_run, text_run)}
    if arms != {indication_run: "indication", text_run: "text"}:
        raise SystemExit(f"expected text-only arms 'indication' and 'text', got {arms}")
    stacks = stack.main(["--pair", image_run, indication_run, "--pair", image_run, text_run, "--out", str(out),
                         "--n-boot", str(n_boot), "--report", "v3_stack.md"])["runs"]
    record = {"bench": str(bench), "image_run": image_run, "indication_run": indication_run, "text_run": text_run,
              "stack_indication": stacks[0], "stack_text": stacks[1]}
    setup_path(out).write_text(json.dumps(record, indent=2))
    logger.info("v3 output dir ready: %s (baselines %s, %s)", out, *stacks)
    return record


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default=None, help="M5's benchmark dir (default: BENCH_OUT or $OUTPUT_PATH/v2_bench)")
    parser.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
    parser.add_argument("--image-run", default=IMAGE_RUN, help="Stage 1: the image-only run Jev reads")
    parser.add_argument("--indication-run", default=INDICATION_RUN, help="v2's text-only run on the indication")
    parser.add_argument("--text-run", default=TEXT_RUN, help="v2's text-only run on all three sources")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    return setup(Path(args.bench) if args.bench else default_bench_dir(),
                 Path(args.out) if args.out else default_out_dir(),
                 args.image_run, args.indication_run, args.text_run, args.n_boot)


if __name__ == "__main__":
    main()
