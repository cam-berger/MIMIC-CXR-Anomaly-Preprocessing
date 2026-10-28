"""
v3.2 (exploratory, docs/V3_PLAN.md §8b): Jev's answers as inputs to v2's val combiner. No Jev requests.

    python -m v3.combine             # -> V3_OUT/reports/v3_combine.md (and v3_combine_stack.md from v2.stack)

Each Jev run used here is rewritten as log-odds, clipped at CLIP because Jev's probabilities come rounded to 0.01
and can be 0 or 1 (predictions/{run}_logodds.parquet; monotone, so its AUROC is unchanged). v2.stack then fits
each of §8b's pairs on val, and every stacked run is compared, paired on the official test, with what it is read
against. The Jev runs are those of the format val chose in v3.0.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from v2 import stack
from v2.common import LABELS, TEST, read_predictions, setup_logging, write_predictions
from v2.evaluate import _get, compare_runs, evaluate_run, fmt_ci

from . import prompt
from .common import default_out_dir, load_setup
from .report import choose_format

logger = logging.getLogger("v3.combine")

CLIP = 0.005
JEV_ARMS = ("jev_text", "jev_img", "jev_img_est_text")  # §8b's Jev inputs


def to_logodds(out_dir: Path, run: str) -> str:
    """predictions/{run}_logodds: the Jev run's probabilities as clipped log-odds, tagged for v2.stack's run names."""
    wide, meta = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")
    p = np.clip(wide[list(LABELS)].to_numpy(np.float64), CLIP, 1 - CLIP)
    name = f"{run}_logodds"
    write_predictions(out_dir, name, wide["study_id"], wide["split"], np.log(p / (1 - p)), {
        "kind": "jev_logodds", "milestone": "v3.2", "source_run": run, "stack_tag": run, "clip": CLIP,
        "arm": meta.get("arm"), "format": meta.get("format"), "model": meta.get("model"),
    })
    return name


def pairs(setup: dict, fmt: str) -> list[dict]:
    """§8b's stacks A-E: the two inputs, and the run each is read against."""
    img, stack_txt = setup["image_run"], setup["stack_text"]
    j = {arm: prompt.run_name(arm, fmt) for arm in JEV_ARMS}
    spec = [("A", img, j["jev_text"], [stack_txt, img]),
            ("B", img, j["jev_img"], [img]),
            ("C", stack_txt, j["jev_text"], [stack_txt]),
            ("D", stack_txt, j["jev_img"], [stack_txt]),
            ("E", stack_txt, j["jev_img_est_text"], [stack_txt])]
    return [{"label": label, "base": base, "jev": jev, "against": against} for label, base, jev, against in spec]


def build(out_dir: Path, n_boot: int = 1000) -> dict:
    setup = load_setup(out_dir)
    choice = choose_format(out_dir)
    if choice is None:
        raise SystemExit(f"val hasn't chosen the format yet: run {prompt.PRIMARY} on val in both formats (§7)")
    spec = pairs(setup, choice["chosen"])
    for p in spec:
        path = Path(out_dir) / "predictions" / f"{p['jev']}.parquet"
        if not path.exists():
            raise SystemExit(f"{p['jev']} not found: run it on val and the official test first")
        wide, _ = read_predictions(path)
        if not (wide["split"] == TEST).any():
            raise SystemExit(f"{p['jev']} has no official-test answers yet")
    logodds = {run: to_logodds(out_dir, run) for run in sorted({p["jev"] for p in spec})}
    args = [a for p in spec for a in ("--pair", p["base"], logodds[p["jev"]])]
    runs = stack.main([*args, "--out", str(out_dir), "--n-boot", str(n_boot), "--report", "v3_combine_stack.md"])["runs"]
    rows = []
    for p, run in zip(spec, runs):
        rep = evaluate_run(out_dir, run, n_boot)
        cmps = [{"against": b, "v2": compare_runs(out_dir, run, b, "test_clean", "v2", n_boot),
                 "u0": compare_runs(out_dir, run, b, "test_clean", "u0", n_boot)} for b in p["against"]]
        rows.append({**p, "run": run, "val": _get(rep, "val", "v2"), "test": _get(rep, "test_clean", "v2"),
                     "cmps": cmps})
    return {"format": choice["chosen"], "rows": rows}


def write(out_dir: Path, result: dict) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    lines = ["# v3.2 (exploratory): Jev's answers as inputs to v2's val combiner", "",
             f"Jev runs in the `{result['format']}` format, as log-odds clipped at {CLIP}. Per class, v2.stack's "
             "logistic regression on two runs' scores, fit on val (val's own scores cross-fitted). Official test, "
             "macro AUROC over the frozen measurable classes, paired patient bootstrap (docs/V3_PLAN.md §8b). "
             "Stacks C-E feed the combiner's output into a second combiner, so their val scores are slightly "
             "optimistic; no test study was used for any fit.", "",
             "| Stack | Inputs | Val | Official test | Read against | A minus B | A minus B (u0) |",
             "|---|---|---|---|---|---|---|"]
    for row in result["rows"]:
        for i, c in enumerate(row["cmps"]):
            head = (f"| {row['label']} | `{row['base']}` + `{row['jev']}` | {auc(row['val'])} | {auc(row['test'])} "
                    if i == 0 else "| | | | ")
            lines.append(f"{head}| `{c['against']}` | {fmt_ci(c['v2']['diff'], c['v2']['diff_ci'])} | "
                         f"{fmt_ci(c['u0']['diff'], c['u0']['diff_ci'])} |")
    lines += ["", "An interval above 0 says Jev's reading carries information the combiner can use. Exploratory: "
              "a lead to confirm on a second seed or another test set, not a result to rely on."]
    path = Path(out_dir) / "reports" / "v3_combine.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    result = build(out_dir, args.n_boot)
    path = write(out_dir, result)
    for row in result["rows"]:
        c = row["cmps"][0]
        logger.info("%s %s - %s: %s", row["label"], row["run"], c["against"], fmt_ci(c["v2"]["diff"], c["v2"]["diff_ci"]))
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
