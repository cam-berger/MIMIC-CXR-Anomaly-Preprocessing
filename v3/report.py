"""
v3's report (docs/V3_PLAN.md §5-7): every Jev arm and v2 baseline on val and the official test, the context ladder,
gate J1 and the red flags, all computed by v2.evaluate.

    python -m v3.report              # -> V3_OUT/reports/v3_report.md (and reports/{run}.md per run)

Arms or test predictions not run yet show as "not run"; the val prompt rule (§7 step 4) is read from val alone.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from v2.common import LABELS, TEST, read_predictions, setup_logging
from v2.evaluate import _get, compare_runs, evaluate_run, fmt, fmt_ci

from . import prompt
from .common import default_out_dir, load_setup

logger = logging.getLogger("v3.report")

LOST_ESTIMATES = 0.02   # §6: jev_est this far below the image run means Jev drops the image's information
VAL_REVISE = 0.03       # §7 step 4: jev_est_text this far below the val stack on val means revise the prompt
LEAK_CLASS = 0.90       # §6: a text-only class above this on the official test
COARSE = 20             # §6: fewer distinct scores per finding than this (median of 12) is reported as coarse


def _has(out_dir: Path, run: str) -> bool:
    return (Path(out_dir) / "predictions" / f"{run}.parquet").exists()


def _cmp(out_dir: Path, a: str, b: str, test_set: str, metric: str, n_boot: int) -> Optional[dict]:
    if not (_has(out_dir, a) and _has(out_dir, b)):
        return None
    try:
        return compare_runs(out_dir, a, b, test_set, metric, n_boot)
    except SystemExit as exc:  # e.g. no shared test studies yet
        logger.info("%s - %s on %s: %s", a, b, test_set, exc)
        return None


def _diff(c: Optional[dict]) -> str:
    return fmt_ci(c["diff"], c["diff_ci"]) if c else "not run"


def _auc(res: Optional[dict]) -> str:
    return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "not run"


def distinct_scores(out_dir: Path, run: str) -> Optional[float]:
    """Median over the 12 findings of the number of distinct official-test scores (all studies if no test yet)."""
    wide, _ = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")
    test = wide[wide["split"] == TEST]
    rows = test if len(test) else wide
    return float(np.median([rows[label].nunique() for label in LABELS])) if len(rows) else None


def build(out_dir: Path, n_boot: int = 1000) -> dict:
    s = load_setup(out_dir)
    img, stack_ind, stack_txt, txt = s["image_run"], s["stack_indication"], s["stack_text"], s["text_run"]
    roles = {
        "jev_est": "estimates", "jev_est_indication": "+ indication", "jev_est_text": "+ complaint, prior report (J1)",
        "jev_text": "text only (control)", "jev_img_est_text": "+ the image (v3.1)",
        img: "stage 1, image only", stack_ind: "val combiner, + indication", stack_txt: "val combiner, + all text (J1)",
        txt: "TF-IDF, text only",
    }
    runs = [r for r in roles if _has(out_dir, r)]
    reports = {r: evaluate_run(out_dir, r, n_boot) for r in runs}
    table = []
    for r in roles:
        rep = reports.get(r)
        meta = rep["meta"] if rep else {}
        table.append({
            "run": r, "role": roles[r], "kind": meta.get("kind"),
            "val": _get(rep, "val", "v2") if rep else None,
            "test": _get(rep, "test_clean", "v2") if rep else None,
            "test_u0": _get(rep, "test_clean", "u0") if rep else None,
            "answered": meta.get("studies_answered"), "expected": meta.get("studies_expected"),
            "tokens": meta.get("input_tokens"), "model": meta.get("model"), "prompt_hash": meta.get("prompt_hash"),
            "distinct": distinct_scores(out_dir, r) if rep and meta.get("kind") == "jev" else None,
        })
    ladder_spec = [
        ("Jev keeps the estimates", "jev_est", img),
        ("+ indication", "jev_est_indication", "jev_est"),
        ("+ complaint and prior report", "jev_est_text", "jev_est_indication"),
        ("All the text", "jev_est_text", "jev_est"),
        ("Jev against TF-IDF, text alone", "jev_text", txt),
        ("Like for like with MMBT's inputs", "jev_est_indication", stack_ind),
        ("J1: Jev against the val combiner", "jev_est_text", stack_txt),
    ]
    if _has(out_dir, "jev_img_est_text"):
        ladder_spec.insert(4, ("+ the image (v3.1)", "jev_img_est_text", "jev_est_text"))
    ladder = [{"step": step, "a": a, "b": b,
               "v2": _cmp(out_dir, a, b, "test_clean", "v2", n_boot),
               "u0": _cmp(out_dir, a, b, "test_clean", "u0", n_boot)} for step, a, b in ladder_spec]
    j1 = next(row["v2"] for row in ladder if row["a"] == prompt.PRIMARY and row["b"] == stack_txt)
    val_rule = _cmp(out_dir, prompt.PRIMARY, stack_txt, "val", "v2", n_boot)
    return {"setup": s, "table": table, "ladder": ladder, "j1": j1, "verdict": j1_verdict(j1),
            "val_rule": val_rule, "flags": red_flags(table, ladder, img)}


def j1_verdict(c: Optional[dict]) -> str:
    if c is None:
        return "not run: needs jev_est_text and the val combiner on the official test"
    if not np.isfinite(c["diff"]):
        return "n/a: no measurable classes"
    if c["a_better"]:
        return "PASS: Jev combines the estimates and the text better than the logistic regression"
    if c["b_better"]:
        return "FAIL: Jev loses information the logistic regression keeps"
    return "no detectable gain over the logistic regression (the CI includes 0)"


def red_flags(table: list, ladder: list, image_run: str) -> list[str]:
    by_run = {row["run"]: row for row in table}
    flags = []
    jt, im = by_run.get("jev_text", {}).get("test"), by_run.get(image_run, {}).get("test")
    if jt and im and jt["macro_auroc"] > im["macro_auroc"]:
        flags.append(f"LEAKAGE: `jev_text` ({fmt(jt['macro_auroc'])}) beats the image run ({fmt(im['macro_auroc'])}) "
                     "on the official test: stop and inspect locally (§6)")
    if jt:
        high = [c["label"] for c in jt["per_class"] if c["measurable"] and c["auroc"] > LEAK_CLASS]
        if high:
            flags.append(f"LEAKAGE: `jev_text` above {LEAK_CLASS} on {', '.join(high)}: stop and inspect locally (§6)")
    for row in table:
        if row["kind"] != "jev" or not row["expected"]:
            continue
        for split, n in row["expected"].items():
            got = (row["answered"] or {}).get(split, 0)
            if got and n - got > 0.01 * n:
                flags.append(f"FAILURES: `{row['run']}` answered {got} of {n} {split} studies (§6)")
        if row["distinct"] is not None and row["distinct"] < COARSE:
            flags.append(f"COARSE: `{row['run']}` has a median of {row['distinct']:.0f} distinct scores per finding")
    keep = next(row["v2"] for row in ladder if row["a"] == "jev_est" and row["b"] == image_run)
    if keep and keep["diff"] < -LOST_ESTIMATES:
        flags.append(f"LOST ESTIMATES: `jev_est` is {fmt(-keep['diff'])} below the image run: the ladder's steps "
                     "start from a weaker reading of the image (§6)")
    return flags


def write(out_dir: Path, result: dict) -> Path:
    s = result["setup"]
    lines = ["# v3: Jev as the combiner", "",
             f"Stage 1 `{s['image_run']}`; baselines `{s['stack_indication']}`, `{s['stack_text']}`, "
             f"`{s['text_run']}`. Official test, macro AUROC over the frozen measurable classes, paired patient "
             "bootstrap (docs/V3_PLAN.md §5). Headline under the v2 label policy; positive vs rest (u0) next to it.",
             "", f"**Gate J1:** {_diff(result['j1'])} -> **{result['verdict']}**", ""]
    v = result["val_rule"]
    if v:
        revise = v["diff"] < -VAL_REVISE
        lines += [f"Val prompt rule (§7 step 4): `{prompt.PRIMARY}` minus `{s['stack_text']}` on val "
                  f"{_diff(v)} -> " + (f"more than {VAL_REVISE} below: revise the prompt on val before the test run"
                                       if revise else "within the margin: the prompt stands"), ""]
    lines += ["## Red flags", ""] + ([f"- {f}" for f in result["flags"]] or ["- none"]) + [""]
    lines += ["## Runs", "",
              "| Run | Role | Val | Official test | Official test (u0) | Answered (val / test) | Input tokens | "
              "Distinct scores |", "|---|---|---|---|---|---|---|---|"]
    for row in result["table"]:
        ans = row["answered"]
        answered = f"{ans.get('val', 0):,} / {ans.get('test', 0):,}" if ans else ""
        lines.append(f"| `{row['run']}` | {row['role']} | {_auc(row['val'])} | {_auc(row['test'])} | "
                     f"{_auc(row['test_u0'])} | {answered} | {row['tokens'] if row['tokens'] is not None else ''} | "
                     f"{'' if row['distinct'] is None else int(row['distinct'])} |")
    lines += ["", "## The context ladder (official test, paired)", "",
              "| Step | A | B | A minus B | A minus B (u0) |", "|---|---|---|---|---|"]
    for row in result["ladder"]:
        lines.append(f"| {row['step']} | `{row['a']}` | `{row['b']}` | {_diff(row['v2'])} | {_diff(row['u0'])} |")
    models = sorted({row["model"] for row in result["table"] if row["model"]})
    hashes = {row["run"]: row["prompt_hash"] for row in result["table"] if row["prompt_hash"]}
    lines += ["", f"Jev model: {', '.join(f'`{m}`' for m in models) or 'none yet'}. Prompt hashes: "
              + (", ".join(f"`{r}` {h}" for r, h in hashes.items()) or "none yet") + ".", "",
              "Known limitations: one stage-1 seed; the labels are the CheXpert labeler's, which understate image "
              "models by about 0.03 (M7a); val chose the image model and the combiner's inputs."]
    path = Path(out_dir) / "reports" / "v3_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    (path.with_suffix(".json")).write_text(json.dumps(result, indent=2, default=str))
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
    logger.info("J1: %s -> %s", _diff(result["j1"]), result["verdict"])
    for flag in result["flags"]:
        logger.warning("%s", flag)
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
