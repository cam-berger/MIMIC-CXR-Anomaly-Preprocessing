"""
v3's report (docs/V3_PLAN.md §5-7): the question format val chose, every Jev run and v2 baseline on val and the
official test, the context ladder, gate J1 and the red flags, all computed by v2.evaluate.

    python -m v3.report              # -> V3_OUT/reports/v3_report.md (and reports/{run}.md per run)

The format is chosen on val alone: jev_est_text in both formats, the higher val macro AUROC wins, a tie goes to
score (§7). Until then the ladder and J1 show "not run"; so do arms or test predictions not run yet.
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
ROLES = {"jev_est": "estimates", "jev_est_indication": "+ indication", "jev_est_text": "+ complaint, prior report (J1)",
         "jev_text": "text only (control)", "jev_img_est_text": "+ the image (v3.1)"}


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


def choose_format(out_dir: Path, n_boot: int = 20) -> Optional[dict]:
    """
    §7: the format whose jev_est_text has the higher val macro AUROC (v2 policy); a tie goes to score, the finer one.
    None until both val runs exist. Only the point estimate decides; the interval is reported.
    """
    runs = {f: prompt.run_name(prompt.PRIMARY, f) for f in prompt.FORMATS}
    if not all(_has(out_dir, r) for r in runs.values()):
        return None
    try:
        c = compare_runs(out_dir, runs["score"], runs["noul"], "val", "v2", n_boot, write=False)
    except SystemExit:
        return None
    chosen = "noul" if np.isfinite(c["diff"]) and c["diff"] < 0 else "score"
    return {"chosen": chosen, "val": {"score": c["macro_a"], "noul": c["macro_b"]}, "diff": c["diff"],
            "diff_ci": c["diff_ci"], "runs": runs}


def distinct_scores(out_dir: Path, run: str) -> Optional[float]:
    """Median over the 12 findings of the number of distinct official-test scores (all studies if no test yet)."""
    wide, _ = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")
    test = wide[wide["split"] == TEST]
    rows = test if len(test) else wide
    return float(np.median([rows[label].nunique() for label in LABELS])) if len(rows) else None


def build(out_dir: Path, n_boot: int = 1000) -> dict:
    s = load_setup(out_dir)
    img, stack_ind, stack_txt, txt = s["image_run"], s["stack_indication"], s["stack_text"], s["text_run"]
    choice = choose_format(out_dir, n_boot)
    chosen = choice["chosen"] if choice else None
    roles = {img: "stage 1, image only", stack_ind: "val combiner, + indication",
             stack_txt: "val combiner, + all text (J1)", txt: "TF-IDF, text only"}
    for f in ([chosen] if chosen else []) + [f for f in prompt.FORMATS if f != chosen]:
        for arm, role in ROLES.items():
            run = prompt.run_name(arm, f)
            if f == chosen and arm in prompt.V3_0_ARMS or _has(out_dir, run):
                roles[run] = f"{role}, {f}" + ("" if f == chosen or not chosen else " (not chosen)")
    reports = {r: evaluate_run(out_dir, r, n_boot) for r in roles if _has(out_dir, r)}
    table = []
    for r, role in roles.items():
        rep = reports.get(r)
        meta = rep["meta"] if rep else {}
        table.append({
            "run": r, "role": role, "kind": meta.get("kind"),
            "val": _get(rep, "val", "v2") if rep else None,
            "test": _get(rep, "test_clean", "v2") if rep else None,
            "test_u0": _get(rep, "test_clean", "u0") if rep else None,
            "answered": meta.get("studies_answered"), "expected": meta.get("studies_expected"),
            "tokens": meta.get("input_tokens"), "model": meta.get("model"), "prompt_hash": meta.get("prompt_hash"),
            "distinct": distinct_scores(out_dir, r) if rep and meta.get("kind") == "jev" else None,
        })
    ladder, j1, val_rule = [], None, None
    if chosen:
        j = {arm: prompt.run_name(arm, chosen) for arm in prompt.ARMS}
        spec = [("Jev keeps the estimates", j["jev_est"], img),
                ("+ indication", j["jev_est_indication"], j["jev_est"]),
                ("+ complaint and prior report", j["jev_est_text"], j["jev_est_indication"]),
                ("All the text", j["jev_est_text"], j["jev_est"]),
                ("Jev against TF-IDF, text alone", j["jev_text"], txt),
                ("Like for like with MMBT's inputs", j["jev_est_indication"], stack_ind),
                ("J1: Jev against the val combiner", j["jev_est_text"], stack_txt)]
        if _has(out_dir, j["jev_img_est_text"]):
            spec.insert(4, ("+ the image (v3.1)", j["jev_img_est_text"], j["jev_est_text"]))
        ladder = [{"step": step, "a": a, "b": b, "v2": _cmp(out_dir, a, b, "test_clean", "v2", n_boot),
                   "u0": _cmp(out_dir, a, b, "test_clean", "u0", n_boot)} for step, a, b in spec]
        j1 = ladder[-1]["v2"]
        val_rule = _cmp(out_dir, j["jev_est_text"], stack_txt, "val", "v2", n_boot)
    return {"setup": s, "choice": choice, "table": table, "ladder": ladder, "j1": j1, "verdict": j1_verdict(j1),
            "val_rule": val_rule, "flags": red_flags(table, ladder, img, chosen)}


def j1_verdict(c: Optional[dict]) -> str:
    if c is None:
        return "not run: needs val's format choice, then jev_est_text and the val combiner on the official test"
    if not np.isfinite(c["diff"]):
        return "n/a: no measurable classes"
    if c["a_better"]:
        return "PASS: Jev combines the estimates and the text better than the logistic regression"
    if c["b_better"]:
        return "FAIL: Jev loses information the logistic regression keeps"
    return "no detectable gain over the logistic regression (the CI includes 0)"


def red_flags(table: list, ladder: list, image_run: str, chosen: Optional[str]) -> list[str]:
    by_run = {row["run"]: row for row in table}
    flags = []
    for f in prompt.FORMATS:
        name = prompt.run_name("jev_text", f)
        jt, im = by_run.get(name, {}).get("test"), by_run.get(image_run, {}).get("test")
        if jt and im and jt["macro_auroc"] > im["macro_auroc"]:
            flags.append(f"LEAKAGE: `{name}` ({fmt(jt['macro_auroc'])}) beats the image run "
                         f"({fmt(im['macro_auroc'])}) on the official test: stop and inspect locally (§6)")
        if jt:
            high = [c["label"] for c in jt["per_class"] if c["measurable"] and c["auroc"] > LEAK_CLASS]
            if high:
                flags.append(f"LEAKAGE: `{name}` above {LEAK_CLASS} on {', '.join(high)}: stop and inspect locally (§6)")
    for row in table:
        if row["kind"] != "jev" or not row["expected"]:
            continue
        for split, n in row["expected"].items():
            got = (row["answered"] or {}).get(split, 0)
            if got and n - got > 0.01 * n:
                flags.append(f"FAILURES: `{row['run']}` answered {got} of {n} {split} studies (§6)")
        if row["distinct"] is not None and row["distinct"] < COARSE:
            flags.append(f"COARSE: `{row['run']}` has a median of {row['distinct']:.0f} distinct scores per finding")
    if chosen:
        keep = ladder[0]["v2"]
        if keep and keep["diff"] < -LOST_ESTIMATES:
            flags.append(f"LOST ESTIMATES: `{ladder[0]['a']}` is {fmt(-keep['diff'])} below the image run: the "
                         "ladder's steps start from a weaker reading of the image (§6)")
    return flags


def write(out_dir: Path, result: dict) -> Path:
    s, choice = result["setup"], result["choice"]
    lines = ["# v3: Jev as the combiner", "",
             f"Stage 1 `{s['image_run']}`; baselines `{s['stack_indication']}`, `{s['stack_text']}`, "
             f"`{s['text_run']}`. Official test, macro AUROC over the frozen measurable classes, paired patient "
             "bootstrap (docs/V3_PLAN.md §5). Headline under the v2 label policy; positive vs rest (u0) next to it.", ""]
    if choice:
        r = choice["runs"]
        lines += [f"**Question format (§7, val):** `{r['score']}` {fmt(choice['val']['score'])} against `{r['noul']}` "
                  f"{fmt(choice['val']['noul'])}; score minus noul {fmt_ci(choice['diff'], choice['diff_ci'])} -> "
                  f"**{choice['chosen']}**", ""]
    else:
        lines += [f"**Question format (§7, val):** not chosen yet: needs `{prompt.PRIMARY}` on val in both formats", ""]
    lines += [f"**Gate J1:** {_diff(result['j1'])} -> **{result['verdict']}**", ""]
    v = result["val_rule"]
    if v:
        revise = v["diff"] < -VAL_REVISE
        lines += [f"Val prompt rule (§7 step 4): `{v['a']}` minus `{s['stack_text']}` on val {_diff(v)} -> "
                  + (f"more than {VAL_REVISE} below: revise the prompt on val before the test run" if revise else
                     "within the margin: the prompt stands"), ""]
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
    lines += ["", "## The context ladder (official test, paired)", ""]
    if result["ladder"]:
        lines += ["| Step | A | B | A minus B | A minus B (u0) |", "|---|---|---|---|---|"]
        for row in result["ladder"]:
            lines.append(f"| {row['step']} | `{row['a']}` | `{row['b']}` | {_diff(row['v2'])} | {_diff(row['u0'])} |")
    else:
        lines.append("Not run: it is read in the format val chooses.")
    models = sorted({row["model"] for row in result["table"] if row["model"]})
    hashes = {row["run"]: row["prompt_hash"] for row in result["table"] if row["prompt_hash"]}
    lines += ["", f"Jev model: {', '.join(f'`{m}`' for m in models) or 'none yet'}. Prompt hashes: "
              + (", ".join(f"`{r}` {h}" for r, h in hashes.items()) or "none yet") + ".", "",
              "Known limitations: one stage-1 seed; the labels are the CheXpert labeler's, which understate image "
              "models by about 0.03 (M7a); val chose the image model, the combiner's inputs and the question format."]
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
    if result["choice"]:
        logger.info("Format (val): %s", result["choice"]["chosen"])
    logger.info("J1: %s -> %s", _diff(result["j1"]), result["verdict"])
    for flag in result["flags"]:
        logger.warning("%s", flag)
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
