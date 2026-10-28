"""
v3.4 (exploratory, docs/V3_PLAN.md §8d): two ways to close the gap between Jev's report labels and its probabilities.
No Jev requests: the cached v3.3 answers (predictions/labeler_jev*.parquet) only.

    python -m v3.bridge --labels output/mimic-cxr-2.1.0-test-set-labeled.csv     # -> reports/v3_bridge.md

1. Hierarchy: on the benchmark's training split, how often the CheXpert labeler marks a parent finding positive when
   any child is (HIERARCHY). A parent at HIERARCHY_MIN or above gets the rule: present when any child is present.
2. Cut-offs: per finding, two cut-offs on Jev's probability score turn it back into present / uncertain / absent,
   fit on the radiologist's labels by FOLDS-fold cross-fitting grouped by patient, so each study's label comes from
   cut-offs fit on the other folds.
Every run is scored against the radiologist by v2.evaluate, as in v3.3. Console output and the report are aggregates.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from v2.common import LABELS, TRAIN, label_matrix, load_manifest, read_predictions, setup_logging, targets_and_mask
from v2.common import write_predictions
from v2.evaluate import auroc, compare_runs, eval_rows, evaluate_set, fmt, fmt_ci

from .common import default_out_dir
from .label import radiologist_set, resolve_paths

logger = logging.getLogger("v3.bridge")

HIERARCHY = {"Lung Opacity": ("Atelectasis", "Consolidation", "Edema", "Lung Lesion", "Pneumonia"),
             "Enlarged Cardiomediastinum": ("Cardiomegaly",)}
HIERARCHY_MIN = 0.90
FOLDS = 5
GRID = 41
BASE_RUNS = ("labeler_chexpert", "labeler_jev", "labeler_jev_prob")


def hierarchy_rates(labels: pd.DataFrame) -> dict:
    """Per parent: P(parent positive | any child positive) under the given labels (1 = positive), with counts."""
    raw = label_matrix(labels)
    out = {}
    for parent, children in HIERARCHY.items():
        child = (raw[:, [LABELS.index(c) for c in children]] == 1.0).any(axis=1)
        par = raw[:, LABELS.index(parent)] == 1.0
        out[parent] = {"n_child_positive": int(child.sum()), "parent_given_child": float(par[child].mean())
                       if child.any() else float("nan"),
                       "per_child": {c: float(par[raw[:, LABELS.index(c)] == 1.0].mean())
                                     if (raw[:, LABELS.index(c)] == 1.0).any() else float("nan") for c in children}}
    return out


def apply_hierarchy(scores: np.ndarray, parents: list) -> np.ndarray:
    """Each listed parent's score raised to its children's highest: present whenever a child is."""
    out = scores.astype(np.float64).copy()
    for parent in parents:
        cols = [LABELS.index(c) for c in HIERARCHY[parent]]
        k = LABELS.index(parent)
        out[:, k] = np.maximum(out[:, k], out[:, cols].max(axis=1))
    return out


def three_level(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(s >= hi, 1.0, np.where(s >= lo, 0.5, 0.0))


def fit_cuts(s: np.ndarray, y: np.ndarray) -> Optional[tuple[float, float]]:
    """The (lo, hi) pair, lo <= hi, from GRID quantiles of s, with the highest AUROC of the three-level score."""
    if len(np.unique(y)) < 2:
        return None
    grid = np.unique(np.quantile(s, np.linspace(0, 1, GRID)))
    best, best_auc = None, -1.0
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            a = auroc(y, three_level(s, lo, hi))
            if a > best_auc + 1e-12:
                best, best_auc = (float(lo), float(hi)), a
    return best


def crossfit_cuts(S: np.ndarray, fallback: np.ndarray, Y: np.ndarray, M: np.ndarray, groups: np.ndarray,
                  folds: int = FOLDS) -> tuple[np.ndarray, dict]:
    """Three-level labels for every row from cut-offs fit on the other folds; `fallback` where a fit isn't possible."""
    out = fallback.astype(np.float64).copy()
    kept = {label: 0 for label in LABELS}
    n_folds = min(folds, len(np.unique(groups)))
    for train, held in GroupKFold(n_splits=n_folds).split(S, groups=groups):
        for k, label in enumerate(LABELS):
            rows = train[M[train, k]]
            cuts = fit_cuts(S[rows, k], Y[rows, k]) if len(rows) else None
            if cuts is None:
                kept[label] += 1
                continue
            out[held, k] = three_level(S[held, k], *cuts)
    return out, {"folds": n_folds, "folds_keeping_jev_answer": kept}


def build(out_dir: Path, labels_path: Path, n_boot: int = 1000) -> dict:
    labeler, reader, frozen, info = radiologist_set(out_dir, labels_path)
    preds = {}
    for run in BASE_RUNS:
        path = Path(out_dir) / "predictions" / f"{run}.parquet"
        if not path.exists():
            raise SystemExit(f"predictions/{run}.parquet not found: run `python -m v3.label run` first (§8c)")
        preds[run], _ = read_predictions(path)
    ids = preds["labeler_jev"]["study_id"].to_numpy()
    if not (np.array_equal(ids, preds["labeler_jev_prob"]["study_id"].to_numpy())):
        raise SystemExit("labeler_jev and labeler_jev_prob cover different studies")
    rows = reader.drop_duplicates("study_id").set_index("study_id").loc[ids].reset_index()
    Y, M = targets_and_mask(label_matrix(rows), "v2")
    choice = preds["labeler_jev"][list(LABELS)].to_numpy(np.float64)
    prob = preds["labeler_jev_prob"][list(LABELS)].to_numpy(np.float64)
    groups, splits = rows["subject_id"].to_numpy(), rows["split"].to_numpy()

    manifest = load_manifest(out_dir).drop_duplicates("study_id")
    train_rates = hierarchy_rates(manifest[manifest["split"] == TRAIN])
    parents = [p for p, r in train_rates.items() if r["parent_given_child"] >= HIERARCHY_MIN]
    radiologist_rates = hierarchy_rates(eval_rows(reader, "test_clean").drop_duplicates("study_id"))

    meta = {"kind": "labeler", "milestone": "v3.4", "exploratory": True}
    cut, cut_info = crossfit_cuts(prob, choice, Y, M, groups)
    new = {"labeler_jev_cut": (cut, {**meta, **cut_info, "score": "Jev's probabilities, cross-fitted cut-offs"})}
    if parents:
        new["labeler_jev_hier"] = (apply_hierarchy(choice, parents),
                                   {**meta, "parents": parents, "score": "Jev's answer + the hierarchy"})
        cut_h, info_h = crossfit_cuts(apply_hierarchy(prob, parents), apply_hierarchy(choice, parents), Y, M, groups)
        new["labeler_jev_cut_hier"] = (cut_h, {**meta, **info_h, "parents": parents,
                                               "score": "cut-offs on the hierarchy-adjusted probabilities"})
    for run, (scores, m) in new.items():
        write_predictions(out_dir, run, ids, splits, scores, m)
        preds[run], _ = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")

    scored = {run: {m: evaluate_set(p, reader, frozen, "test_clean", m, n_boot) for m in ("v2", "u0")}
              for run, p in preds.items()}
    cmp = {f"{run}:{b}:{m}": compare_runs(out_dir, run, b, "test_clean", m, n_boot, write=False, manifest=reader,
                                          frozen=frozen)
           for run in [r for r in preds if r != "labeler_chexpert"] for b in ("labeler_chexpert", "labeler_jev")
           if run != b for m in ("v2", "u0")}
    lo = LABELS.index("Lung Opacity")
    rad_lo = label_matrix(rows)[:, lo]
    confusion = {jev: {rad: int(((choice[:, lo] == level) & mask).sum())
                       for rad, mask in (("radiologist positive", rad_lo == 1.0), ("radiologist uncertain", rad_lo == -1.0),
                                         ("radiologist negative or blank", (rad_lo == 0.0) | np.isnan(rad_lo)))}
                 for jev, level in (("Jev present", 1.0), ("Jev uncertain", 0.5), ("Jev absent", 0.0))}
    return {"info": info, "frozen": frozen, "train_rates": train_rates, "parents": parents,
            "radiologist_rates": radiologist_rates, "scored": scored, "cmp": cmp, "confusion": confusion,
            "runs": list(preds), "new_runs": list(new)}


def write(out_dir: Path, r: dict) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    def diff(key):
        c = r["cmp"].get(key)
        return fmt_ci(c["diff"], c["diff_ci"]) if c else "-"

    lines = ["# v3.4 (exploratory): closing the gap in Jev's report labels", "",
             f"The radiologist's {r['info']['in_official_test']:,} official-test studies "
             f"({r['info']['patients']:,} patients), scored as in v3.3 against the radiologist's labels "
             "(docs/V3_PLAN.md §8d). No new Jev requests. Both fixes were prompted by v3.3's results on these same "
             "studies, so read this as a lead, not a revision of L1.", "",
             "## 1. The hierarchy, decided on the training split", "",
             f"| Parent | Studies with a positive child (train) | P(parent positive given a child), train | "
             f"Rule (at least {HIERARCHY_MIN}) | Same rate, radiologist (diagnostic) |", "|---|---|---|---|---|"]
    for parent, t in r["train_rates"].items():
        rr = r["radiologist_rates"][parent]
        lines.append(f"| {parent} | {t['n_child_positive']:,} | {fmt(t['parent_given_child'])} | "
                     f"{'applied' if parent in r['parents'] else 'not applied'} | {fmt(rr['parent_given_child'])} |")
    lines += ["", "Per child, train: " + "; ".join(
        f"{p}: " + ", ".join(f"{c} {fmt(v)}" for c, v in t["per_child"].items()) for p, t in r["train_rates"].items()),
        "", "## 2. Every labeler against the radiologist", "",
        "| Labeler | Macro AUROC (v2) | Positive vs rest | Minus the CheXpert labeler (v2) | (u0) | Minus Jev's "
        "answer (v2) |", "|---|---|---|---|---|---|"]
    for run in r["runs"]:
        s = r["scored"][run]
        lines.append(f"| `{run}` | {auc(s['v2'])} | {auc(s['u0'])} | {diff(f'{run}:labeler_chexpert:v2')} | "
                     f"{diff(f'{run}:labeler_chexpert:u0')} | {diff(f'{run}:labeler_jev:v2')} |")
    lines += ["", "## Per finding (v2 policy)", "", "| Finding | Radiologist positives | " +
              " | ".join(f"`{run}`" for run in r["runs"]) + " |", "|---|---|" + "---|" * len(r["runs"])]
    per = {run: {c["label"]: c for c in r["scored"][run]["v2"]["per_class"]} for run in r["runs"]}
    for label in LABELS:
        if label in r["frozen"]["test_clean:v2"]:
            lines.append(f"| {label} | {per['labeler_chexpert'][label]['n_pos']:,} | "
                         + " | ".join(fmt(per[run][label]["auroc"]) for run in r["runs"]) + " |")
    lines += ["", "## Lung Opacity: Jev's answer by the radiologist's label (counts, diagnostic)", "",
              "| | " + " | ".join(next(iter(r["confusion"].values()))) + " |", "|---|---|---|---|"]
    for jev, row in r["confusion"].items():
        lines.append(f"| {jev} | " + " | ".join(f"{v:,}" for v in row.values()) + " |")
    lines += ["", "The cut-offs learn from the radiologist's labels (cross-fitted, grouped by patient); the CheXpert "
              "labeler learns from none, but a three-level score's AUROC can't change under a monotone relabeling of "
              "its levels, so the labeler couldn't gain from cut-offs of its own."]
    path = Path(out_dir) / "reports" / "v3_bridge.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str))
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
    parser.add_argument("--labels", help="the radiologist's labels (default: MIMIC_CXR_JPG_PATH/...test-set-labeled.csv)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    labels = resolve_paths(None, args.labels, need_reports=False)[1]
    result = build(out_dir, labels, args.n_boot)
    path = write(out_dir, result)
    logger.info("Hierarchy rule for: %s", ", ".join(result["parents"]) or "no parent")
    for run in result["new_runs"]:
        c = result["cmp"][f"{run}:labeler_chexpert:v2"]
        logger.info("%s - labeler_chexpert: %s", run, fmt_ci(c["diff"], c["diff_ci"]))
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
