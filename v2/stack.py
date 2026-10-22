"""
M4b part 3: what does text add to the fine-tuned image, when the combination is fit
on studies no model here trained on? (docs/V2_PLAN.md §6 M4b)

    python -m v2.stack --pair rad-dino_518_ft_s1 rad-dino_518_ctx_text \\
                       --pair rad-dino_518_probe rad-dino_518_ctx_text

Fitting text on top of fine-tuned features uses training studies the fine-tune has
already fit, where the image looks more reliable than it is. Here, per class, a
logistic regression (C = 1 on standardized inputs) combines two existing runs'
logits, the image run's and a text-only run's, and is fit on val. Val's own scores
come from 5-fold cross-fitting within val, grouped by patient; test scores come
from the fit on all of val. A class with only one label value in val, or in a
fold's training part, keeps the image run's score unchanged.

Writes predictions/{image_run}_stack_{text arm}.parquet and reports/m4b_stack.md
(reports/m5_stack.md in M5's benchmark directory, scored on the official test).
On frozen features (an image run of kind "probe") it also compares the stack with
the concatenation of the same inputs, {prefix}_ctx_image_{text arm}: the check
that the two-score combiner loses little.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import LABELS, VAL, default_out_dir, label_matrix, load_manifest, read_predictions, setup_logging
from .common import targets_and_mask, write_predictions
from .evaluate import _get, compare_runs, evaluate_run, fmt, fmt_ci, split_scheme

logger = logging.getLogger("v2.stack")

FOLDS = 5
C = 1.0
CHECK_MARGIN = 0.005  # §6 M4b: a stack this far below the concatenation on frozen features loses information


def _fit(X: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000)).fit(X, y)


def stack_scores(S_img: np.ndarray, S_txt: np.ndarray, raw: np.ndarray, is_val: np.ndarray, groups: np.ndarray,
                 folds: int = FOLDS) -> tuple[np.ndarray, dict]:
    """
    Stacked logits [n, 12] for every row. Val rows get cross-fitted scores (their own fold never trains the model
    that scores them); other rows are scored by the fit on all of val. Returns (scores, per-class info).
    """
    Y, M = targets_and_mask(raw, "v2")
    vidx = np.flatnonzero(is_val)
    n_folds = min(folds, len(np.unique(groups[vidx])))
    if n_folds < 2:
        raise SystemExit("val needs at least 2 patients to cross-fit the combiner")
    fold = np.empty(len(vidx), dtype=int)
    for f, (_, held) in enumerate(GroupKFold(n_splits=n_folds).split(vidx, groups=groups[vidx])):
        fold[held] = f
    out = S_img.astype(np.float64).copy()
    info = {}
    for k, label in enumerate(LABELS):
        X = np.stack([S_img[:, k], S_txt[:, k]], axis=1).astype(np.float64)
        fit_rows = [vidx[M[vidx, k] & (fold != f)] for f in range(n_folds)] + [vidx[M[vidx, k]]]
        if any(len(np.unique(Y[r, k])) < 2 for r in fit_rows):
            info[label] = {"kept_image_score": True}
            continue
        full = _fit(X[fit_rows[-1]], Y[fit_rows[-1], k])
        out[~is_val, k] = full.decision_function(X[~is_val])
        for f in range(n_folds):
            held = vidx[fold == f]
            out[held, k] = _fit(X[fit_rows[f]], Y[fit_rows[f], k]).decision_function(X[held])
        w = full[-1].coef_[0]
        info[label] = {"kept_image_score": False, "image_weight": round(float(w[0]), 4),
                       "text_weight": round(float(w[1]), 4), "intercept": round(float(full[-1].intercept_[0]), 4)}
    return out, info


def stack_pair(out_dir: Path, image_run: str, text_run: str, folds: int = FOLDS) -> tuple[str, dict]:
    pred_dir = Path(out_dir) / "predictions"
    a, meta_a = read_predictions(pred_dir / f"{image_run}.parquet")
    b, meta_b = read_predictions(pred_dir / f"{text_run}.parquet")
    both = a.merge(b, on=["study_id", "split"], suffixes=("", "::text"))
    lost = max(len(a), len(b)) - len(both)
    if lost:
        logger.warning("%s and %s: %d studies are in only one of them and are left out", image_run, text_run, lost)
    manifest = load_manifest(out_dir).drop_duplicates("study_id").set_index("study_id")
    man = manifest.loc[both["study_id"].to_numpy()]
    is_val = (both["split"] == VAL).to_numpy()
    if not is_val.any() or is_val.all():
        raise SystemExit(f"{image_run} and {text_run} need both val and test predictions")
    S_img = both[list(LABELS)].to_numpy(np.float64)
    S_txt = both[[f"{label}::text" for label in LABELS]].to_numpy(np.float64)
    scores, info = stack_scores(S_img, S_txt, label_matrix(man.reset_index()), is_val, man["subject_id"].to_numpy(),
                                folds)
    kept = [label for label, i in info.items() if i["kept_image_score"]]
    if kept:
        logger.warning("%s: %d classes keep the image score (one label value in val or a fold): %s", image_run,
                       len(kept), ", ".join(kept))
    run = f"{image_run}_stack_{meta_b.get('arm') or text_run}"
    write_predictions(out_dir, run, both["study_id"], both["split"], scores, {
        "kind": "stack", "image_run": image_run, "text_run": text_run, "folds": folds, "C": C,
        "image_kind": meta_a.get("kind"), "image_arm": meta_a.get("arm"), "image_head": meta_a.get("head"),
        "text_arm": meta_b.get("arm"), "encoder": meta_a.get("encoder"), "size": meta_a.get("size"),
        "image_features": "fine-tuned" if meta_a.get("kind") == "finetune" else meta_a.get("image_features", "frozen"),
        "n_val": int(is_val.sum()), "per_label": info,
    })
    return run, {"image_run": image_run, "text_run": text_run, "image_kind": meta_a.get("kind"),
                 "text_arm": meta_b.get("arm"), "info": info, "kept": kept}


def write_report(out_dir: Path, stacks: dict, n_boot: int) -> Path:
    bench = split_scheme(out_dir) == "official"
    if bench:  # M5: test_clean is the official test split, and test_v1 is empty
        title, where = "# M5: combiners fit on val, on the benchmark", "§6 M5b amendment"
        head = "| Stacked run | Image run | Text run | official test | minus the image run | minus the image run, " \
               "positive vs rest |"
        sets = (("test_clean", "v2"), ("test_clean", "u0"))
    else:
        title, where = "# M4b: a combiner fit on val (exploratory)", "§6 M4b part 3"
        head = "| Stacked run | Image run | Text run | test_clean | minus the image run, test_clean | " \
               "minus the image run, test_v1 | minus the image run, test_v1 (v1 metric) |"
        sets = (("test_clean", "v2"), ("test_v1", "v2"), ("test_v1", "v1"))
    lines = [title, "",
             f"Per class, a logistic regression on two runs' logits (C = {C}, standardized), fit on val; val's own "
             f"scores are cross-fitted ({FOLDS} folds grouped by patient), test scores come from the fit on all of "
             f"val (docs/V2_PLAN.md {where}).", "", head, "|---" * (len(sets) + 4) + "|"]
    checks, weights = [], []
    for run, s in stacks.items():
        clean = _get(evaluate_run(out_dir, run, n_boot), "test_clean", "v2")
        diffs = []
        for test_set, metric in sets:
            try:
                c = compare_runs(out_dir, run, s["image_run"], test_set, metric, n_boot)
                diffs.append(fmt_ci(c["diff"], c["diff_ci"]))
            except SystemExit as exc:
                logger.warning("%s - %s on %s skipped: %s", run, s["image_run"], test_set, exc)
                diffs.append("-")
        auc = fmt_ci(clean["macro_auroc"], clean["macro_auroc_ci"]) if clean else "-"
        lines.append(f"| `{run}` | `{s['image_run']}` | `{s['text_run']}` | {auc} | " + " | ".join(diffs) + " |")
        concat = f"{s['image_run'][: -len('_probe')]}_ctx_image_{s['text_arm']}" \
            if s["image_run"].endswith("_probe") and s["text_arm"] else None
        if s["image_kind"] == "probe" and concat and (Path(out_dir) / "predictions" / f"{concat}.parquet").exists():
            checks.append((run, concat, compare_runs(out_dir, run, concat, "test_clean", "v2", n_boot)))
        used = [i for i in s["info"].values() if not i["kept_image_score"]]
        if used:
            weights.append((run, np.mean([i["image_weight"] for i in used]), np.mean([i["text_weight"] for i in used]),
                            s["kept"]))
    lines += ["", "## The check: the stack against the concatenation of the same inputs, on frozen features", ""]
    if not checks:
        lines.append("- not run: needs a stack on a `*_probe` image run and its `*_ctx_image_{text arm}` run")
    for run, concat, c in checks:
        verdict = (f"more than {CHECK_MARGIN} below: the two-score combiner loses information, so its fine-tuned "
                   "estimate is a lower bound" if c["diff"] < -CHECK_MARGIN else
                   f"within {CHECK_MARGIN} or above: the combiner loses little")
        on = "the official test" if bench else "test_clean"
        lines.append(f"- `{run}` minus `{concat}` on {on}: {fmt_ci(c['diff'], c['diff_ci'])} -> {verdict}")
    lines += ["", "## Weights (fit on all of val, standardized inputs; mean over classes)", "",
              "| Stacked run | Image weight | Text weight | Classes kept at the image score |", "|---|---|---|---|"]
    for run, wi, wt, kept in weights:
        lines.append(f"| `{run}` | {fmt(wi)} | {fmt(wt)} | {', '.join(kept) or 'none'} |")
    lines += ["", "Known limitation: val also chose the probes' C and the fine-tune's epoch, so both inputs are "
              "slightly optimistic on val."]
    if not bench:
        lines[-1] += (" Under the v1 metric only findings the report mentions count, which biases text gains toward "
                      "zero (§6 M4b amendment): read the v2 columns first.")
    path = Path(out_dir) / "reports" / ("m5_stack.md" if bench else "m4b_stack.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("IMAGE_RUN", "TEXT_RUN"),
                        help="An image run and a text-only run to combine (repeatable)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    for image_run, text_run in args.pair:
        for run in (image_run, text_run):
            if not (out_dir / "predictions" / f"{run}.parquet").exists():
                raise SystemExit(f"predictions/{run}.parquet not found")
    stacks = dict(stack_pair(out_dir, image_run, text_run, args.folds) for image_run, text_run in args.pair)
    path = write_report(out_dir, stacks, args.n_boot)
    for run, s in stacks.items():
        c = compare_runs(out_dir, run, s["image_run"], "test_clean", "v2", args.n_boot)
        logger.info("%s - %s (test_clean): %s", run, s["image_run"], fmt_ci(c["diff"], c["diff_ci"]))
    logger.info("Report: %s", path)
    return {"runs": list(stacks), "report": path}


if __name__ == "__main__":
    main()
