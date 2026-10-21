"""
Metrics for v2: the only place results are computed (docs/V2_PLAN.md §3.2).

    python -m v2.evaluate --run rad-dino_224_probe       # one run -> reports/{run}.md/.json
    python -m v2.evaluate --all                          # every run + summary.md + gates
    python -m v2.evaluate --compare RUN_A RUN_B          # paired bootstrap on test_clean

Per test set: per-class AUROC with a 95% CI from a patient-level bootstrap,
AUPRC next to prevalence, and the macro AUROC over that test set's measurable
classes (frozen in measurable_classes.json by prepare_data).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from .common import (
    LABELS,
    MIN_COUNT_VAL,
    TEST,
    VAL,
    default_out_dir,
    label_matrix,
    load_manifest,
    load_measurable,
    measurable,
    read_predictions,
    setup_logging,
    targets_and_mask,
)

logger = logging.getLogger("v2.evaluate")

N_BOOT = 1000
G1_THRESHOLD = 0.77
# (test set, label policy) pairs every run is evaluated on, headline first.
EVAL_SETS = (("test_clean", "v2"), ("test_v1", "v2"), ("test_v1", "v1"), ("val", "v2"))
MILESTONE = {"probe": "M1", "zero_shot": "M2", "finetune": "M3", "fuse": "M4"}
# Encoders whose test_v1 numbers are not reportable (§4): they may have trained on those images
# and their reports. They are shown, flagged, but never used for the gates.
TEST_CLEAN_ONLY = {"medsiglip"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U, ties averaged); NaN if a class is absent."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    return float(average_precision_score(y, s))


def per_class_auroc(Y: np.ndarray, M: np.ndarray, S: np.ndarray, cols: Iterable[int]) -> np.ndarray:
    return np.array([auroc(Y[M[:, k], k], S[M[:, k], k]) for k in cols])


def bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row indices for `n_boot` resamples of whole patients (groups), with replacement."""
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(counts), len(counts))
        lens = counts[pick]
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        idx = np.repeat(starts[pick] - offsets, lens) + np.arange(lens.sum())
        out.append(order[idx])
    return out


def ci(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# ---------------------------------------------------------------------------
# Joining predictions with the manifest
# ---------------------------------------------------------------------------

def eval_rows(manifest: pd.DataFrame, test_set: str) -> pd.DataFrame:
    if test_set == "val":
        return manifest[manifest["split"] == VAL]
    flag = {"test_clean": "in_test_clean", "test_v1": "in_test_v1"}[test_set]
    return manifest[(manifest["split"] == TEST) & manifest[flag]]


def joined(pred: pd.DataFrame, manifest: pd.DataFrame, test_set: str) -> tuple[pd.DataFrame, int]:
    """Manifest rows of `test_set` that have predictions, with scores; and how many lack them."""
    rows = eval_rows(manifest, test_set)
    scores = pred.drop(columns="split").rename(columns={label: f"score::{label}" for label in LABELS})
    merged = rows.merge(scores, on="study_id", how="inner")
    return merged.sort_values("study_id").reset_index(drop=True), len(rows) - len(merged)


def classes_for(test_set: str, metric: str, frozen: dict, rows: pd.DataFrame) -> list[str]:
    key = f"{test_set}:{metric}"
    if test_set == "val":
        return measurable(label_matrix(rows), metric, MIN_COUNT_VAL)
    if key not in frozen:
        raise SystemExit(f"measurable_classes.json has no {key!r}: re-run v2.prepare_data")
    return list(frozen[key])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_set(
    pred: pd.DataFrame,
    manifest: pd.DataFrame,
    frozen: dict,
    test_set: str,
    metric: str,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Optional[dict]:
    rows, missing = joined(pred, manifest, test_set)
    if len(rows) == 0:
        return None
    Y, M = targets_and_mask(label_matrix(rows), metric)
    S = rows[[f"score::{label}" for label in LABELS]].to_numpy(np.float64)
    classes = classes_for(test_set, metric, frozen, rows)
    cols = [LABELS.index(c) for c in classes]

    point = per_class_auroc(Y, M, S, range(len(LABELS)))
    boots = bootstrap_indices(rows["subject_id"].to_numpy(), n_boot, seed)
    boot_class = np.full((n_boot, len(LABELS)), np.nan)
    for b, idx in enumerate(boots):
        boot_class[b] = per_class_auroc(Y[idx], M[idx], S[idx], range(len(LABELS)))
    with np.errstate(invalid="ignore"):
        boot_macro = np.nanmean(boot_class[:, cols], axis=1) if cols else np.full(n_boot, np.nan)

    per_class = []
    for k, label in enumerate(LABELS):
        m = M[:, k]
        n_pos = int(Y[m, k].sum())
        per_class.append({
            "label": label,
            "measurable": label in classes,
            "n_pos": n_pos,
            "n_neg": int(m.sum()) - n_pos,
            "prevalence": float(n_pos / m.sum()) if m.sum() else float("nan"),
            "auroc": float(point[k]),
            "auroc_ci": ci(boot_class[:, k]),
            "auprc": auprc(Y[m, k], S[m, k]),
        })
    macro = float(np.mean(point[cols])) if cols else float("nan")
    return {
        "test_set": test_set,
        "metric": metric,
        "n_studies": int(len(rows)),
        "n_patients": int(rows["subject_id"].nunique()),
        "n_missing_predictions": int(missing),
        "classes": classes,
        "macro_auroc": macro,
        "macro_auroc_ci": ci(boot_macro),
        "macro_auprc": float(np.nanmean([per_class[k]["auprc"] for k in cols])) if cols else float("nan"),
        "macro_auroc_all12": float(np.nanmean(point)),
        "n_boot": n_boot,
        "per_class": per_class,
    }


def caveats(meta: dict) -> dict:
    """Known pretraining exposure of each encoder to MIMIC-CXR test images (§4)."""
    encoder = meta.get("encoder", "")
    notes = {}
    if encoder == "rad-dino":
        notes["test_v1"] = "RAD-DINO saw these images (without labels) in pretraining: not a headline number."
    if encoder == "medsiglip":
        notes["test_v1"] = "MedSigLIP's MIMIC-CXR training split is undocumented: report test_clean only."
        notes["test_clean"] = "Flagged: MedSigLIP's MIMIC-CXR training split is undocumented (§4)."
    return notes


def evaluate_run(out_dir: Path, run: str, n_boot: int = N_BOOT, manifest=None, frozen=None) -> dict:
    manifest = load_manifest(out_dir) if manifest is None else manifest
    frozen = load_measurable(out_dir) if frozen is None else frozen
    pred, meta = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")
    results = []
    for test_set, metric in EVAL_SETS:
        res = evaluate_set(pred, manifest, frozen, test_set, metric, n_boot)
        if res is not None:
            results.append(res)
    report = {"run": run, "meta": meta, "caveats": caveats(meta), "results": results,
              "exposure_check": exposure_check(pred, manifest, frozen, n_boot)}
    write_report(out_dir, report)
    return report


def exposure_check(pred: pd.DataFrame, manifest: pd.DataFrame, frozen: dict, n_boot: int) -> Optional[dict]:
    """
    Macro AUROC on test_v1 over test_clean's measurable classes, so the two test sets compare like
    with like. Encoders that saw test_v1's images in pretraining (RAD-DINO, maybe MedSigLIP) should
    show a larger test_v1 - test_clean gap than one that did not (DINOv2).
    """
    classes = list(frozen.get("test_clean:v2", []))
    rows, _ = joined(pred, manifest, "test_v1")
    if not classes or len(rows) == 0:
        return None
    Y, M = targets_and_mask(label_matrix(rows), "v2")
    S = rows[[f"score::{label}" for label in LABELS]].to_numpy(np.float64)
    cols = [LABELS.index(c) for c in classes]
    point = float(np.mean(per_class_auroc(Y, M, S, cols)))
    boots = [np.nanmean(per_class_auroc(Y[i], M[i], S[i], cols))
             for i in bootstrap_indices(rows["subject_id"].to_numpy(), n_boot, 1)]
    return {"classes": classes, "test_v1_macro": point, "test_v1_macro_ci": ci(np.array(boots))}


def fmt(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def fmt_ci(point: float, interval: list) -> str:
    return f"{fmt(point)} [{fmt(interval[0])}, {fmt(interval[1])}]"


def write_report(out_dir: Path, report: dict) -> None:
    rep_dir = Path(out_dir) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{report['run']}.json").write_text(json.dumps(report, indent=2, default=str))
    meta = report["meta"]
    lines = [f"# {report['run']}", ""]
    described = {k: meta.get(k) for k in ("kind", "encoder", "size", "commit", "created") if meta.get(k) is not None}
    lines += [", ".join(f"{k}: `{v}`" for k, v in described.items()), ""]
    for res in report["results"]:
        title = f"{res['test_set']} ({res['metric']} label policy)"
        lines += [
            f"## {title}",
            "",
            f"{res['n_studies']:,} studies, {res['n_patients']:,} patients"
            + (f"; **{res['n_missing_predictions']:,} studies lack predictions**" if res["n_missing_predictions"] else "")
            + ".",
            "",
            f"**Macro AUROC ({len(res['classes'])} measurable classes): "
            f"{fmt_ci(res['macro_auroc'], res['macro_auroc_ci'])}**; "
            f"macro AUPRC {fmt(res['macro_auprc'])}; mean over all 12: {fmt(res['macro_auroc_all12'])}.",
        ]
        note = report["caveats"].get(res["test_set"])
        if note:
            lines += ["", f"> {note}"]
        lines += ["", "| Class | Measurable | Pos | Neg | Prevalence | AUROC [95% CI] | AUPRC |",
                  "|---|---|---|---|---|---|---|"]
        for c in res["per_class"]:
            lines.append(
                f"| {c['label']} | {'yes' if c['measurable'] else ''} | {c['n_pos']:,} | {c['n_neg']:,} | "
                f"{fmt(c['prevalence'])} | {fmt_ci(c['auroc'], c['auroc_ci'])} | {fmt(c['auprc'])} |"
            )
        lines.append("")
    (rep_dir / f"{report['run']}.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def compare_runs(
    out_dir: Path,
    run_a: str,
    run_b: str,
    test_set: str = "test_clean",
    metric: str = "v2",
    n_boot: int = N_BOOT,
    seed: int = 0,
    write: bool = True,
) -> dict:
    """Paired patient bootstrap of macro AUROC(A) - macro AUROC(B) on the same studies."""
    manifest, frozen = load_manifest(out_dir), load_measurable(out_dir)
    pa, _ = read_predictions(Path(out_dir) / "predictions" / f"{run_a}.parquet")
    pb, _ = read_predictions(Path(out_dir) / "predictions" / f"{run_b}.parquet")
    ra, _ = joined(pa, manifest, test_set)
    rb, _ = joined(pb, manifest, test_set)
    common = np.intersect1d(ra["study_id"], rb["study_id"])
    ra = ra[ra["study_id"].isin(common)].reset_index(drop=True)
    rb = rb[rb["study_id"].isin(common)].reset_index(drop=True)
    if len(ra) == 0:
        raise SystemExit(f"{run_a} and {run_b} share no {test_set} studies")
    Y, M = targets_and_mask(label_matrix(ra), metric)
    cols_names = classes_for(test_set, metric, frozen, ra)
    cols = [LABELS.index(c) for c in cols_names]
    score_cols = [f"score::{label}" for label in LABELS]
    Sa, Sb = ra[score_cols].to_numpy(np.float64), rb[score_cols].to_numpy(np.float64)

    def macro(idx):
        return np.nanmean(per_class_auroc(Y[idx], M[idx], Sa[idx], cols)) - np.nanmean(
            per_class_auroc(Y[idx], M[idx], Sb[idx], cols))

    full = np.arange(len(ra))
    point = macro(full)
    diffs = np.array([macro(idx) for idx in bootstrap_indices(ra["subject_id"].to_numpy(), n_boot, seed)])
    interval = ci(diffs)
    result = {
        "a": run_a, "b": run_b, "test_set": test_set, "metric": metric, "classes": cols_names,
        "n_studies": int(len(ra)), "n_patients": int(ra["subject_id"].nunique()),
        "macro_a": float(np.nanmean(per_class_auroc(Y, M, Sa, cols))),
        "macro_b": float(np.nanmean(per_class_auroc(Y, M, Sb, cols))),
        "diff": float(point), "diff_ci": interval,
        "share_boot_a_better": float(np.mean(diffs[np.isfinite(diffs)] > 0)),
        "a_better": bool(interval[0] > 0), "b_better": bool(interval[1] < 0),
        "n_boot": n_boot,
    }
    if write:
        rep_dir = Path(out_dir) / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        stem = f"compare_{run_a}__{run_b}__{test_set}"
        (rep_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
        verdict = (f"{run_a} is better" if result["a_better"] else
                   f"{run_b} is better" if result["b_better"] else "no significant difference")
        (rep_dir / f"{stem}.md").write_text(
            f"# {run_a} vs {run_b} on {test_set} ({metric} policy)\n\n"
            f"{result['n_studies']:,} studies, {result['n_patients']:,} patients, {len(cols)} classes.\n\n"
            f"| | Macro AUROC |\n|---|---|\n| {run_a} | {fmt(result['macro_a'])} |\n| {run_b} | {fmt(result['macro_b'])} |\n\n"
            f"**Difference (A - B): {fmt_ci(point, interval)}** -> {verdict} (paired 95% CI, {n_boot} patient resamples).\n"
        )
    return result


# ---------------------------------------------------------------------------
# Summary across runs, gates and results-log rows
# ---------------------------------------------------------------------------

def _get(report: dict, test_set: str, metric: str) -> Optional[dict]:
    for res in report["results"]:
        if res["test_set"] == test_set and res["metric"] == metric:
            return res
    return None


def summarize(out_dir: Path, n_boot: int = N_BOOT, runs: Optional[list[str]] = None) -> str:
    pred_dir = Path(out_dir) / "predictions"
    # --smoke runs only check the setup; they are never results.
    runs = runs or sorted(p.stem for p in pred_dir.glob("*.parquet") if not p.stem.endswith("_smoke"))
    if not runs:
        raise SystemExit(f"No predictions in {pred_dir}")
    manifest, frozen = load_manifest(out_dir), load_measurable(out_dir)
    reports = {run: evaluate_run(out_dir, run, n_boot, manifest, frozen) for run in runs}

    lines = ["# v2 results summary", "", "Macro AUROC [95% CI] over each test set's measurable classes. "
             "Model choices use `val` only.", "",
             "| Run | Kind | val | test_clean | test_v1 | test_v1 (v1 metric) |", "|---|---|---|---|---|---|"]
    flagged = False
    for run, rep in reports.items():
        cells = []
        for test_set, metric in (("val", "v2"), ("test_clean", "v2"), ("test_v1", "v2"), ("test_v1", "v1")):
            res = _get(rep, test_set, metric)
            mark = " †" if test_set == "test_v1" and rep["meta"].get("encoder") in TEST_CLEAN_ONLY else ""
            flagged = flagged or bool(mark)
            cells.append(fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) + mark if res else "-")
        lines.append(f"| {run} | {rep['meta'].get('kind', '?')} | " + " | ".join(cells) + " |")
    if flagged:
        lines += ["", "† Not reportable (§4): MedSigLIP may have trained on these images and their reports. "
                  "Flagged runs are never used for the gates."]

    checks = [(run, rep) for run, rep in reports.items() if rep.get("exposure_check") and _get(rep, "test_clean", "v2")]
    if checks:
        lines += ["", "## Exposure check (same classes on both test sets)", "",
                  "Macro AUROC over test_clean's measurable classes. A larger test_v1 - test_clean gap than "
                  "DINOv2's suggests test_v1 is inflated by pretraining exposure (§3.1).", "",
                  "| Run | test_clean | test_v1, same classes | Gap |", "|---|---|---|---|"]
        for run, rep in checks:
            clean, ex = _get(rep, "test_clean", "v2"), rep["exposure_check"]
            gap = ex["test_v1_macro"] - clean["macro_auroc"]
            lines.append(f"| {run} | {fmt(clean['macro_auroc'])} | {fmt_ci(ex['test_v1_macro'], ex['test_v1_macro_ci'])} "
                         f"| {gap:+.3f} |")

    def best(kind: str) -> Optional[str]:
        scored = [(_get(rep, "val", "v2") or {}).get("macro_auroc", float("nan")) for rep in reports.values()]
        cands = [(s, r) for s, (r, rep) in zip(scored, reports.items())
                 if rep["meta"].get("kind") == kind and np.isfinite(s)
                 and rep["meta"].get("encoder") not in TEST_CLEAN_ONLY]
        return max(cands)[1] if cands else None

    def missing(kind: str) -> str:
        runs_of_kind = [rep for rep in reports.values() if rep["meta"].get("kind") == kind]
        if not runs_of_kind:
            return f"no {kind} runs yet"
        if all(rep["meta"].get("encoder") in TEST_CLEAN_ONLY for rep in runs_of_kind):
            return f"n/a: only flagged {kind} runs (§4), which the gates don't use"
        return f"n/a: val has too few positives/negatives to pick a {kind} run"

    lines += ["", "## Gates", ""]
    best_probe = best("probe")
    if best_probe:
        rep = reports[best_probe]
        res = _get(rep, "test_v1", "v1") or _get(rep, "test_clean", "v2")
        where = f"{res['test_set']} ({res['metric']} metric)"
        if not np.isfinite(res["macro_auroc"]):
            verdict = "n/a: no measurable classes (rehearsal data?)"
        elif res["macro_auroc"] >= G1_THRESHOLD:
            verdict = "PASS: rent the GPU for M3"
        else:
            verdict = "FAIL: study per-class results first"
        excluded = " (flagged encoders excluded, §4)" if any(
            r["meta"].get("encoder") in TEST_CLEAN_ONLY for r in reports.values()) else ""
        lines.append(f"- **G1** (best probe by val{excluded}: `{best_probe}`): "
                     f"{fmt_ci(res['macro_auroc'], res['macro_auroc_ci'])} on {where} vs {G1_THRESHOLD} -> **{verdict}**")
    else:
        lines.append(f"- **G1**: {missing('probe')}")
    best_ft = best("finetune")
    if best_ft and best_probe:
        cmp = compare_runs(out_dir, best_ft, best_probe, "test_clean", "v2", n_boot)
        if not np.isfinite(cmp["diff"]):
            verdict = "n/a: no measurable classes (rehearsal data?)"
        elif cmp["a_better"]:
            verdict = "PASS: the fine-tuned model is the v2 baseline"
        else:
            verdict = "FAIL: the frozen probe stays the baseline"
        lines.append(f"- **G2** (`{best_ft}` - `{best_probe}` on test_clean): "
                     f"{fmt_ci(cmp['diff'], cmp['diff_ci'])} -> **{verdict}**")
    else:
        lines.append(f"- **G2**: {missing('finetune') if not best_ft else missing('probe')}")
    primaries = sorted(r for r, rep in reports.items() if rep["meta"].get("arm") == "image_text")
    frozen_seen = False
    for run in primaries:
        exploratory = reports[run]["meta"].get("image_features", "frozen") != "frozen"
        frozen_seen |= not exploratory
        name = "M4 on fine-tuned features (exploratory, not G4)" if exploratory else "**G4**"
        ref = run[: -len("image_text")] + "image"
        if ref not in reports:
            lines.append(f"- {name}: `{ref}` is missing; it is the reference for `{run}`")
            continue
        cmp = compare_runs(out_dir, run, ref, "test_clean", "v2", n_boot)
        if not np.isfinite(cmp["diff"]):
            verdict = "n/a: no measurable classes (rehearsal data?)"
        elif cmp["a_better"]:
            verdict = "PASS: pre-imaging text adds to the image"
        else:
            verdict = "FAIL: no detectable gain from pre-imaging text"
        prefix = run[: -len("_ctx_image_text")]
        lines.append(f"- {name} (`{run}` - `{ref}` on test_clean): {fmt_ci(cmp['diff'], cmp['diff_ci'])} -> "
                     f"**{verdict}**. Read the leakage red flags in reports/m4_{prefix}.md first.")
    if not frozen_seen:
        lines.append("- **G4**: no M4 runs on frozen image features yet")

    lines += ["", "## Rows for the results log (docs/V2_PLAN.md §9)", "", "```"]
    today = date.today().isoformat()
    for run, rep in reports.items():
        meta = rep["meta"]
        day = str(meta.get("created") or today)[:10]  # when the run was made, not when the summary is
        for test_set, metric in (("test_clean", "v2"), ("test_v1", "v1")):
            res = _get(rep, test_set, metric)
            if res:
                label = f"`{test_set}`" + (", v1 metric" if metric == "v1" else "")
                lines.append(f"| {day} | {MILESTONE.get(meta.get('kind'), '?')} | {meta.get('commit', '?')} | "
                             f"{run} | {label} | {fmt_ci(res['macro_auroc'], res['macro_auroc_ci'])} | "
                             f"{len(res['classes'])} | |")
    lines += ["```", ""]
    text = "\n".join(lines)
    (Path(out_dir) / "reports" / "summary.md").write_text(text)
    return text


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", nargs="+", help="Evaluate these runs (predictions/{run}.parquet)")
    group.add_argument("--all", action="store_true", help="Evaluate every run and write reports/summary.md")
    group.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Paired bootstrap A - B")
    parser.add_argument("--test-set", default="test_clean", choices=["test_clean", "test_v1", "val"])
    parser.add_argument("--metric", default="v2", choices=["v2", "v1"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()

    if args.compare:
        res = compare_runs(out_dir, *args.compare, args.test_set, args.metric, args.n_boot)
        print(f"{res['a']} - {res['b']} on {res['test_set']}: {fmt_ci(res['diff'], res['diff_ci'])} "
              f"({res['n_studies']} studies, {len(res['classes'])} classes)")
    elif args.all:
        print(summarize(out_dir, args.n_boot))
    else:
        for run in args.run:
            rep = evaluate_run(out_dir, run, args.n_boot)
            for res in rep["results"]:
                print(f"{run:40s} {res['test_set']:10s} {res['metric']}  "
                      f"{fmt_ci(res['macro_auroc'], res['macro_auroc_ci'])}  ({len(res['classes'])} classes)")


if __name__ == "__main__":
    main()
