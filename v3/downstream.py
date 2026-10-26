"""
v3.5 (docs/V3_PLAN.md §8e): Jev's report labels as training targets for v2's probe, judged by a radiologist.
No Jev requests: the labels come from v3.label's cache, so run these first (each is cached and resumable):

    python -m v3.label run --split train --limit 20000 --model MODEL
    python -m v3.label run --split val --model MODEL
    python -m v3.downstream --model MODEL --labels output/mimic-cxr-2.1.0-test-set-labeled.csv  # -> v3_downstream.md

Three probes on the same training studies and frozen RAD-DINO features, differing only in their targets: the
CheXpert labeler's labels, Jev's answers, and Jev's probabilities as soft targets. Each chooses C on val with its own
source's labels. v2.evaluate scores them against the radiologist's labels on M7a's studies (gate L2) and, for
reference, against the CheXpert labeler's labels on the whole official test.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from v2.common import LABELS, TEST, TRAIN, VAL, label_matrix, load_manifest, load_measurable, read_predictions
from v2.common import setup_logging, targets_and_mask, write_predictions
from v2.embed import load_npz
from v2.evaluate import auroc, compare_runs, evaluate_set, fmt, fmt_ci
from v2.probe import DEFAULT_CS, LabelFit, Probe, edge_note, fit_probe, jobs_for_memory, split_xy

from . import label
from .common import default_out_dir, load_setup
from .run import load_cache

logger = logging.getLogger("v3.downstream")

N_TRAIN = 20000
ARMS = {"probe20k_chexpert": "the CheXpert labeler's labels", "probe20k_jev_hard": "Jev's answers",
        "probe20k_jev_soft": "Jev's probabilities, as soft targets"}
REFERENCE = "rad-dino_518_probe"  # v2: the same probe on all 213k training studies, CheXpert labels
MAX_UNANSWERED = 0.01


def jev_targets(cache: dict, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(raw labels in the labeler's coding: present 1, uncertain -1, absent 0; soft targets P(present) + 0.5 P(unc.))"""
    code = {"present": 1.0, "uncertain": -1.0, "absent": 0.0}
    raw = np.array([[code[c] for c in cache[int(s)]["choices"]] for s in ids], dtype=np.float64)
    probs = np.array([cache[int(s)]["probs"] for s in ids], dtype=np.float64)  # (n, 12, present/uncertain/absent)
    return raw, probs[:, :, 0] + 0.5 * probs[:, :, 1]


def _fit_soft_label(label_name, Z, p, Zv, yv, mv, Cs, max_iter) -> LabelFit:
    """v2's per-label fit with soft targets: each row once as positive (weight p) and once as negative (1 - p)."""
    dim = Z.shape[1]
    if p.max() - p.min() < 1e-12:
        return LabelFit(label_name, None, {}, np.zeros(dim), 0.0, "constant soft target: constant score")
    Z2, y2, w2 = np.concatenate([Z, Z]), np.r_[np.ones(len(p)), np.zeros(len(p))], np.r_[p, 1.0 - p]
    best, best_auc, scores, raw = None, -np.inf, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=max_iter).fit(Z2, y2, sample_weight=w2)
            a = auroc(yv[mv], clf.decision_function(Zv[mv])) if mv.any() else float("nan")
            raw[C] = a
            scores[str(C)] = None if np.isnan(a) else round(float(a), 4)
            if best is None or (np.isfinite(a) and a > best_auc):
                best, best_auc = (C, clf), (a if np.isfinite(a) else best_auc)
    note = edge_note(Cs, raw, best[0]) if np.isfinite(best_auc) else f"val AUROC undefined: kept C={best[0]}"
    C, clf = best
    return LabelFit(label_name, C, scores, clf.coef_[0].astype(np.float64), float(clf.intercept_[0]), note)


def fit_soft_probe(X_tr: np.ndarray, P_tr: np.ndarray, X_va: np.ndarray, raw_va: np.ndarray, Cs=DEFAULT_CS,
                   n_jobs: int = -1, max_iter: int = 2000) -> Probe:
    """fit_probe (v2) with soft training targets P_tr [n, 12]; C chosen by val AUROC against raw_va (v2 policy)."""
    Y_va, M_va = targets_and_mask(raw_va, "v2")
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_va = scaler.transform(X_tr), scaler.transform(X_va)
    fits = Parallel(n_jobs=jobs_for_memory(n_jobs, 2 * Z_tr.shape[0], Z_tr.shape[1]))(
        delayed(_fit_soft_label)(name, Z_tr, P_tr[:, k], Z_va, Y_va[:, k], M_va[:, k], sorted(Cs), max_iter)
        for k, name in enumerate(LABELS))
    return Probe(scaler, list(fits))


def answered(cache: dict, ids: np.ndarray, what: str) -> np.ndarray:
    have = np.array([int(s) in cache for s in ids])
    missing = int((~have).sum())
    if missing > MAX_UNANSWERED * len(ids):
        raise SystemExit(f"{missing} of {len(ids)} {what} studies have no Jev label: run `python -m v3.label run` "
                         f"for them first (§8e)")
    if missing:
        logger.warning("%d of %d %s studies have no Jev label and are left out of every arm", missing, len(ids), what)
    return np.asarray(ids)[have]


def train(out_dir: Path, model: str, emb_path: Path, n_train: int = N_TRAIN, n_jobs: int = -1) -> dict:
    """Fit the three probes and write their val and test predictions. Returns aggregate facts for the report."""
    probes, rows, X_all, facts, meta = fit_probes(out_dir, model, emb_path, n_train, n_jobs)
    for run, probe in probes.items():
        write_predictions(out_dir, run, rows["study_id"], rows["split"], probe.predict(X_all), {
            "kind": "probe", "milestone": "v3.5", "targets": ARMS[run], "embeddings": str(emb_path),
            "encoder": meta.get("encoder"), "size": meta.get("size"), "n_train": facts["n_train"],
            "n_val_for_C": facts["n_val"], "Cs": list(DEFAULT_CS), "per_label": probe.summary(),
            "label_prompt_hash": label.label_hash(), "label_model": model})
    return facts


def fit_probes(out_dir: Path, model: str, emb_path: Path, n_train: int = N_TRAIN, n_jobs: int = -1):
    """
    The three probes, deterministic given the same inputs; (probes, the val and test rows, their features, aggregate
    facts, the embeddings' meta). v3.6 refits them this way and checks them against the stored predictions.
    """
    manifest = load_manifest(out_dir)
    ids, emb, meta = load_npz(emb_path)
    parts = split_xy(manifest, ids, emb)
    del emb
    cache = load_cache(label.cache_path(out_dir, model))
    sample = label.target_rows(out_dir, "train", None, n_train)["study_id"].to_numpy()
    (tr_rows, X_tr), (va_rows, X_va), (te_rows, X_te) = parts[TRAIN], parts[VAL], parts[TEST]
    keep_tr = answered(cache, sample, "training")
    sel = tr_rows["study_id"].isin(keep_tr).to_numpy()
    tr_rows, X_tr = tr_rows[sel].reset_index(drop=True), X_tr[sel]
    keep_va = answered(cache, va_rows["study_id"].to_numpy(), "val")
    sel = va_rows["study_id"].isin(keep_va).to_numpy()
    va_fit_rows, X_va_fit = va_rows[sel].reset_index(drop=True), X_va[sel]
    jev_tr, soft_tr = jev_targets(cache, tr_rows["study_id"].to_numpy())
    jev_va, _ = jev_targets(cache, va_fit_rows["study_id"].to_numpy())
    logger.info("v3.5 probes: %d training studies, %d val studies for C, dim %d", len(tr_rows), len(va_fit_rows),
                X_tr.shape[1])
    probes = {
        "probe20k_chexpert": fit_probe(X_tr, label_matrix(tr_rows), X_va_fit, label_matrix(va_fit_rows), n_jobs=n_jobs),
        "probe20k_jev_hard": fit_probe(X_tr, jev_tr, X_va_fit, jev_va, n_jobs=n_jobs),
        "probe20k_jev_soft": fit_soft_probe(X_tr, soft_tr, X_va_fit, jev_va, n_jobs=n_jobs),
    }
    rows = pd.concat([va_rows, te_rows], ignore_index=True)
    facts = {"n_train": int(len(tr_rows)), "n_val": int(len(va_fit_rows)),
             "unanswered": {"train": int(len(sample) - len(keep_tr)), "val": int(len(va_rows) - len(keep_va))},
             "C": {run: {f.label: f.C for f in p.fits} for run, p in probes.items()}}
    return probes, rows, np.concatenate([X_va, X_te]), facts, meta


def link_reference(out_dir: Path) -> Optional[str]:
    """v2's full-data probe, linked from the benchmark when it is there."""
    bench = Path(load_setup(out_dir)["bench"])
    for suffix in (".parquet", ".json"):
        src, dst = bench / "predictions" / f"{REFERENCE}{suffix}", Path(out_dir) / "predictions" / f"{REFERENCE}{suffix}"
        if not src.exists():
            return None
        if not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(src)
    return REFERENCE


def build_report(out_dir: Path, labels_path: Path, facts: dict, n_boot: int = 1000) -> dict:
    labeler, reader, frozen_r, info = label.radiologist_set(out_dir, labels_path)
    manifest, frozen = load_manifest(out_dir), load_measurable(out_dir)
    runs = [*ARMS] + ([REFERENCE] if link_reference(out_dir) else [])
    preds = {run: read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")[0] for run in runs}
    rad = {run: {m: evaluate_set(p, reader, frozen_r, "test_clean", m, n_boot) for m in ("v2", "u0")}
           for run, p in preds.items()}
    chex = {run: evaluate_set(p, manifest, frozen, "test_clean", "v2", n_boot) for run, p in preds.items()}
    cmp = {}
    for run in [r for r in runs if r != "probe20k_chexpert"]:
        for m in ("v2", "u0"):
            cmp[f"{run}:radiologist:{m}"] = compare_runs(out_dir, run, "probe20k_chexpert", "test_clean", m, n_boot,
                                                         write=False, manifest=reader, frozen=frozen_r)
        cmp[f"{run}:chexpert:v2"] = compare_runs(out_dir, run, "probe20k_chexpert", "test_clean", "v2", n_boot,
                                                 write=False)
    l2 = cmp["probe20k_jev_soft:radiologist:v2"]
    verdict = ("n/a: no measurable classes" if not np.isfinite(l2["diff"]) else
               "PASS: Jev's soft labels train a better probe, as judged by the radiologist" if l2["a_better"] else
               "FAIL: the CheXpert labeler's labels train a better probe" if l2["b_better"] else
               "no detectable difference between the label sources (the CI includes 0)")
    return {"info": info, "frozen": frozen_r, "facts": facts, "runs": runs, "rad": rad, "chex": chex, "cmp": cmp,
            "l2": l2, "verdict": verdict}


def write_report(out_dir: Path, r: dict) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    def diff(key):
        c = r["cmp"].get(key)
        return fmt_ci(c["diff"], c["diff_ci"]) if c else "-"

    f, l2 = r["facts"], r["l2"]
    role = {**ARMS, REFERENCE: "reference: all 213k training studies, CheXpert labels (v2)"}
    lines = ["# v3.5: Jev's labels as training targets, judged by a radiologist", "",
             f"Three v2 probes on the same {f['n_train']:,} training studies and frozen RAD-DINO 518 px features, "
             f"differing only in their targets; each chose C on {f['n_val']:,} val studies with its own source's "
             f"labels (docs/V3_PLAN.md §8e). Studies without a Jev label, left out of every arm: "
             f"{f['unanswered']['train']} training, {f['unanswered']['val']} val.", "",
             f"**Gate L2:** `probe20k_jev_soft` minus `probe20k_chexpert` against the radiologist "
             f"{fmt_ci(l2['diff'], l2['diff_ci'])} ({len(l2['classes'])} classes, {r['info']['in_official_test']} "
             f"studies, v2 policy) -> **{r['verdict']}**", "",
             "## Against the radiologist (M7a's 596 official-test studies)", "",
             "| Probe | Targets | Macro AUROC (v2) | Positive vs rest | Minus `probe20k_chexpert` (v2) | (u0) |",
             "|---|---|---|---|---|---|"]
    for run in r["runs"]:
        lines.append(f"| `{run}` | {role[run]} | {auc(r['rad'][run]['v2'])} | {auc(r['rad'][run]['u0'])} | "
                     f"{diff(f'{run}:radiologist:v2')} | {diff(f'{run}:radiologist:u0')} |")
    lines += ["", "## Against the CheXpert labeler's labels (the whole official test; favours `probe20k_chexpert`)", "",
              "| Probe | Macro AUROC | Minus `probe20k_chexpert` |", "|---|---|---|"]
    for run in r["runs"]:
        lines.append(f"| `{run}` | {auc(r['chex'][run])} | {diff(f'{run}:chexpert:v2')} |")
    lines += ["", "## Per finding, against the radiologist (v2 policy)", "",
              "| Finding | Radiologist positives | " + " | ".join(f"`{run}`" for run in r["runs"]) + " |",
              "|---|---|" + "---|" * len(r["runs"])]
    per = {run: {c["label"]: c for c in r["rad"][run]["v2"]["per_class"]} for run in r["runs"]}
    for name in LABELS:
        if name in r["frozen"]["test_clean:v2"]:
            lines.append(f"| {name} | {per[r['runs'][0]][name]['n_pos']:,} | "
                         + " | ".join(fmt(per[run][name]["auroc"]) for run in r["runs"]) + " |")
    lines += ["", "Chosen C per finding: " + "; ".join(
        f"`{run}` " + ", ".join(f"{k} {v:g}" if v is not None else f"{k} -" for k, v in cs.items())
        for run, cs in f["C"].items())]
    path = Path(out_dir) / "reports" / "v3_downstream.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str))
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="The Jev version whose cached labels to use")
    parser.add_argument("--labels", help="the radiologist's labels (default: MIMIC_CXR_JPG_PATH/...test-set-labeled.csv)")
    parser.add_argument("--embeddings", help="default: the benchmark's embeddings/rad-dino_518.npz")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    labels = label.resolve_paths(None, args.labels, need_reports=False)[1]
    emb = Path(args.embeddings) if args.embeddings else Path(load_setup(out_dir)["bench"]) / "embeddings" / "rad-dino_518.npz"
    if not emb.exists():
        raise SystemExit(f"{emb} not found: pass --embeddings")
    facts = train(out_dir, args.model, emb, args.n_train, args.n_jobs)
    result = build_report(out_dir, labels, facts, args.n_boot)
    path = write_report(out_dir, result)
    logger.info("L2: %s -> %s", fmt_ci(result["l2"]["diff"], result["l2"]["diff_ci"]), result["verdict"])
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
