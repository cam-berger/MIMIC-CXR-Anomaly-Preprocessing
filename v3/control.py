"""
v3.7 (exploratory, docs/V3_PLAN.md §8g): the hierarchy control. v3.5's CheXpert-labeled probe refit with the
hierarchy applied to its labels (a parent positive whenever a child is), against Jev's probes, on the CheXpert test
set and on M7a's MIMIC-CXR studies. No Jev requests.

    python -m v3.control --model MODEL --labels output/mimic-cxr-2.1.0-test-set-labeled.csv   # -> reports/v3_control.md

Needs v3.5 (predictions/probe20k_*) and v3.6 (V3_OUT/chexpert/) first. The probe trains on v3.5's own studies,
features, C grid and selection (downstream.training_data); only its labels differ.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v2.common import LABELS, label_matrix, load_manifest, load_measurable, read_predictions, setup_logging
from v2.common import write_predictions
from v2.embed import load_npz
from v2.evaluate import compare_runs, evaluate_set, fmt, fmt_ci
from v2.external import FIVE, _env_dir
from v2.probe import DEFAULT_CS, fit_probe

from . import chexpert, downstream, label
from .bridge import HIERARCHY
from .common import default_out_dir, load_setup

logger = logging.getLogger("v3.control")

RUN = "probe20k_chexpert_hier"
RUNS = ("probe20k_chexpert", RUN, "probe20k_jev_hard", "probe20k_jev_soft")
PAIRS = (("probe20k_jev_soft", RUN), ("probe20k_jev_hard", RUN), (RUN, "probe20k_chexpert"),
         ("probe20k_jev_soft", "probe20k_chexpert"))


def hierarchical(raw: np.ndarray) -> np.ndarray:
    """Raw labels (1 / 0 / -1 / blank) with each parent set to 1 wherever a child is 1; otherwise unchanged."""
    out = np.asarray(raw, dtype=np.float64).copy()
    for parent, children in HIERARCHY.items():
        child = (out[:, [LABELS.index(c) for c in children]] == 1.0).any(axis=1)
        out[child, LABELS.index(parent)] = 1.0
    return out


def fit_and_apply(out_dir: Path, model: str, bench_emb: Path, ext_emb: Path, chex: Path, n_train: int,
                  n_jobs: int = -1) -> dict:
    d = downstream.training_data(out_dir, model, bench_emb, n_train)
    raw_tr, raw_va = hierarchical(label_matrix(d["tr_rows"])), hierarchical(label_matrix(d["va_fit_rows"]))
    probe = fit_probe(d["X_tr"], raw_tr, d["X_va_fit"], raw_va, n_jobs=n_jobs)
    meta = {"kind": "probe", "milestone": "v3.7", "exploratory": True, "n_train": int(len(d["tr_rows"])),
            "targets": "the CheXpert labeler's labels with the hierarchy applied", "Cs": list(DEFAULT_CS),
            "per_label": probe.summary()}
    write_predictions(out_dir, RUN, d["rows"]["study_id"], d["rows"]["split"], probe.predict(d["X_all"]), meta)
    ext_ids, ext_X, ext_meta = load_npz(ext_emb)
    if (d["meta"].get("encoder"), d["meta"].get("size")) != (ext_meta.get("encoder"), ext_meta.get("size")):
        raise SystemExit("the external embeddings don't match the benchmark's encoder and size")
    ext = load_manifest(chex).drop_duplicates("study_id")
    pos = pd.Series(np.arange(len(ext_ids)), index=ext_ids).reindex(ext["study_id"])
    if pos.isna().any():
        raise SystemExit(f"{int(pos.isna().sum())} external studies have no embedding")
    write_predictions(chex, RUN, ext["study_id"], ext["split"], probe.predict(ext_X[pos.to_numpy(np.int64)]), meta)
    rates = {p: float(np.mean(raw_tr[(raw_tr[:, [LABELS.index(c) for c in kids]] == 1.0).any(axis=1),
                                     LABELS.index(p)] == 1.0)) for p, kids in HIERARCHY.items()}
    return {"n_train": int(len(d["tr_rows"])), "train_rate_after_rule": rates,
            "C": {f.label: f.C for f in probe.fits}}


def score(where: Path, manifest: pd.DataFrame, frozen: dict, n_boot: int) -> dict:
    preds = {run: read_predictions(Path(where) / "predictions" / f"{run}.parquet")[0] for run in RUNS}
    auc = {run: evaluate_set(p, manifest, frozen, "test_clean", "v2", n_boot) for run, p in preds.items()}
    cmp = {f"{a}:{b}": compare_runs(where, a, b, "test_clean", "v2", n_boot, write=False, manifest=manifest,
                                    frozen=frozen) for a, b in PAIRS}
    return {"auc": auc, "cmp": cmp}


def build_report(out_dir: Path, chex: Path, labels_path: Path, fitted: dict, n_boot: int = 1000) -> dict:
    c_manifest = load_manifest(chex)
    views = {"CheXpert test, measurable classes": score(chex, c_manifest, load_measurable(chex), n_boot),
             "CheXpert test, five standard findings": score(chex, c_manifest, {"test_clean:v2": list(FIVE)}, n_boot)}
    _, reader, frozen_r, info = label.radiologist_set(out_dir, labels_path)
    views["MIMIC-CXR, M7a's radiologist"] = score(out_dir, reader, frozen_r, n_boot)
    primary = views["CheXpert test, measurable classes"]["cmp"][f"probe20k_jev_soft:{RUN}"]
    verdict = ("n/a: no measurable classes" if not np.isfinite(primary["diff"]) else
               "Jev's labels add something beyond the hierarchy (the CI is above 0)" if primary["a_better"] else
               "the hierarchy rule recovers the L3 gain and more: the gain is the definition" if primary["b_better"] else
               "the hierarchy rule recovers the L3 gain: no detectable difference left between Jev and the rule")
    return {"fitted": fitted, "views": views, "primary": primary, "verdict": verdict, "info": info}


def write_report(out_dir: Path, r: dict) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    p, f = r["primary"], r["fitted"]
    lines = ["# v3.7 (exploratory): the hierarchy control", "",
             f"`{RUN}` is v3.5's CheXpert-labeled probe ({f['n_train']:,} training studies, the same features, C grid "
             "and selection) with one change: before fitting, Lung Opacity is set positive wherever Atelectasis, "
             "Consolidation, Edema, Lung Lesion or Pneumonia is, and Enlarged Cardiomediastinum wherever Cardiomegaly "
             "is, in the CheXpert labeler's labels (docs/V3_PLAN.md §8g). No Jev requests. After the rule, P(parent | "
             "a child positive) on the training studies is " + ", ".join(f"{k} {v:.3f}" for k, v in
                                                                        f["train_rate_after_rule"].items()) + ".", "",
             f"**Primary:** `probe20k_jev_soft` minus `{RUN}` on the CheXpert test set (measurable classes) "
             f"{fmt_ci(p['diff'], p['diff_ci'])} -> **{r['verdict']}**", ""]
    for view, s in r["views"].items():
        lines += [f"## {view}", "", "| Probe | Macro AUROC |", "|---|---|"]
        lines += [f"| `{run}` | {auc(s['auc'][run])} |" for run in RUNS]
        lines += ["", "| Comparison | Paired difference |", "|---|---|"]
        lines += [f"| `{a}` minus `{b}` | {fmt_ci(s['cmp'][f'{a}:{b}']['diff'], s['cmp'][f'{a}:{b}']['diff_ci'])} |"
                  for a, b in PAIRS]
        lines.append("")
    lines += ["## Per finding, CheXpert test set", "", "| Finding | Positives | " + " | ".join(f"`{run}`" for run in RUNS)
              + " |", "|---|---|" + "---|" * len(RUNS)]
    s = r["views"]["CheXpert test, measurable classes"]["auc"]
    per = {run: {c["label"]: c for c in s[run]["per_class"]} for run in RUNS}
    for name in LABELS:
        if per[RUNS[0]][name]["measurable"] or name in FIVE:
            lines.append(f"| {name} | {per[RUNS[0]][name]['n_pos']:,} | "
                         + " | ".join(fmt(per[run][name]["auroc"]) for run in RUNS) + " |")
    path = Path(out_dir) / "reports" / "v3_control.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str))
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="The Jev version whose cached labels v3.5 used")
    parser.add_argument("--labels", help="the radiologist's labels (default: MIMIC_CXR_JPG_PATH/...test-set-labeled.csv)")
    parser.add_argument("--ext", help="M7b's CheXpert output dir (default: EXT_OUT or $OUTPUT_PATH/v2_chexpert)")
    parser.add_argument("--embeddings", help="the benchmark's embeddings (default: BENCH_OUT's rad-dino_518.npz)")
    parser.add_argument("--ext-embeddings", help="the external embeddings (default: EXT_OUT's rad-dino_518.npz)")
    parser.add_argument("--n-train", type=int, default=downstream.N_TRAIN)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    ext = Path(args.ext) if args.ext else _env_dir("EXT_OUT", "v2_chexpert")
    bench_emb = Path(args.embeddings) if args.embeddings else (
        Path(load_setup(out_dir)["bench"]) / "embeddings" / "rad-dino_518.npz")
    ext_emb = Path(args.ext_embeddings) if args.ext_embeddings else ext / "embeddings" / "rad-dino_518.npz"
    labels = label.resolve_paths(None, args.labels, need_reports=False)[1]
    chex = chexpert.chexpert_dir(out_dir, ext)
    for run in RUNS[2:] + ("probe20k_chexpert",):
        for where in (out_dir, chex):
            if not (Path(where) / "predictions" / f"{run}.parquet").exists():
                raise SystemExit(f"{where}/predictions/{run}.parquet not found: run v3.downstream and v3.chexpert first")
    fitted = fit_and_apply(out_dir, args.model, bench_emb, ext_emb, chex, args.n_train, args.n_jobs)
    result = build_report(out_dir, chex, labels, fitted, args.n_boot)
    path = write_report(out_dir, result)
    logger.info("Jev soft - %s (CheXpert test): %s -> %s", RUN, fmt_ci(result["primary"]["diff"],
                                                                      result["primary"]["diff_ci"]), result["verdict"])
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
