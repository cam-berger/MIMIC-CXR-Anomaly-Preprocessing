"""
v3.6 (docs/V3_PLAN.md §8f): the v3.5 probes on the CheXpert test set, a second radiologist-labeled judge. No Jev
requests, and nothing is trained or chosen on this set.

    python -m v3.chexpert --model MODEL          # -> V3_OUT/reports/v3_chexpert.md

The three probes are refit exactly as v3.5 fit them (deterministic: the refit must reproduce their stored
predictions to within REPRODUCE_TOL), applied to v2 M7b's RAD-DINO embeddings of the CheXpert test images (EXT_OUT),
and scored by v2.evaluate in V3_OUT/chexpert/, which links EXT_OUT's manifest, frozen classes and audit. Nothing is
written into EXT_OUT.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v2.common import LABELS, load_manifest, load_measurable, read_predictions, setup_logging, write_predictions
from v2.embed import load_npz
from v2.evaluate import compare_runs, evaluate_set, fmt, fmt_ci
from v2.external import FIVE, _env_dir

from . import downstream
from .common import default_out_dir, load_setup
from .setup import _link

logger = logging.getLogger("v3.chexpert")

REPRODUCE_TOL = 1e-6
LINKS = ("manifest.parquet", "measurable_classes.json", "audit.json")
REFERENCE = downstream.REFERENCE  # M7b applied v2's full-data probe to this set


def chexpert_dir(out_dir: Path, ext: Path) -> Path:
    """
    V3_OUT/chexpert: links to EXT_OUT's manifest, frozen classes and audit (and M7b's reference probe), and to its
    images when they are there (v3.8 applies fine-tuned checkpoints to them).
    """
    ext, d = Path(ext).resolve(), Path(out_dir).absolute() / "chexpert"
    missing = [n for n in LINKS if not (ext / n).exists()]
    if missing:
        raise SystemExit(f"{ext} lacks {', '.join(missing)}: run v2's M7b (`run.sh m7b`) first")
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    for name in LINKS:
        _link(ext / name, d / name)
    if (ext / "images").exists():
        _link(ext / "images", d / "images")
    for suffix in (".parquet", ".json"):
        src = ext / "predictions" / f"{REFERENCE}{suffix}"
        if src.exists():
            _link(src, d / "predictions" / f"{REFERENCE}{suffix}")
    return d


def apply(out_dir: Path, model: str, bench_emb: Path, ext_emb: Path, chex: Path, n_train: int = downstream.N_TRAIN,
          n_jobs: int = -1) -> dict:
    """Refit, check against the stored v3.5 predictions, and write each probe's scores on the external studies."""
    probes, rows, X_all, facts, meta = downstream.fit_probes(out_dir, model, bench_emb, n_train, n_jobs)
    ext_ids, ext_X, ext_meta = load_npz(ext_emb)
    if (meta.get("encoder"), meta.get("size")) != (ext_meta.get("encoder"), ext_meta.get("size")):
        raise SystemExit(f"the external embeddings are {ext_meta.get('encoder')} {ext_meta.get('size')}, the probes' "
                         f"{meta.get('encoder')} {meta.get('size')}")
    ext = load_manifest(chex).drop_duplicates("study_id")
    pos = pd.Series(np.arange(len(ext_ids)), index=ext_ids).reindex(ext["study_id"])
    if pos.isna().any():
        raise SystemExit(f"{int(pos.isna().sum())} external studies have no embedding")
    X_ext = ext_X[pos.to_numpy(np.int64)]
    diffs = {}
    for run, probe in probes.items():
        stored, _ = read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")
        stored = stored.set_index("study_id").loc[rows["study_id"], list(LABELS)].to_numpy(np.float64)
        diffs[run] = float(np.max(np.abs(probe.predict(X_all) - stored)))
        if not diffs[run] <= REPRODUCE_TOL:
            raise SystemExit(f"RED FLAG: the refit {run} differs from its v3.5 predictions by up to {diffs[run]:.2e}: "
                             "stop (§8f)")
        write_predictions(chex, run, ext["study_id"], ext["split"], probe.predict(X_ext), {
            "kind": "probe", "milestone": "v3.6", "source_run": run, "targets": downstream.ARMS[run],
            "n_train": facts["n_train"], "reproduced_max_diff": diffs[run], "per_label": probe.summary()})
    logger.info("Refit probes reproduce v3.5's predictions: largest difference %.1e", max(diffs.values()))
    return {"reproduced_max_diff": diffs, "n_train": facts["n_train"], "n_external": int(len(ext))}


def build_report(chex: Path, applied: dict, n_boot: int = 1000) -> dict:
    manifest, frozen = load_manifest(chex), load_measurable(chex)
    five = {"test_clean:v2": list(FIVE)}
    runs = [*downstream.ARMS] + ([REFERENCE] if (Path(chex) / "predictions" / f"{REFERENCE}.parquet").exists() else [])
    preds = {run: read_predictions(Path(chex) / "predictions" / f"{run}.parquet")[0] for run in runs}
    scored = {run: {"measurable": evaluate_set(p, manifest, frozen, "test_clean", "v2", n_boot),
                    "five": evaluate_set(p, manifest, five, "test_clean", "v2", n_boot)} for run, p in preds.items()}
    cmp = {f"{run}:{which}": compare_runs(chex, run, "probe20k_chexpert", "test_clean", "v2", n_boot, write=False,
                                          manifest=manifest, frozen=frozen if which == "measurable" else five)
           for run in runs if run != "probe20k_chexpert" for which in ("measurable", "five")}
    l3 = cmp["probe20k_jev_soft:measurable"]
    verdict = ("n/a: no measurable classes" if not np.isfinite(l3["diff"]) else
               "PASS: Jev's soft labels train a better probe on the CheXpert test set" if l3["a_better"] else
               "FAIL: the CheXpert labeler's labels train a better probe on the CheXpert test set" if l3["b_better"] else
               "no detectable difference between the label sources (the CI includes 0)")
    return {"applied": applied, "runs": runs, "scored": scored, "cmp": cmp, "l3": l3, "verdict": verdict,
            "classes": frozen["test_clean:v2"], "n_studies": int(manifest["study_id"].nunique())}


def write_report(out_dir: Path, r: dict, l2: Optional[dict]) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    def diff(key):
        c = r["cmp"].get(key)
        return fmt_ci(c["diff"], c["diff_ci"]) if c else "-"

    l3 = r["l3"]
    role = {**downstream.ARMS, REFERENCE: "reference: all 213k training studies, CheXpert labels (v2, M7b)"}
    lines = ["# v3.6: the v3.5 probes on the CheXpert test set", "",
             f"{r['n_studies']} Stanford studies, labels from the majority vote of 5 radiologists reading the images "
             "(v2 M7b). The three v3.5 probes were refit exactly as v3.5 fit them (largest difference from their "
             f"stored predictions {max(r['applied']['reproduced_max_diff'].values()):.1e}) and applied to RAD-DINO "
             "518 px embeddings of these images; nothing was trained or chosen here (docs/V3_PLAN.md §8f). "
             "Caveat: RAD-DINO's encoder saw unlabeled CheXpert images in pretraining, alike for every probe.", "",
             f"**Gate L3:** `probe20k_jev_soft` minus `probe20k_chexpert` {fmt_ci(l3['diff'], l3['diff_ci'])} "
             f"({len(l3['classes'])} measurable classes, v2 policy) -> **{r['verdict']}**", ""]
    if l2:
        lines += [f"Read with L2 (the 596 MIMIC-CXR studies a radiologist labeled): "
                  f"{fmt_ci(l2['diff'], l2['diff_ci'])}.", ""]
    lines += ["| Probe | Targets | Measurable classes | Five standard findings | Minus `probe20k_chexpert` "
              "(measurable) | (five) |", "|---|---|---|---|---|---|"]
    for run in r["runs"]:
        s = r["scored"][run]
        lines.append(f"| `{run}` | {role[run]} | {auc(s['measurable'])} | {auc(s['five'])} | "
                     f"{diff(f'{run}:measurable')} | {diff(f'{run}:five')} |")
    lines += ["", f"Measurable classes: {', '.join(r['classes'])}. Five standard findings: {', '.join(FIVE)}.", "",
              "## Per finding", "", "| Finding | Positives | " + " | ".join(f"`{run}`" for run in r["runs"]) + " |",
              "|---|---|" + "---|" * len(r["runs"])]
    per = {run: {c["label"]: c for c in r["scored"][run]["measurable"]["per_class"]} for run in r["runs"]}
    for name in LABELS:
        if name in r["classes"] or name in FIVE:
            lines.append(f"| {name} | {per[r['runs'][0]][name]['n_pos']:,} | "
                         + " | ".join(fmt(per[run][name]["auroc"]) for run in r["runs"]) + " |")
    path = Path(out_dir) / "reports" / "v3_chexpert.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str))
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="The Jev version whose cached labels v3.5 used")
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
    for p in (bench_emb, ext_emb):
        if not p.exists():
            raise SystemExit(f"{p} not found")
    chex = chexpert_dir(out_dir, ext)
    applied = apply(out_dir, args.model, bench_emb, ext_emb, chex, args.n_train, args.n_jobs)
    result = build_report(chex, applied, args.n_boot)
    l2_path = Path(out_dir) / "reports" / "v3_downstream.json"
    l2 = json.loads(l2_path.read_text()).get("l2") if l2_path.exists() else None
    path = write_report(out_dir, result, l2)
    logger.info("L3: %s -> %s", fmt_ci(result["l3"]["diff"], result["l3"]["diff_ci"]), result["verdict"])
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
