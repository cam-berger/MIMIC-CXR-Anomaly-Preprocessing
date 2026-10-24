"""
M5: our runs next to published MIMIC-CXR results, scored the same way (docs/ANALYSIS.md, M5).

    python -m v2.lineup --out output/v2_bench

Needs an output directory prepared with --population all --split-scheme official,
where test_clean is the official test split. Every run there is scored on that
split under "positive vs rest" (the u0 policy, the published convention) and
under v2's policy, over the classes measurable in it. Published results enter
through their per-class tables, averaged over the same classes, and are read by
the rule fixed in M5: below (under our 95% CI), level (inside it) or above.
Writes reports/lineup.md; only aggregates are logged.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import LABELS, default_out_dir, load_manifest, load_measurable, read_predictions, setup_logging
from .evaluate import _get, evaluate_run, evaluate_set, fmt, fmt_ci, split_scheme
from .fuse import ARMS, text_flags

logger = logging.getLogger("v2.lineup")


@dataclass(frozen=True)
class Published:
    name: str
    inputs: str
    role: Optional[str]      # the like-for-like role it is compared in; None: context only (another protocol)
    per_class: dict          # AUROC per class, for those of our 12 classes the paper reports
    macro: float             # the paper's own macro AUROC
    macro_over: str          # what that macro averages
    protocol: str
    source: str


def _per_class(values) -> dict:
    return dict(zip(LABELS, values))


# Transcribed from the papers. MMBT and its baselines: Jacenków, O'Neil and Tsaftaris, ISBI 2022 (arXiv 2202.06076),
# Table 3, in LABELS order; Table 1 gives the macros. CheXclusion: Seyyed-Kalantari et al., PSB 2021
# (arXiv 2003.00827), Table 2, column CXR ("Airspace Opacity" is Lung Opacity).
MMBT_PROTOCOL = "official split; frontal studies with an indication or history field; positive vs rest; 224 px; 3 runs"
MMBT_SOURCE = "Jacenków et al., ISBI 2022, Table 3"
PUBLISHED = (
    Published("ResNet-50", "image", "image", _per_class(
        (0.726, 0.746, 0.699, 0.816, 0.633, 0.633, 0.665, 0.683, 0.861, 0.790, 0.661, 0.783)),
        0.738, "14 labels", MMBT_PROTOCOL, MMBT_SOURCE),
    Published("BERT", "indication", "indication", _per_class(
        (0.665, 0.741, 0.668, 0.707, 0.690, 0.659, 0.699, 0.625, 0.716, 0.723, 0.689, 0.865)),
        0.711, "14 labels (its per-class table averages 0.719)", MMBT_PROTOCOL, MMBT_SOURCE),
    Published("Attentive", "image + indication", "image + indication", _per_class(
        (0.736, 0.773, 0.734, 0.819, 0.666, 0.670, 0.709, 0.692, 0.871, 0.786, 0.709, 0.851)),
        0.766, "14 labels", MMBT_PROTOCOL, MMBT_SOURCE),
    Published("MMBT", "image + indication", "image + indication", _per_class(
        (0.758, 0.826, 0.771, 0.843, 0.743, 0.729, 0.759, 0.715, 0.886, 0.869, 0.752, 0.880)),
        0.806, "14 labels", MMBT_PROTOCOL, MMBT_SOURCE),
    Published("CheXclusion DenseNet-121", "image", None, _per_class(
        (0.837, 0.828, 0.844, 0.904, 0.757, 0.718, 0.772, 0.782, 0.933, 0.848, 0.748, 0.903)),
        0.834, "14 labels", "random 80/10/10 patient split; frontal and lateral images; positive vs rest",
        "Seyyed-Kalantari et al., PSB 2021, Table 2"),
    Published("DenseNet-121 (Wang, Zhang and Oberst)", "image", None, {}, 0.803, "13 labels",
              "random 80/10/10 patient split; positive vs rest", "CHIL 2026, arXiv 2509.19671"),
    Published("MBRANet", "image", None, {}, 0.805, "average AUC", "split not verified", "Scientific Reports 2024"),
    Published("CaMCheX", "two views + indication + vitals, 1024 px", None, {}, 0.934, "14 labels",
              "split not stated; no leakage control described", "ML4H 2025, arXiv 2511.09581"),
)
ROLE_BY_ARM = {"image": "image", "image_indication": "image + indication", "indication": "indication"}


def role(meta: dict) -> Optional[str]:
    """
    What a run of ours is compared as: its inputs, if a published model had the same ones. Text on a stronger
    image model is lined up through the val combiner (a stack with the `indication` run); M4's fusion on
    fine-tuned features stays exploratory (M5b).
    """
    if meta.get("kind") in ("probe", "finetune"):
        return "image"
    if meta.get("kind") == "fuse":
        return None if meta.get("image_features") == "fine-tuned" else ROLE_BY_ARM.get(meta.get("arm"))
    if meta.get("kind") == "attend" and meta.get("arm") == "h0":
        return "image"
    if meta.get("kind") == "text_finetune":  # M6: a language model fine-tuned on one text source
        return ROLE_BY_ARM.get(meta.get("arm"))
    if meta.get("kind") == "stack" and meta.get("text_arm") and \
            role({"kind": meta.get("image_kind"), "arm": meta.get("image_arm")}) == "image":
        return ROLE_BY_ARM.get(f"image_{meta['text_arm']}")
    return None


def published_mean(ref: Published, classes: list) -> Optional[float]:
    """The reference's mean over `classes`, if it reports every one of them."""
    if not classes or not all(c in ref.per_class for c in classes):
        return None
    return float(np.mean([ref.per_class[c] for c in classes]))


def verdict(published: Optional[float], ci: list) -> str:
    """M5: below if under our 95% CI, above if over it, level (a tie) otherwise."""
    if published is None or not all(np.isfinite(ci)):
        return "-"
    if published < ci[0]:
        return "below ours"
    if published > ci[1]:
        return "**above ours**"
    return "level"


def indication_subset(out_dir: Path, manifest: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The manifest with test_clean cut to official-test studies whose indication is not empty (MMBT's test set)."""
    ctx_path = Path(out_dir) / "context.parquet"
    if not ctx_path.exists():
        return None
    ctx = pd.read_parquet(ctx_path, columns=["study_id", "indication"])
    with_ind = set(ctx.loc[ctx["indication"].fillna("").str.strip().ne(""), "study_id"])
    sub = manifest.copy()
    sub["in_test_clean"] = sub["in_test_clean"] & sub["study_id"].isin(with_ind)
    return sub


def lineup(out_dir: Path, n_boot: int = 1000, runs: Optional[list] = None) -> dict:
    if split_scheme(out_dir) != "official":
        raise SystemExit(f"{out_dir} holds the v2 cohort. The lineup needs M5's benchmark directory: "
                         "prepare_data --population all --split-scheme official --out <its own dir>")
    pred_dir = Path(out_dir) / "predictions"
    runs = runs or sorted(p.stem for p in pred_dir.glob("*.parquet") if not p.stem.endswith("_smoke"))
    if not runs:
        raise SystemExit(f"No predictions in {pred_dir}")
    manifest, frozen = load_manifest(out_dir), load_measurable(out_dir)
    if "test_clean:u0" not in frozen:
        raise SystemExit("measurable_classes.json has no test_clean:u0: re-run prepare_data at this commit")
    sub = indication_subset(out_dir, manifest)
    reps = {run: evaluate_run(out_dir, run, n_boot, manifest, frozen) for run in runs}
    rows = {}
    for run, rep in reps.items():
        row = {"meta": rep["meta"], "role": role(rep["meta"]), "u0": _get(rep, "test_clean", "u0"),
               "v2": _get(rep, "test_clean", "v2"), "val": _get(rep, "val", "v2"), "subset": None}
        if sub is not None and row["role"]:
            pred, _ = read_predictions(pred_dir / f"{run}.parquet")
            row["subset"] = evaluate_set(pred, sub, frozen, "test_clean", "u0", n_boot)
        rows[run] = row

    # M4's red flags on every text-only run come first: a flagged lineup isn't read (M5, M6).
    flags = []
    images = {r[: -len("_ctx_image")]: rows[r]["v2"] for r in rows if r.endswith("_ctx_image") and rows[r]["v2"]}
    # A fine-tuned text run belongs to no embeddings: it is held to the weakest `image` arm, the strictest test.
    weakest = min(images.values(), key=lambda res: res["macro_auroc"], default=None)
    for run, row in rows.items():
        kind, meta = row["meta"].get("kind"), row["meta"]
        if (kind == "fuse" and not meta.get("image") and meta.get("arm") in ARMS) or kind == "text_finetune":
            ref = images.get(run.split("_ctx_")[0]) if "_ctx_" in run else weakest
            flags += text_flags(run, row["v2"], ref)

    audit = json.loads((Path(out_dir) / "audit.json").read_text())
    test = audit["sizes"]["test_clean"]
    lines = ["# M5: our runs next to published MIMIC-CXR results", "",
             f"Official test split: {test['studies']:,} studies, {test['patients']:,} patients "
             f"(population: {audit.get('population')}). Macro AUROC [95% patient-bootstrap CI] over the classes "
             "measurable there, under positive vs rest (u0, the published convention) unless noted. Published "
             "numbers are averaged over the same classes from their per-class tables (docs/ANALYSIS.md, M5).", ""]
    lines += ["## Red flags (read these first)", ""]
    lines += [f"- **RED FLAG:** {f}. Don't read the lineup until the extraction is inspected." for f in flags] or \
        ["- none"]

    lines += ["", "## Our runs", "", "| Run | Compared as | val (v2) | Official test, positive vs rest | "
              "Official test, v2 policy | Studies with an indication, positive vs rest |", "|---|---|---|---|---|---|"]

    def cell(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    for run, row in rows.items():
        lines.append(f"| `{run}` | {row['role'] or '-'} | {cell(row['val'])} | {cell(row['u0'])} | "
                     f"{cell(row['v2'])} | {cell(row['subset'])} |")

    lines += ["", "## Like for like (same protocol, same inputs)", "",
              "| Ours | Published | Inputs | Classes | Ours | Published | Published is | "
              "Ours, indication subset | Published is |", "|---|---|---|---|---|---|---|---|---|"]
    compared = []
    for run, row in rows.items():
        if not row["role"] or not row["u0"]:
            continue
        classes = row["u0"]["classes"]
        for ref in PUBLISHED:
            if ref.role != row["role"]:
                continue
            pub = published_mean(ref, classes)
            ours, ci = row["u0"]["macro_auroc"], row["u0"]["macro_auroc_ci"]
            sub_res = row["subset"]
            sub_cell = cell(sub_res) if sub_res else "-"
            sub_verdict = verdict(pub, sub_res["macro_auroc_ci"]) if sub_res else "-"
            lines.append(f"| `{run}` | {ref.name} | {ref.inputs} | {len(classes)} | {fmt_ci(ours, ci)} | {fmt(pub)} | "
                         f"{verdict(pub, ci)} | {sub_cell} | {sub_verdict} |")
            compared.append({"run": run, "published": ref.name, "ours": ours, "ci": ci, "published_mean": pub,
                             "verdict": verdict(pub, ci)})
    if not compared:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    best = {}
    for run, row in rows.items():
        score = (row["val"] or {}).get("macro_auroc", float("nan"))
        if row["role"] in ("image", "image + indication") and row["u0"] and np.isfinite(score):
            if row["role"] not in best or score > best[row["role"]][1]:
                best[row["role"]] = (run, score)
    if best:
        refs = [r for r in PUBLISHED if r.per_class]
        lines += ["", "## Per class, official test, positive vs rest (AUROC)", "",
                  "Our best run by val for each role, next to every published per-class table.", "",
                  "| Class | " + " | ".join(f"`{run}`" for run, _ in best.values()) + " | "
                  + " | ".join(r.name for r in refs) + " |", "|---|" + "---|" * (len(best) + len(refs))]
        per = {run: {c["label"]: c["auroc"] for c in rows[run]["u0"]["per_class"]} for run, _ in best.values()}
        for label in LABELS:
            ours = " | ".join(fmt(per[run][label]) for run, _ in best.values())
            pubs = " | ".join(fmt(r.per_class.get(label, float("nan"))) for r in refs)
            lines.append(f"| {label} | {ours} | {pubs} |")

    lines += ["", "## Published results", "",
              "| Model | Inputs | Paper's macro AUROC | Over | Compared like for like | Protocol | Source |",
              "|---|---|---|---|---|---|---|"]
    for ref in PUBLISHED:
        lines.append(f"| {ref.name} | {ref.inputs} | {fmt(ref.macro)} | {ref.macro_over} | "
                     f"{'yes' if ref.role else 'no: context only'} | {ref.protocol} | {ref.source} |")
    lines += ["", "Caveats (M5): 12 of the 14 CheXpert labels (No Finding and Support Devices are not v2 classes); "
              "our labels come from the MIMIC-CXR-JPG CheXpert file, where MMBT ran the labeler on full reports; one "
              "frontal image per study; published numbers carry their own uncertainty, so *level* is a tie."]
    path = Path(out_dir) / "reports" / "lineup.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return {"path": path, "flags": flags, "compared": compared, "rows": rows}


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="M5's benchmark output dir")
    parser.add_argument("--runs", nargs="+", help="Runs to line up (default: every run in the directory)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    res = lineup(out_dir, args.n_boot, args.runs)
    for flag in res["flags"]:
        logger.warning("RED FLAG: %s", flag)
    for c in res["compared"]:
        logger.info("%s vs %s: %s vs %s -> published is %s", c["run"], c["published"],
                    fmt_ci(c["ours"], c["ci"]), fmt(c["published_mean"]), c["verdict"].strip("*"))
    logger.info("Report: %s", res["path"])
    return res


if __name__ == "__main__":
    main()
