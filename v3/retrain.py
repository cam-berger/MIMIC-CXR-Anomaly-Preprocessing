"""
v3.8 (docs/V3_PLAN.md §8h): Jev's report labels on the whole training split, as targets for the main model, M5b's
fine-tuned RAD-DINO. Three arms differ only in their targets (the CheXpert labeler, the labeler with v3.7's hierarchy
rule, Jev's soft labels), each with seeds 0 and 1:

    python -m v3.label run --split train --all --model MODEL     # the labels (sends report text to Jev; resumable)
    python -m v3.retrain targets --model MODEL                   # Mac: retrain/targets.parquet (record-level: local)
    python -m v3.retrain gpu --bench output/v2_bench             # GPU box: a smoke run, then the six fine-tunes
    python -m v3.retrain apply                                    # Mac: each checkpoint on the CheXpert test images
    python -m v3.retrain report --labels output/mimic-cxr-2.1.0-test-set-labeled.csv   # -> reports/v3_retrain.md

v2.finetune runs unchanged, on a copy of the benchmark's manifest whose label columns hold the arm's targets. A value
strictly between 0 and 1 (Jev's soft targets) passes v2's label policy as it is, with mask 1; every other value
follows the v2 policy. Runs are ft213k_{arm}_s{seed}, in V3_OUT/runs/ and V3_OUT/predictions/.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v2.common import LABELS, TEST, TRAIN, VAL, label_matrix, load_manifest, load_measurable, read_predictions
from v2.common import setup_logging, targets_and_mask
from v2.evaluate import compare_runs, evaluate_set, fmt, fmt_ci
from v2.external import FIVE, _env_dir

from . import chexpert, downstream, label
from .common import default_bench_dir, default_out_dir, load_setup
from .control import hierarchical
from .run import load_cache
from .setup import _link

logger = logging.getLogger("v3.retrain")

ARMS = {"jev_soft": "Jev's soft labels, P(present) + 0.5 P(uncertain)",
        "chexpert_hier": "the CheXpert labeler with v3.7's hierarchy rule",
        "chexpert": "the CheXpert labeler (the current model's labels)"}
SEEDS = (0, 1)
PREFIX = "ft213k"
# M5b's rad-dino_518_ft_attn recipe (v2/scripts/run.sh bench_ft): only the targets differ.
RECIPE = ["--model", "rad-dino", "--size", "518", "--head", "attention", "--epochs", "3", "--batch-size", "16",
          "--accum", "2", "--max-hours", "3", "--milestone", "v3.8"]
RECIPE_CONFIG = {"model": "rad-dino", "size": 518, "head": "attention", "grid": 12, "epochs": 3, "encoder_lr": 2e-5,
                 "head_lr": 1e-3, "llrd": 0.75, "weight_decay": 0.05, "warmup_epochs": 1.0, "clip": 1.0}
EFFECTIVE_BATCH = 32
REFERENCE = "rad-dino_518_ft_attn_s{seed}"  # v2's runs of the same recipe: M5 (seed 0), M8 (seed 1)
RERUN_TOL = 0.005
MAX_UNANSWERED = 0.01
GATES = {"L4": ("chexpert", "jev_soft", "chexpert_hier"), "L5": ("radiologist", "jev_soft", "chexpert")}
PAIRS = (("jev_soft", "chexpert_hier"), ("jev_soft", "chexpert"), ("chexpert_hier", "chexpert"))
JUDGES = {"chexpert": "CheXpert test, measurable classes", "five": "CheXpert test, five standard findings",
          "radiologist": "MIMIC-CXR, M7a's radiologist (596 official-test studies)",
          "labeler": "Official test against the CheXpert labeler's labels (favours `ft213k_chexpert`)"}
GPU_LINKS = ("manifest.parquet", "measurable_classes.json", "audit.json", "images")


def run_name(arm: str, seed: int, smoke: bool = False) -> str:
    return f"{PREFIX}_{arm}_s{seed}" + ("_smoke" if smoke else "")


def targets_path(out_dir: Path) -> Path:
    return Path(out_dir) / "retrain" / "targets.parquet"


def _cols(kind: str) -> list[str]:
    return [f"{kind}:{c}" for c in LABELS]


# --- the targets (Mac) ------------------------------------------------------------------------------------------------

def build_targets(out_dir: Path, model: str) -> dict:
    """
    Jev's soft targets on every training study and its answers on val, from v3.label's cache. More than 1% of either
    split unanswered stops the step (§8h); the unanswered are left out of every arm. Returns aggregates only.
    """
    manifest = load_manifest(out_dir).drop_duplicates("study_id")
    cache = load_cache(label.cache_path(out_dir, model))
    models = {rec["model"] for rec in cache.values()}
    if len(models) > 1:
        raise SystemExit("labeler answers come from more than one model: §3 pins one")
    parts, facts = [], {"model": model, "answered_by": sorted(models), "label_prompt_hash": label.label_hash()}
    for split in (TRAIN, VAL):
        ids = np.sort(manifest.loc[manifest["split"] == split, "study_id"].to_numpy())
        have = np.array([int(s) in cache for s in ids], dtype=bool)
        missing = int((~have).sum())
        if missing > MAX_UNANSWERED * len(ids):
            raise SystemExit(f"{missing} of {len(ids)} {split} studies have no Jev label: run "
                             f"`python -m v3.label run --split {split}{' --all' if split == TRAIN else ''}` again (§8h)")
        keep = ids[have]
        raw, soft = downstream.jev_targets(cache, keep)
        df = pd.DataFrame({"study_id": keep, "split": split})
        df[_cols("soft")], df[_cols("jev")] = soft, raw
        parts.append(df)
        facts[split] = {"studies": int(len(ids)), "answered": int(len(keep)), "unanswered": missing,
                        "empty_reports": int(sum(bool(cache[int(s)].get("empty_report")) for s in keep))}
    targets = pd.concat(parts, ignore_index=True)
    path = targets_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(path, index=False)
    facts["sha256"] = _sha(path)
    facts["shares"] = target_shares(manifest, targets)
    path.with_suffix(".json").write_text(json.dumps(facts, indent=2))
    logger.info("Targets: %d training and %d val studies (%d and %d without a Jev label) -> %s",
                facts[TRAIN]["answered"], facts[VAL]["answered"], facts[TRAIN]["unanswered"],
                facts[VAL]["unanswered"], path)
    return facts


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_targets(out_dir: Path) -> pd.DataFrame:
    path = targets_path(out_dir)
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m v3.retrain targets` on the Mac and bring it along (§8h)")
    return pd.read_parquet(path)


def target_shares(manifest: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Per finding, each arm's mean training target (v2 policy where it applies): aggregates for the report."""
    tr = targets[targets["split"] == TRAIN]
    rows = manifest.set_index("study_id").loc[tr["study_id"]].reset_index()
    out = {}
    for arm in ARMS:
        y, m = soft_targets_and_mask(arm_raw(rows, tr, arm, TRAIN))
        out[arm] = {c: round(float((y[:, k] * m[:, k]).sum() / max(m[:, k].sum(), 1)), 4) for k, c in enumerate(LABELS)}
    return out


def write_targets_report(out_dir: Path, facts: dict) -> Path:
    path = Path(out_dir) / "reports" / "v3_retrain_targets.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tr, va = facts[TRAIN], facts[VAL]
    lines = ["# v3.8: the training targets", "",
             f"Aggregates only (docs/V3_PLAN.md §8h). Jev `{facts['model']}` (answered by "
             f"{', '.join(f'`{m}`' for m in facts['answered_by'])}), labeler prompt `{facts['label_prompt_hash']}`; "
             f"targets file `{facts['sha256']}`.", "",
             "| Split | Studies | With a Jev label | Without (left out of every arm) | Empty reports |", "|---|---|---|---|---|",
             f"| train | {tr['studies']:,} | {tr['answered']:,} | {tr['unanswered']:,} | {tr['empty_reports']:,} |",
             f"| val | {va['studies']:,} | {va['answered']:,} | {va['unanswered']:,} | {va['empty_reports']:,} |", "",
             "## Mean training target per finding", "", "| Finding | " + " | ".join(f"`{a}`" for a in ARMS) + " |",
             "|---|" + "---|" * len(ARMS)]
    for c in LABELS:
        lines.append(f"| {c} | " + " | ".join(f"{facts['shares'][a][c]:.3f}" for a in ARMS) + " |")
    path.write_text("\n".join(lines) + "\n")
    return path


# --- the arms (GPU) ---------------------------------------------------------------------------------------------------

def arm_raw(rows: pd.DataFrame, targets: pd.DataFrame, arm: str, split: str) -> np.ndarray:
    """An arm's label values for manifest `rows` of one split: soft targets on train, hard labels on val."""
    if arm == "chexpert":
        return label_matrix(rows)
    if arm == "chexpert_hier":
        return hierarchical(label_matrix(rows))
    if arm == "jev_soft":
        t = targets.set_index("study_id")
        return t.loc[rows["study_id"], _cols("soft" if split == TRAIN else "jev")].to_numpy(np.float64)
    raise ValueError(f"unknown arm {arm!r} (expected one of {', '.join(ARMS)})")


def arm_manifest(manifest: pd.DataFrame, targets: pd.DataFrame, arm: str) -> pd.DataFrame:
    """The benchmark's manifest with train and val cut to the studies in `targets` and labeled by the arm."""
    parts = []
    for split in (TRAIN, VAL):
        ids = targets.loc[targets["split"] == split, "study_id"]
        rows = manifest[(manifest["split"] == split) & manifest["study_id"].isin(ids)].copy()
        rows[list(LABELS)] = arm_raw(rows, targets, arm, split)
        parts.append(rows)
    parts.append(manifest[manifest["split"] == TEST])
    return pd.concat(parts, ignore_index=True)


def soft_targets_and_mask(raw: np.ndarray, metric: str = "v2") -> tuple[np.ndarray, np.ndarray]:
    """v2's label policy, except that a value strictly between 0 and 1 is a soft target: kept as it is, mask 1."""
    y, m = targets_and_mask(raw, metric)
    raw = np.asarray(raw, dtype=np.float64)
    soft = (raw > 0.0) & (raw < 1.0)
    if not soft.any():
        return y, m
    if metric != "v2":
        raise ValueError("soft targets are training targets: the v2 policy only")
    return np.where(soft, raw, y).astype(np.float32), m | soft


@contextmanager
def arm_targets(ft, manifest: pd.DataFrame):
    """v2.finetune (`ft`) reads `manifest` instead of the benchmark's and takes soft targets as they are."""
    saved = ft.load_manifest, ft.targets_and_mask
    ft.load_manifest = lambda out_dir: manifest.copy()
    ft.targets_and_mask = soft_targets_and_mask
    try:
        yield
    finally:
        ft.load_manifest, ft.targets_and_mask = saved


def prepare_dir(out_dir: Path, bench: Path) -> None:
    """On a GPU box: link the uploaded benchmark's manifest and images into the v3 dir. Nothing is written there."""
    out, bench = Path(out_dir).absolute(), Path(bench).resolve()
    if out.resolve() == bench:
        raise SystemExit("--out must differ from the benchmark: v3 never writes into it")
    out.mkdir(parents=True, exist_ok=True)
    for name in GPU_LINKS:
        if not (out / name).exists() and (bench / name).exists():
            _link(bench / name, out / name)
    missing = [n for n in ("manifest.parquet", "images") if not (out / n).exists()]
    if missing:
        raise SystemExit(f"{out} lacks {', '.join(missing)}: upload the benchmark's manifest and images (v2/README.md §3)")


def train(out_dir: Path, arm: str, seed: int, extra: tuple = (), smoke: bool = False) -> str:
    """One fine-tune of the arm with v2.finetune; `extra` arguments follow the recipe's (later ones win)."""
    from v2 import finetune as ft

    targets = load_targets(out_dir)
    manifest = arm_manifest(load_manifest(out_dir), targets, arm)
    run = run_name(arm, seed, smoke)
    argv = [*RECIPE, "--seed", str(seed), "--out", str(out_dir), "--run", run, *extra]
    argv += ["--smoke"] if smoke else ["--resume"]
    logger.info("v3.8 %s: %s on %d training studies", run, ARMS[arm], int((manifest["split"] == TRAIN).sum()))
    with arm_targets(ft, manifest):
        ft.main(argv)
    (Path(out_dir) / "runs" / run / "v3_targets.json").write_text(json.dumps({
        "arm": arm, "targets": ARMS[arm], "targets_sha256": _sha(targets_path(out_dir)), "seed": seed,
        "n_train": int((manifest["split"] == TRAIN).sum()), "n_val": int((manifest["split"] == VAL).sum())}, indent=2))
    return run


def gpu(out_dir: Path, seeds: tuple = SEEDS, arms: tuple = tuple(ARMS), extra: tuple = ()) -> list[str]:
    """A smoke run on a few real studies, then each seed's arms in turn; runs with predictions are skipped."""
    load_targets(out_dir)
    train(out_dir, "jev_soft", seeds[0], extra, smoke=True)
    done = []
    for seed in seeds:
        for arm in arms:
            run = run_name(arm, seed)
            if (Path(out_dir) / "predictions" / f"{run}.parquet").exists():
                logger.info("%s already has predictions: skipped", run)
                continue
            done.append(train(out_dir, arm, seed, extra))
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    return done


# --- the CheXpert test images and the report (Mac) --------------------------------------------------------------------

def link_references(out_dir: Path, ext: Path, chex: Path) -> list[str]:
    """v2's two runs of the recipe, from the benchmark and from M7b/M8's external predictions, where they exist."""
    bench, linked = Path(load_setup(out_dir)["bench"]), []
    for seed in SEEDS:
        ref = REFERENCE.format(seed=seed)
        pairs = [(bench, Path(out_dir)), (Path(ext).resolve(), Path(chex))]
        if not all((src / "predictions" / f"{ref}.parquet").exists() for src, _ in pairs):
            continue
        for src, dst in pairs:
            for suffix in (".parquet", ".json"):
                s, d = src / "predictions" / f"{ref}{suffix}", dst / "predictions" / f"{ref}{suffix}"
                if s.exists() and not (d.exists() or d.is_symlink()):
                    _link(s, d)
        linked.append(ref)
    return linked


def all_runs() -> list[str]:
    return [run_name(arm, seed) for seed in SEEDS for arm in ARMS]


def apply(out_dir: Path, ext: Path, workers: int = 4, device: str = "auto") -> list[str]:
    """Each run's best checkpoint on the CheXpert test images (v2.external, with its reload check), into V3_OUT/chexpert."""
    from v2.external import apply_finetune

    chex = chexpert.chexpert_dir(out_dir, ext)
    if not (chex / "images").exists():
        raise SystemExit(f"{ext} has no images/: the CheXpert test images come from v2's M7b (`run.sh m7b`)")
    link_references(out_dir, ext, chex)
    runs = [r for r in all_runs() if (Path(out_dir) / "predictions" / f"{r}.parquet").exists()]
    if not runs:
        raise SystemExit("no ft213k_* predictions in V3_OUT: bring the GPU runs home first (v3/README.md)")
    done = []
    for run in runs:
        if (chex / "predictions" / f"{run}.parquet").exists():
            logger.info("%s is already applied: skipped", run)
            continue
        apply_finetune(Path(out_dir), chex, run, workers, device, milestone="v3.8")
        done.append(run)
    return done


def recipe_problems(meta: dict) -> list[str]:
    """How a run's settings differ from M5b's recipe, from its predictions' metadata."""
    cfg, out = meta.get("config", {}), []
    for key, want in RECIPE_CONFIG.items():
        got = cfg.get(key)
        if not (got == want or (isinstance(want, float) and isinstance(got, (int, float)) and np.isclose(got, want))):
            out.append(f"{key} {got} (recipe {want})")
    batch = (cfg.get("batch_size") or 0) * (cfg.get("accum") or 0)
    if batch != EFFECTIVE_BATCH:
        out.append(f"effective batch {batch} (recipe {EFFECTIVE_BATCH})")
    epochs = len(meta.get("history") or [])
    if meta.get("stopped_early") or epochs < RECIPE_CONFIG["epochs"]:
        out.append(f"{epochs} epochs done{' (stopped by --max-hours)' if meta.get('stopped_early') else ''}")
    return out


def _gate(cmps: list[dict], n_seeds: int) -> str:
    if n_seeds < len(SEEDS):
        return f"PENDING: §8h needs both seeds, {n_seeds} of {len(SEEDS)} here"
    if not all(np.isfinite(c["diff"]) for c in cmps):
        return "n/a: no measurable classes"
    if all(c["a_better"] for c in cmps):
        return "PASS"
    if all(c["b_better"] for c in cmps):
        return "FAIL"
    return "no detectable difference (a CI includes 0, or the seeds disagree)"


def reading(l4: str, l5: str) -> str:
    if l4.startswith("PENDING") or l5.startswith("PENDING"):
        return "pending: both seeds are needed before the gates are read"
    p4, p5 = l4 == "PASS", l5 == "PASS"
    if p4 and p5:
        return "Jev's labels make a better main model, on both judges and beyond the definition"
    if p4:
        return "Jev's edge is in matching CheXpert's definitions; on MIMIC the current labels stand"
    if p5:
        return "Jev's labels improve the model on MIMIC, but on CheXpert the hierarchy rule does as well"
    return "v3.7's edge doesn't hold at full scale with the main model: the definition is the lever, and it needs no Jev"


def build_report(out_dir: Path, chex: Path, labels_path: Path, n_boot: int = 1000) -> dict:
    tmeta = json.loads(targets_path(out_dir).with_suffix(".json").read_text())
    pred_dir = Path(out_dir) / "predictions"
    seeds = [s for s in SEEDS if all((pred_dir / f"{run_name(a, s)}.parquet").exists() for a in ARMS)]
    if not seeds:
        raise SystemExit("no seed has all three ft213k_* runs in V3_OUT/predictions: bring the GPU runs home first")
    missing = [run_name(a, s) for s in seeds for a in ARMS if not (Path(chex) / "predictions" /
                                                                  f"{run_name(a, s)}.parquet").exists()]
    if missing:
        raise SystemExit(f"not applied to the CheXpert test images: {', '.join(missing)}: run `python -m v3.retrain apply`")
    refs = [REFERENCE.format(seed=s) for s in seeds
            if all((d / "predictions" / f"{REFERENCE.format(seed=s)}.parquet").exists() for d in (Path(out_dir), chex))]
    runs = [run_name(a, s) for s in seeds for a in ARMS]
    c_manifest = load_manifest(chex)
    _, reader, frozen_r, info = label.radiologist_set(out_dir, labels_path)
    judges = {"chexpert": (Path(chex), c_manifest, load_measurable(chex)),
              "five": (Path(chex), c_manifest, {"test_clean:v2": list(FIVE)}),
              "radiologist": (Path(out_dir), reader, frozen_r),
              "labeler": (Path(out_dir), load_manifest(out_dir), load_measurable(out_dir))}
    auc, cmp = {}, {}
    for judge, (where, manifest, frozen) in judges.items():
        auc[judge] = {run: evaluate_set(read_predictions(where / "predictions" / f"{run}.parquet")[0], manifest, frozen,
                                        "test_clean", "v2", n_boot) for run in runs + refs}
        for seed in seeds:
            for a, b in PAIRS:
                cmp[f"{judge}:{a}:{b}:{seed}"] = compare_runs(where, run_name(a, seed), run_name(b, seed), "test_clean",
                                                              "v2", n_boot, write=False, manifest=manifest,
                                                              frozen=frozen)
    gates = {g: _gate([cmp[f"{judge}:{a}:{b}:{s}"] for s in seeds], len(seeds)) for g, (judge, a, b) in GATES.items()}
    metas = {run: read_predictions(pred_dir / f"{run}.parquet")[1] for run in runs}
    flags = [f"RECIPE: `{run}`: " + "; ".join(p) for run, m in metas.items() if (p := recipe_problems(m))]
    sha = _sha(targets_path(out_dir))
    for run in runs:
        side = Path(out_dir) / "runs" / run / "v3_targets.json"
        got = json.loads(side.read_text()).get("targets_sha256") if side.exists() else None
        if got != sha:
            flags.append(f"TARGETS: `{run}` was trained on targets {got or '(no record)'}, the local file is {sha}")
    for seed in seeds:
        ref = REFERENCE.format(seed=seed)
        if ref in refs:
            d = auc["labeler"][run_name("chexpert", seed)]["macro_auroc"] - auc["labeler"][ref]["macro_auroc"]
            if abs(d) > RERUN_TOL:
                flags.append(f"RERUN: `{run_name('chexpert', seed)}` minus v2's `{ref}` {d:+.3f} on the official test "
                             f"(CheXpert labels), beyond {RERUN_TOL}: explain before reading the gates (§8h)")
    for split in (TRAIN, VAL):
        f = tmeta[split]
        if f["unanswered"] > MAX_UNANSWERED * f["studies"]:
            flags.append(f"COVERAGE: {f['unanswered']} of {f['studies']} {split} studies without a Jev label")
    reload = {run: read_predictions(Path(chex) / "predictions" / f"{run}.parquet")[1].get("reload_check", {})
              for run in runs}
    return {"seeds": seeds, "runs": runs, "refs": refs, "info": info, "targets": tmeta, "auc": auc, "cmp": cmp,
            "gates": gates, "reading": reading(gates["L4"], gates["L5"]), "flags": flags, "reload": reload,
            "frozen": {"chexpert": judges["chexpert"][2], "radiologist": frozen_r},
            "runs_meta": {run: {"best_epoch": m.get("best_epoch"), "best_val": m.get("best_val_macro_auroc"),
                                "epochs": len(m.get("history") or []),
                                "minutes": round(sum(h.get("seconds", 0) for h in m.get("history") or []) / 60, 1),
                                "n_train": (m.get("config") or {}).get("n_train")} for run, m in metas.items()}}


def write_report(out_dir: Path, r: dict) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    def diff(key):
        c = r["cmp"].get(key)
        return fmt_ci(c["diff"], c["diff_ci"]) if c else "-"

    seeds, t = r["seeds"], r["targets"]
    head = " | ".join(f"Seed {s}" for s in seeds)
    lines = ["# v3.8: Jev's labels for the main model", "",
             f"M5b's fine-tuned RAD-DINO recipe, trained on {t[TRAIN]['answered']:,} training studies three ways that "
             f"differ only in their targets (docs/V3_PLAN.md §8h); each run kept its best epoch on {t[VAL]['answered']:,} "
             f"val studies labeled by its own source. Seeds: {', '.join(map(str, seeds))}. Paired 95% CIs from patient "
             "resamples, v2 policy.", ""]
    for g, (judge, a, b) in GATES.items():
        per_seed = "; ".join(f"seed {s} {diff(f'{judge}:{a}:{b}:{s}')}" for s in seeds)
        lines.append(f"**Gate {g}:** `ft213k_{a}` minus `ft213k_{b}` on the {JUDGES[judge].split(' (')[0]}: "
                     f"{per_seed} -> **{r['gates'][g]}**")
        lines.append("")
    lines += [f"**Reading:** {r['reading']}.", "", "## Red flags", ""] + ([f"- {f}" for f in r["flags"]] or ["- none"])
    lines.append("")
    for judge, title in JUDGES.items():
        lines += [f"## {title}", "", f"| Arm | {head} |", "|---|" + "---|" * len(seeds)]
        for arm in ARMS:
            lines.append(f"| `ft213k_{arm}` | " + " | ".join(auc(r["auc"][judge][run_name(arm, s)]) for s in seeds) + " |")
        for ref in r["refs"]:
            lines.append(f"| v2's `{ref}` | " + " | ".join(auc(r["auc"][judge][ref]) if ref.endswith(f"s{s}") else "-"
                                                        for s in seeds) + " |")
        lines += ["", f"| Paired difference | {head} |", "|---|" + "---|" * len(seeds)]
        for a, b in PAIRS:
            lines.append(f"| `{a}` minus `{b}` | " + " | ".join(diff(f"{judge}:{a}:{b}:{s}") for s in seeds) + " |")
        lines.append("")
    s0 = seeds[0]
    for judge in ("chexpert", "radiologist"):
        lines += [f"## Per finding, {JUDGES[judge].split(' (')[0]} (seed {s0})", "",
                  "| Finding | Positives | " + " | ".join(f"`{a}`" for a in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
        per = {a: {c["label"]: c for c in r["auc"][judge][run_name(a, s0)]["per_class"]} for a in ARMS}
        for name in LABELS:
            c = per["chexpert"][name]
            if c.get("measurable") or (judge == "chexpert" and name in FIVE):
                lines.append(f"| {name} | {c['n_pos']:,} | " + " | ".join(fmt(per[a][name]["auroc"]) for a in ARMS) + " |")
        lines.append("")
    lines += ["## Runs", "", "| Run | Training studies | Epochs | Best epoch | Best val macro AUROC | Minutes | "
              "Reload check (correlation) |", "|---|---|---|---|---|---|---|"]
    for run, m in r["runs_meta"].items():
        corr = r["reload"].get(run, {}).get("correlation")
        lines.append(f"| `{run}` | {m['n_train'] or 0:,} | {m['epochs']} | {m['best_epoch']} | {fmt(m['best_val'])} | "
                     f"{m['minutes']} | {fmt(corr) if corr is not None else '-'} |")
    path = Path(out_dir) / "reports" / "v3_retrain.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str))
    return path


# --- CLI --------------------------------------------------------------------------------------------------------------

def _extra(args) -> tuple:
    if (args.batch_size is None) != (args.accum is None):
        raise SystemExit("pass --batch-size and --accum together")
    if args.batch_size is not None and args.batch_size * args.accum != EFFECTIVE_BATCH:
        raise SystemExit(f"the recipe's effective batch is {EFFECTIVE_BATCH} (16 x 2, or 8 x 4 if memory forces it)")
    extra = []
    for flag in ("batch_size", "accum", "workers", "device"):
        if getattr(args, flag) is not None:
            extra += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
    return tuple(extra + (["--grad-checkpointing"] if args.grad_checkpointing else []))


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_tg = sub.add_parser("targets", help="Mac: Jev's training and val targets from v3.label's cache")
    p_tg.add_argument("--model", required=True, help="The Jev version whose cached labels to use")
    p_tr = sub.add_parser("train", help="One fine-tune (GPU)")
    p_tr.add_argument("--arm", required=True, choices=list(ARMS))
    p_tr.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    p_tr.add_argument("--smoke", action="store_true", help="v2.finetune's smoke: 64 / 32 / 32 studies, 2 epochs")
    p_gpu = sub.add_parser("gpu", help="GPU: a smoke run, then every arm for each seed (resumable)")
    p_gpu.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    for p in (p_tr, p_gpu):
        p.add_argument("--bench", help="the uploaded benchmark (default: BENCH_OUT or $OUTPUT_PATH/v2_bench)")
        p.add_argument("--batch-size", type=int, help="with --accum, if memory forces it: 8 and 4 keep the batch")
        p.add_argument("--accum", type=int)
        p.add_argument("--workers", type=int)
        p.add_argument("--device", help="auto | cuda | mps | cpu")
        p.add_argument("--grad-checkpointing", action="store_true")
    p_ap = sub.add_parser("apply", help="Mac: each checkpoint on the CheXpert test images")
    p_ap.add_argument("--workers", type=int, default=4)
    p_ap.add_argument("--device", default="auto")
    p_rep = sub.add_parser("report", help="Mac: gates L4 and L5")
    p_rep.add_argument("--labels", help="the radiologist's labels (default: MIMIC_CXR_JPG_PATH/...test-set-labeled.csv)")
    p_rep.add_argument("--n-boot", type=int, default=1000)
    for p in (p_ap, p_rep):
        p.add_argument("--ext", help="M7b's CheXpert output dir (default: EXT_OUT or $OUTPUT_PATH/v2_chexpert)")
    for p in (p_tg, p_tr, p_gpu, p_ap, p_rep):
        p.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
        p.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    if args.command == "targets":
        facts = build_targets(out_dir, args.model)
        logger.info("Report: %s", write_targets_report(out_dir, facts))
        return facts
    if args.command in ("train", "gpu"):
        if not (out_dir / "manifest.parquet").exists():
            prepare_dir(out_dir, Path(args.bench) if args.bench else default_bench_dir())
        extra = _extra(args)
        if args.command == "train":
            return train(out_dir, args.arm, args.seed, extra, args.smoke)
        seeds = tuple(int(s) for s in args.seeds.split(","))
        if not set(seeds) <= set(SEEDS):
            raise SystemExit(f"§8h's seeds are {SEEDS}")
        return gpu(out_dir, seeds, tuple(ARMS), extra)
    ext = Path(args.ext) if args.ext else _env_dir("EXT_OUT", "v2_chexpert")
    if args.command == "apply":
        return apply(out_dir, ext, args.workers, args.device)
    labels = label.resolve_paths(None, args.labels, need_reports=False)[1]
    chex = chexpert.chexpert_dir(out_dir, ext)
    result = build_report(out_dir, chex, labels, args.n_boot)
    path = write_report(out_dir, result)
    for g in GATES:
        logger.info("%s: %s", g, result["gates"][g])
    for flag in result["flags"]:
        logger.warning("%s", flag)
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
