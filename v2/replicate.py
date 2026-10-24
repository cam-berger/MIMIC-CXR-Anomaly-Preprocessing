"""
M8: a second seed of the benchmark fine-tunes, and a longer fine-tune (docs/ANALYSIS.md, M8).

    python -m v2.replicate --bench output/v2_bench --ext output/v2_chexpert

Recomputes the four headline verdicts that rest on seed 0 with seed 1 of the same model: image only
against ResNet-50 and image + indication against MMBT (official test, positive vs rest), G7 on the
CheXpert test set, and M7a's radiologist minus labeler. A verdict stands if both seeds give it; if not,
the report names the weaker one both support. Then the 6-epoch attention head against the 3-epoch one
with the same seed, by M7's adoption rule (paired val gain, 95% interval above 0). Red flags first:
each new run's training recipe must equal its twin's. Writes reports/m8_replication.md in the benchmark
directory; only aggregates are logged.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from .common import LABELS, MIN_COUNT, load_dotenv, load_manifest, load_measurable, read_predictions, setup_logging
from .evaluate import N_BOOT, _get, compare_labels, compare_runs, evaluate_run, evaluate_set, fmt, fmt_ci, split_scheme
from .external import (CHEXZERO_MEAN, FIVE, PREFIX, RADIOLOGIST_FILE, RELOAD_MIN_CORR, _env_dir, _exists,
                       joint_classes, radiologist_manifests, read_radiologist)
from .lineup import PUBLISHED, published_mean, verdict

logger = logging.getLogger("v2.replicate")

SEEDS = (0, 1)
HEADS = ("ft_attn", "ft")  # the attention head (M7's primary model P) first, then mean pooling
LONG, SHORT = "ft_attn_e6_s0", "ft_attn_s0"  # 6 epochs against M5b's 3, same seed
STACK_TAGS = ("indication_bert", "indication", "text_prior_labels")  # M6's adopted text first, then M5b's two
# What a run must share with its twin (M8). The seed and run name differ by design; for the 6-epoch
# run also the epochs and total steps.
RECIPE = ("model", "revision", "size", "head", "grid", "warmup_epochs", "encoder_lr", "head_lr", "llrd",
          "weight_decay", "clip", "precision", "steps_per_epoch", "warmup_steps", "n_train", "n_val")
SAME_LENGTH = ("epochs", "total_steps")
MAY_BE_UNRECORDED = ("revision",)  # runs from before c1159c7's successor recorded no Hub commit
VERSIONS = ("torch", "transformers")
FIVE_SETS = {"test_clean:v2": list(FIVE), "test_clean:u0": list(FIVE)}
# Each kind of verdict from best to worst for our claim; when the seeds differ, both support the worse one.
PUBLISHED_ORDER = ("below ours", "level", "**above ours**")
LABEL_ORDER = ("above 0", "includes 0", "below 0")


def recipe_check(out_dir: Path, run: str, twin: str, same_length: bool = True) -> dict:
    """Where `run`'s saved training config differs from `twin`'s: the recipe (a red flag), then library versions."""
    a, b = (read_predictions(Path(out_dir) / "predictions" / f"{r}.parquet")[1].get("config") or {}
            for r in (run, twin))
    keys = RECIPE + (SAME_LENGTH if same_length else ())
    unrecorded = [k for k in MAY_BE_UNRECORDED if a.get(k) is None or b.get(k) is None]
    differs = {k: [a.get(k), b.get(k)] for k in keys if k not in unrecorded and a.get(k) != b.get(k)}
    batch = [(c.get("batch_size") or 0) * (c.get("accum") or 0) for c in (a, b)]  # what each optimizer step sees
    if batch[0] != batch[1]:
        differs["effective batch"] = batch
    return {"run": run, "twin": twin, "differs": differs, "no_config": not a or not b, "unrecorded": unrecorded,
            "versions": {k: [a.get(k), b.get(k)] for k in VERSIONS if a.get(k) != b.get(k)}}


def combine(v0: Optional[str], v1: Optional[str], order: tuple) -> str:
    """M8: a verdict stands if both seeds give it; otherwise the write-up states the weaker one."""
    if v0 is None or v1 is None:
        return "not decided: no result for seed " + ("1" if v1 is None else "0")
    if v0 == v1:
        return f"stands: {v0}"
    return f"seed-dependent ({v0} / {v1}); both support: {max(v0, v1, key=order.index)}"


def against(rep: Optional[dict], name: str) -> Optional[dict]:
    """A run on the official test (positive vs rest) against a published model's mean over the same classes."""
    res = _get(rep, "test_clean", "u0") if rep else None
    if not res:
        return None
    pub = published_mean(next(r for r in PUBLISHED if r.name == name), res["classes"])
    return {"mean": res["macro_auroc"], "ci": res["macro_auroc_ci"], "published": pub,
            "side": verdict(pub, res["macro_auroc_ci"])}


def five_finding(ext: Optional[Path], manifest, run: str, n_boot: int) -> Optional[dict]:
    """A run's five-finding mean on the CheXpert test set, against CheXzero's (G7)."""
    if ext is None or not _exists(ext, run):
        return None
    pred, _ = read_predictions(ext / "predictions" / f"{run}.parquet")
    res = evaluate_set(pred, manifest, FIVE_SETS, "test_clean", "v2", n_boot)
    return {"mean": res["macro_auroc"], "ci": res["macro_auroc_ci"], "side": verdict(CHEXZERO_MEAN, res["macro_auroc_ci"])}


def label_effect(bench: Path, run: str, rad: Optional[tuple], n_boot: int) -> Optional[dict]:
    """M7a's radiologist minus labeler (v2 policy, paired, same studies) for one run."""
    if rad is None or not _exists(bench, run):
        return None
    labeler, reader, joint = rad
    pred, _ = read_predictions(bench / "predictions" / f"{run}.parquet")
    c = compare_labels(pred, reader, labeler, joint, "test_clean", "v2", n_boot)
    lo, hi = c["diff_ci"]
    return {"mean": c["diff"], "ci": c["diff_ci"], "side": "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"}


def paired(out_dir: Optional[Path], a: str, b: str, test_set: str, metric: str, n_boot: int, manifest=None,
           frozen=None) -> Optional[dict]:
    """Macro AUROC(a) - AUROC(b), paired on the same studies; None unless both runs exist."""
    if out_dir is None or not (_exists(out_dir, a) and _exists(out_dir, b)):
        return None
    return compare_runs(out_dir, a, b, test_set, metric, n_boot, write=False, manifest=manifest, frozen=frozen)


def signed(d: Optional[dict]) -> str:
    return f"{d['diff']:+.3f} [{d['diff_ci'][0]:+.3f}, {d['diff_ci'][1]:+.3f}]" if d else "-"


def replicate(bench: Path, ext: Optional[Path] = None, labels: Optional[Path] = None, prefix: str = PREFIX,
              n_boot: int = N_BOOT, min_count: int = MIN_COUNT) -> dict:
    bench = Path(bench)
    if split_scheme(bench) != "official":
        raise SystemExit(f"{bench} isn't M5's benchmark directory (official split): pass --bench")
    name = {short: f"{prefix}_{short}" for short in
            [f"{h}_s{s}" for h in HEADS for s in SEEDS] + [LONG]}  # LONG's twin, SHORT, is ft_attn_s0
    new = [name["ft_attn_s1"], name["ft_s1"], name[LONG]]
    if not any(_exists(bench, run) for run in new):
        raise SystemExit(f"none of {', '.join(new)} is in {bench}: run m8_gpu, bring the runs home, then m8")
    ext = Path(ext) if ext and (Path(ext) / "manifest.parquet").exists() else None
    ext_manifest = load_manifest(ext) if ext else None
    rad, rad_note = None, f"the radiologist's labels, {RADIOLOGIST_FILE}, weren't found"
    if labels and Path(labels).exists():
        labeler, reader, _ = radiologist_manifests(bench, read_radiologist(Path(labels))[0])
        joint = joint_classes(labeler, reader, min_count)  # M7a's classes: measurable under both sets of labels
        rad, rad_note = ((labeler, reader, joint), None) if joint["test_clean:v2"] else \
            (None, f"no class has {min_count} positives and negatives under both sets of labels")

    manifest, frozen = load_manifest(bench), load_measurable(bench)
    images = list(name.values())
    stacks = [f"{img}_stack_{tag}" for img in images for tag in STACK_TAGS]
    reps = {run: evaluate_run(bench, run, n_boot, manifest, frozen) for run in images + stacks if _exists(bench, run)}
    metas = {run: rep["meta"] for run, rep in reps.items()}

    # Red flags: a recipe that isn't its twin's, a run cut short by its time limit, a reload that doesn't match.
    checks = [recipe_check(bench, name[f"{h}_s1"], name[f"{h}_s0"]) for h in HEADS
              if _exists(bench, name[f"{h}_s1"]) and _exists(bench, name[f"{h}_s0"])]
    if _exists(bench, name[LONG]) and _exists(bench, name[SHORT]):
        checks.append(recipe_check(bench, name[LONG], name[SHORT], same_length=False))
    reloads = {run: read_predictions(ext / "predictions" / f"{run}.parquet")[1].get("reload_check")
               for run in new if ext is not None and _exists(ext, run)}
    flags = [f"`{c['run']}`'s recipe differs from `{c['twin']}`'s: "
             + "; ".join(f"{k} {v[0]} against {v[1]}" for k, v in c["differs"].items()) for c in checks if c["differs"]]
    flags += [f"`{c['run']}` or `{c['twin']}` has no saved training config to check" for c in checks if c["no_config"]]
    flags += [f"`{run}` stopped on its time limit before its last epoch" for run in new
              if metas.get(run, {}).get("stopped_early")]
    flags += [f"`{run}` reloaded for CheXpert correlates {rc['correlation']:.4f} with its benchmark predictions "
              f"(at least {RELOAD_MIN_CORR} needed)" for run, rc in reloads.items()
              if rc and rc["correlation"] < RELOAD_MIN_CORR]

    # The four headline verdicts, seed by seed (M8).
    claims = []
    for title, rule, order, results in (
        ("Image only against ResNet-50", "ResNet-50 below our 95% CI (official test, positive vs rest)",
         PUBLISHED_ORDER, [against(reps.get(name[f"ft_attn_s{s}"]), "ResNet-50") for s in SEEDS]),
        ("Image + indication against MMBT", "MMBT below our 95% CI (the frozen-BERT stack, the same test)",
         PUBLISHED_ORDER, [against(reps.get(f"{name[f'ft_attn_s{s}']}_stack_indication_bert"), "MMBT")
                           for s in SEEDS]),
        (f"G7: CheXpert, five findings, against CheXzero ({CHEXZERO_MEAN})",
         "PASS unless CheXzero is above our 95% CI", PUBLISHED_ORDER,
         [five_finding(ext, ext_manifest, name[f"ft_attn_s{s}"], n_boot) for s in SEEDS]),
        ("M7a: radiologist minus labeler (v2 policy)", "95% CI above 0", LABEL_ORDER,
         [label_effect(bench, name[f"ft_attn_s{s}"], rad, n_boot) for s in SEEDS]),
    ):
        sides = [r["side"] if r else None for r in results]
        claims.append({"claim": title, "rule": rule, "results": results, "verdict": combine(*sides, order)})
    g7 = claims[2]["verdict"]
    if not g7.startswith("not decided"):
        claims[2]["verdict"] += " -> G7 " + ("FAIL" if g7.endswith("**above ours**") else "PASS")

    # Seed noise: seed 1 minus seed 0, paired, for each fine-tune and its stacks.
    pairs = [(name[f"{h}_s1"], name[f"{h}_s0"]) for h in HEADS]
    pairs += [(f"{a}_stack_{tag}", f"{b}_stack_{tag}") for tag in STACK_TAGS for a, b in pairs[:len(HEADS)]]
    noise = []
    for a, b in pairs:
        test = paired(bench, a, b, "test_clean", "u0", n_boot)
        if test:
            noise.append({"a": a, "b": b, "test": test, "val": paired(bench, a, b, "val", "v2", n_boot),
                          "chexpert": paired(ext, a, b, "test_clean", "v2", n_boot, ext_manifest, FIVE_SETS)})

    # Training length: 6 epochs against 3 with the same seed, decided on val by M7's adoption rule.
    length = None
    if _exists(bench, name[LONG]) and _exists(bench, name[SHORT]):
        length = {"val": paired(bench, name[LONG], name[SHORT], "val", "v2", n_boot),
                  "test": paired(bench, name[LONG], name[SHORT], "test_clean", "u0", n_boot),
                  "chexpert": paired(ext, name[LONG], name[SHORT], "test_clean", "v2", n_boot, ext_manifest,
                                     FIVE_SETS)}
        length["adopted"] = bool(length["val"]["diff_ci"][0] > 0)
        length["decision"] = ("**adopted**: future fine-tunes train 6 epochs" if length["adopted"] else
                              "**not adopted**: 3 epochs stay (the paired val gain's 95% CI isn't above 0)")

    out = {"flags": flags, "checks": checks, "reloads": reloads, "claims": claims, "noise": noise,
           "length": length}
    out["path"] = write_report(bench, prefix, ext is not None, rad_note, out, reps, metas)
    return out


def write_report(bench: Path, prefix: str, have_ext: bool, rad_note: Optional[str], out: dict, reps: dict,
                 metas: dict) -> Path:
    long, short = f"{prefix}_{LONG}", f"{prefix}_{SHORT}"
    lines = ["# M8: a second seed, and a longer fine-tune", "",
             "Each headline verdict that rests on seed 0 of the attention-head fine-tune, recomputed with seed 1 "
             "(docs/ANALYSIS.md, M8). The official test is scored positive vs rest, as the published results "
             "are; CheXpert over its five findings; M7a's comparison under the v2 policy. Nothing logged is "
             f"replaced: P stays `{short}` for M7's records, and nothing here is chosen on a test set.", "",
             "## Red flags (read these first)", ""]
    lines += [f"- **RED FLAG:** {f}" for f in out["flags"]] or ["- none"]
    if not have_ext:
        lines.append("- Not recomputed: G7 (no prepared CheXpert test set; pass `--ext`)")
    if rad_note:
        lines.append(f"- Not recomputed: M7a ({rad_note})")
    lines += [f"- Unchecked: {', '.join(c['unrecorded'])} isn't recorded for `{c['run']}` or `{c['twin']}`; compare "
              "the model's Hub commit by hand" for c in out["checks"] if c["unrecorded"]]

    lines += ["", "## Checks", "", "| Run | Twin | Recipe | Library versions | Best epoch | Stopped early | "
              "CheXpert reload |", "|---|---|---|---|---|---|---|"]
    for c in out["checks"]:
        meta, rc = metas.get(c["run"], {}), out["reloads"].get(c["run"])
        best, epochs = meta.get("best_epoch"), (meta.get("config") or {}).get("epochs")
        lines.append(f"| `{c['run']}` | `{c['twin']}` | {'DIFFERS' if c['differs'] else 'same'} | "
                     + ("; ".join(f"{k} {v[0]} against {v[1]}" for k, v in c["versions"].items()) or "same")
                     + f" | {best + 1 if best is not None else '-'} of {epochs or '-'} | "
                     f"{'yes' if meta.get('stopped_early') else 'no'} | "
                     + (f"correlation {rc['correlation']:.4f}" if rc else "not applied") + " |")
    if not out["checks"]:
        lines.append("| - | - | - | - | - | - | - |")

    lines += ["", "## The headline verdicts, seed by seed", "",
              "| Claim | Rule | Seed 0 | Seed 1 | Verdict |", "|---|---|---|---|---|"]
    for c in out["claims"]:
        cells = []
        for r in c["results"]:
            if not r:
                cells.append("not run")
            elif "published" in r:
                cells.append(f"{fmt_ci(r['mean'], r['ci'])}; published {fmt(r['published'])} is {r['side']}")
            elif c["claim"].startswith("G7"):
                cells.append(f"{fmt_ci(r['mean'], r['ci'])}; CheXzero is {r['side']}")
            else:
                cells.append(f"{r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]: {r['side']}")
        lines.append(f"| {c['claim']} | {c['rule']} | {cells[0]} | {cells[1]} | {c['verdict']} |")

    lines += ["", "## Seed noise: seed 1 minus seed 0 (paired)", "",
              "| Seed 1 | Official test, seed 0 / seed 1 | Seed 1 minus seed 0 | Two-seed mean | val (v2 policy), "
              "seed 1 minus seed 0 | CheXpert five findings, seed 1 minus seed 0 |", "|---|---|---|---|---|---|"]
    for n in out["noise"]:
        t = n["test"]
        lines.append(f"| `{n['a']}` | {fmt(t['macro_b'])} / {fmt(t['macro_a'])} | {signed(t)} | "
                     f"{fmt((t['macro_a'] + t['macro_b']) / 2)} | {signed(n['val'])} | {signed(n['chexpert'])} |")
    if not out["noise"]:
        lines.append("| - | - | - | - | - | - |")

    length = out["length"]
    lines += ["", f"## Training length: `{long}` minus `{short}` (6 epochs against 3, paired)", ""]
    if length:
        lines += ["| Set | 3 epochs | 6 epochs | 6 minus 3 |", "|---|---|---|---|"]
        for key, label in (("val", "val (v2 policy): decides"), ("test", "Official test, positive vs rest"),
                           ("chexpert", "CheXpert, five findings")):
            d = length[key]
            if d:
                lines.append(f"| {label} | {fmt(d['macro_b'])} | {fmt(d['macro_a'])} | {signed(d)} |")
        lines += ["", f"Decision (M7's adoption rule): {length['decision']}. Read the differences against the seed "
                  "noise above: one no larger than seed 1 minus seed 0 isn't a schedule effect."]
    else:
        lines.append("Not run: it needs both runs.")

    shown = [run for run in (short, f"{prefix}_ft_attn_s1", long) if run in reps]
    if shown:
        per = {run: {c["label"]: c for c in (_get(reps[run], "test_clean", "u0") or {}).get("per_class", [])}
               for run in shown}
        lines += ["", "## Per class, official test, positive vs rest (AUROC)", "",
                  "| Class | " + " | ".join(f"`{run}`" for run in shown) + " |", "|---|" + "---|" * len(shown)]
        lines += [f"| {label} | " + " | ".join(fmt(per[run][label]["auroc"]) for run in shown) + " |"
                  for label in LABELS if all(label in per[run] for run in shown)]

    path = Path(bench) / "reports" / "m8_replication.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", help="M5's benchmark dir (default: BENCH_OUT, else $OUTPUT_PATH/v2_bench)")
    parser.add_argument("--ext", help="the CheXpert test set's dir (default: EXT_OUT, else $OUTPUT_PATH/v2_chexpert)")
    parser.add_argument("--labels", help=f"the radiologist's labels (default: MIMIC_CXR_JPG_PATH/{RADIOLOGIST_FILE})")
    parser.add_argument("--prefix", default=PREFIX, help="the benchmark's image run prefix")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT, help="M7a's measurable-class threshold")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    load_dotenv()
    bench = Path(args.bench) if args.bench else _env_dir("BENCH_OUT", "v2_bench")
    ext = Path(args.ext) if args.ext else _env_dir("EXT_OUT", "v2_chexpert")
    labels = args.labels or (Path(os.environ["MIMIC_CXR_JPG_PATH"]) / RADIOLOGIST_FILE
                             if os.environ.get("MIMIC_CXR_JPG_PATH") else None)
    out = replicate(bench, ext, labels, args.prefix, args.n_boot, args.min_count)
    for flag in out["flags"]:
        logger.warning("RED FLAG: %s", flag)
    for c in out["claims"]:
        logger.info("%s: %s", c["claim"], c["verdict"])
    if out["length"]:
        logger.info("Training length: %s", out["length"]["decision"].replace("**", ""))
    logger.info("Report: %s", out["path"])
    return out


if __name__ == "__main__":
    main()
