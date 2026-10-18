"""
Wiring check: the whole v2 pipeline on synthetic data, in about a minute on CPU.

    python -m v2.smoke                 # temporary directory, deleted afterwards
    python -m v2.smoke --keep /tmp/v2_smoke

Builds a synthetic MIMIC-CXR-JPG tree with a planted signal (v2.synth), then
runs prepare_data -> embed -> probe -> zero_shot -> finetune (+ resume) -> fuse
-> evaluate with test-size encoders, through the same command-line entry points
you will use on real data. It passes only if every stage runs AND the planted
signal is recovered end to end: with the "test-pixels" encoder, test_clean
macro AUROC must be >= 0.9, and must fall below 0.7 when the embeddings are
deliberately misaligned with their studies (proof that the check has teeth).
No downloads, no MIMIC data needed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import embed, evaluate, finetune, fuse, prepare_data, probe, zero_shot
from .common import load_manifest, setup_logging
from .synth import make_dataset


def run_checks(work: Path, patients: int = 400) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    out = work / "v2"

    def stage(name: str, fn: Callable[[], str]) -> bool:
        t0 = time.time()
        try:
            detail = fn() or ""
            results.append((name, True, f"{detail} ({time.time() - t0:.1f}s)"))
            return True
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going where possible
            traceback.print_exc()
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            return False

    paths = make_dataset(work / "data", n_patients=patients, seed=1)
    common = ["--out", str(out)]

    def m0() -> str:
        prepare_data.main(["--mimic-cxr-jpg", str(paths.mimic_root), "--cohort-dir", str(paths.cohort_dir),
                           "--image-size", "64", "--workers", "2", "--min-count", "5",
                           "--rad-dino-list", str(paths.rad_dino_list), "--extend-test-clean", *common])
        m = load_manifest(out)
        audit = (out / "audit.json").read_text()
        assert set(m["view"]) <= {"PA", "AP"}, "a lateral view was selected"
        assert evaluate.json.loads(audit)["patient_overlap"] == {"train/val": 0, "train/test": 0, "val/test": 0}
        assert m["in_test_clean"].any() and m["in_test_v1"].any(), "a test set is empty"
        rad = evaluate.json.loads(audit)["rad_dino"]["test_clean"]
        assert rad["images_in_list"] == 0 and rad["patients_in_list"] == 0, "test_clean overlaps RAD-DINO's list"
        return f"{len(m)} studies, {(m['source'] != 'cohort').sum()} from --extend-test-clean"

    if not stage("M0 prepare_data + audit", m0):
        return results

    def m1() -> str:
        embed.main(["--model", "test-pixels", "--workers", "0", *common])
        probe.main(["--embeddings", str(out / "embeddings" / "test-pixels_48.npz"), "--jobs", "2", *common])
        rep = evaluate.evaluate_run(out, "test-pixels_48_probe", n_boot=200)
        auc = evaluate._get(rep, "test_clean", "v2")["macro_auroc"]
        assert auc >= 0.9, f"planted signal not recovered: test_clean macro AUROC {auc:.3f}"
        return f"test_clean macro AUROC {auc:.3f} (planted signal recovered)"

    stage("M1 embed + probe (signal recovered)", m1)

    def negative_control() -> str:
        ids, emb, meta = embed.load_npz(out / "embeddings" / "test-pixels_48.npz")
        shuffled = out / "embeddings" / "shuffled_48.npz"
        embed.save_npz(shuffled, ids, emb[np.random.default_rng(0).permutation(len(ids))], meta)
        probe.main(["--embeddings", str(shuffled), "--run", "shuffled_probe", "--jobs", "2", *common])
        auc = evaluate._get(evaluate.evaluate_run(out, "shuffled_probe", n_boot=50), "test_clean", "v2")["macro_auroc"]
        assert auc < 0.7, f"misaligned embeddings still scored {auc:.3f}: the check has no teeth"
        for path in [*(out / "predictions").glob("shuffled_probe.*"), *(out / "reports").glob("shuffled_probe.*"),
                     shuffled]:
            path.unlink()
        return f"misaligned embeddings score {auc:.3f}"

    stage("M1 negative control (misaligned)", negative_control)

    def m2() -> str:
        embed.main(["--model", "tiny-siglip", "--workers", "0", *common])
        zero_shot.main(["--model", "tiny-siglip", *common])
        rep = evaluate.evaluate_run(out, "tiny-siglip_32_zeroshot", n_boot=50)
        return f"{len(rep['results'])} evaluation sets"

    stage("M2 zero-shot", m2)

    def m3() -> str:
        # The tiny random ViT needs a much higher learning rate than the real recipe to learn in seconds.
        args = ["--model", "tiny-dinov2", "--epochs", "4", "--batch-size", "16", "--accum", "1",
                "--encoder-lr", "1e-3", "--head-lr", "1e-2", "--llrd", "1.0",
                "--workers", "0", "--save-embeddings", *common]
        run = finetune.main(args)
        last = out / "runs" / run / "last.pt"
        assert last.exists() and (out / "runs" / run / "best.pt").exists(), "checkpoints missing"
        finetune.main([*args[:3], "6", *args[4:], "--resume"])  # two more epochs from last.pt
        history = evaluate.json.loads((out / "runs" / run / "history.json").read_text())
        assert [h["epoch"] for h in history] == list(range(6)), f"resume did not continue: {history}"
        assert (out / "embeddings" / f"{run}.npz").exists(), "fine-tuned embeddings missing"
        auc = evaluate._get(evaluate.evaluate_run(out, run, n_boot=50), "test_clean", "v2")["macro_auroc"]
        assert auc >= 0.7, f"fine-tuning did not learn the planted signal: test_clean macro AUROC {auc:.3f}"
        return f"6 epochs with a resume; test_clean macro AUROC {auc:.3f}"

    stage("M3 fine-tune + resume (learns)", m3)

    def m4() -> str:
        fuse.main(["--embeddings", str(out / "embeddings" / "tiny-siglip_32.npz"),
                   "--cohort-dir", str(paths.cohort_dir), "--n-boot", "100", "--jobs", "2", *common])
        return "image-only and image+vitals predictions written"

    stage("M4 fuse", m4)

    def reports() -> str:
        text = evaluate.summarize(out, n_boot=100)
        assert "G1" in text and "G2" in text
        n = len(list((out / "reports").glob("*.md")))
        return f"{n} reports, summary with gates"

    stage("evaluate --all", reports)
    return results


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", help="Work in this directory and keep it (default: a temporary directory)")
    parser.add_argument("--patients", type=int, default=400)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="v2_smoke_"))
    if args.keep and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    try:
        results = run_checks(work, args.patients)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    print("\nv2 wiring check")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{'ALL PASSED' if not failed else f'{failed} FAILED'}" + (f"; outputs in {work}" if args.keep else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
