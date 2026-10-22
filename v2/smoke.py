"""
Wiring check: the whole v2 pipeline on synthetic data, in about a minute on CPU.

    python -m v2.smoke                 # temporary directory, deleted afterwards
    python -m v2.smoke --keep /tmp/v2_smoke

Builds a synthetic MIMIC-CXR-JPG tree with a planted signal (v2.synth), then
runs prepare_data -> embed -> probe -> zero_shot -> finetune (+ resume) ->
context + fuse -> M4b (tokens, attend, prior labels, stack) -> M5 (the
benchmark directory and its lineup) -> evaluate with test-size encoders,
through the same
command-line entry points you will use on real data. It passes only if every
stage runs AND the planted signal is recovered end to end: with the
"test-pixels" encoder, test_clean macro AUROC must be >= 0.9, and must fall
below 0.7 when the embeddings are deliberately misaligned with their studies
(proof that the check has teeth). For M4, no study's input may contain its own
findings, while the same text-only arm fed whole reports must score > 0.95
(proof that the leak trap has teeth). For M4b, the attention head must recover
the planted signal from cached tokens, and the prior-label arm fed each study's
own labels instead of its prior study's must score > 0.95, while the honest arm
stays near chance (synthetic studies are independent). For M5, the benchmark
directory must hold the official split and refuse to replace the v2 manifest,
and the planted signal must reach the lineup. No downloads, no MIMIC data
needed.
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
import pandas as pd

from . import attend, context, embed, evaluate, finetune, fuse, lineup, prepare_data, probe, stack, zero_shot
from .common import CHEXPERT_COLUMNS, load_manifest, mimic_csv, setup_logging
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
        # M5b amendment: the same fine-tune with M4b's attention head on the tokens.
        attn = finetune.main([*args[:3], "6", *(a for a in args[4:] if a != "--save-embeddings"), "--head",
                              "attention"])
        auc_attn = evaluate._get(evaluate.evaluate_run(out, attn, n_boot=50), "test_clean", "v2")["macro_auroc"]
        assert auc_attn >= 0.7, f"the attention-head fine-tune did not learn: test_clean macro AUROC {auc_attn:.3f}"
        return f"6 epochs with a resume; test_clean macro AUROC {auc:.3f} (attention head {auc_attn:.3f})"

    stage("M3 fine-tune + resume (learns)", m3)

    def m4() -> str:
        context.main(["--reports", str(paths.reports), "--mimic-iv-ed", str(paths.mimic_iv_ed),
                      "--mimic-iv", str(paths.mimic_iv), "--mimic-cxr-jpg", str(paths.mimic_root), *common])
        ctx = pd.read_parquet(out / "context.parquet")
        leaks = sum(f"leak{study}" in text for col in context.TEXT_SOURCES
                    for study, text in zip(ctx["study_id"], ctx[col]))
        assert leaks == 0, f"{leaks} inputs contain their own study's findings"
        emb = out / "embeddings" / "tiny-siglip_32.npz"
        fuse.main(["--embeddings", str(emb), "--n-boot", "100", "--jobs", "2", *common])
        text_only = evaluate._get(evaluate.evaluate_run(out, "tiny-siglip_32_ctx_text", n_boot=50),
                                  "test_clean", "v2")["macro_auroc"]
        # Negative control: the indication arm fed whole reports, findings included, must look far too good.
        reports = context.Reports(paths.reports)
        whole = [reports.get(subject, study) or "" for subject, study in zip(ctx["subject_id"], ctx["study_id"])]
        reports.close()
        leaky = out / "context_leaky.parquet"
        ctx.assign(indication=whole).to_parquet(leaky, index=False)
        fuse.main(["--embeddings", str(emb), "--context", str(leaky), "--arms", "indication", "--run", "leaky",
                   "--n-boot", "50", "--jobs", "2", *common])
        leaked = evaluate._get(evaluate.evaluate_run(out, "leaky_ctx_indication", n_boot=50),
                               "test_clean", "v2")["macro_auroc"]
        for path in [*(out / "predictions").glob("leaky_*"), *(out / "reports").glob("leaky_*"),
                     out / "reports" / "m4_leaky.md", leaky]:
            path.unlink(missing_ok=True)
        assert leaked > 0.95, f"whole reports scored only {leaked:.3f}: the leak trap has no teeth"
        return f"no input holds its own findings; text-only {text_only:.3f}, whole reports {leaked:.3f}"

    stage("M4 context + fuse (leak trap)", m4)

    def m4b() -> str:
        embed.main(["--model", "test-pixels", "--tokens", "12", "--workers", "0", *common])
        # A few hundred training studies need far more steps per study than the real recipe gives.
        res = attend.main(["--tokens", str(out / "embeddings" / "test-pixels_48_tokens12.npy"), "--seeds", "0",
                           "--max-epochs", "30", "--patience", "30", "--batch-size", "16", "--lr", "3e-3",
                           "--n-boot", "50", "--device", "cpu", *common])
        h0 = evaluate._get(evaluate.evaluate_run(out, "test-pixels_48_attn_h0", n_boot=50), "test_clean",
                           "v2")["macro_auroc"]
        assert h0 >= 0.85, f"the attention head missed the planted signal: test_clean macro AUROC {h0:.3f}"
        assert "G4b" in res["report"].read_text()
        emb = out / "embeddings" / "tiny-siglip_32.npz"
        fuse.main(["--embeddings", str(emb), "--arms", ",".join(fuse.M4B_ARMS), "--n-boot", "50", "--jobs", "2",
                   *common])
        m4b_report = (out / "reports" / "m4b_tiny-siglip_32.md").read_text()
        assert "(from M4)" in m4b_report and "**G4**" not in m4b_report, "M4b must reuse M4's arms, not refit them"
        honest = evaluate._get(evaluate.evaluate_run(out, "tiny-siglip_32_ctx_prior_labels", n_boot=50),
                               "test_clean", "v2")["macro_auroc"]
        # Negative control: each study's own CheXpert row in place of its prior study's must look far too good.
        ctx = pd.read_parquet(out / "context.parquet")
        own = pd.read_csv(mimic_csv(paths.mimic_root, "chexpert")).drop_duplicates("study_id").set_index("study_id")
        leaky = out / "context_leaky.parquet"
        ctx.assign(**{f"prior_label::{c}": own[c].reindex(ctx["study_id"]).to_numpy() for c in CHEXPERT_COLUMNS},
                   prior_labels_found=ctx["study_id"].isin(own.index).to_numpy()).to_parquet(leaky, index=False)
        fuse.main(["--embeddings", str(emb), "--context", str(leaky), "--arms", "prior_labels", "--run", "leakypl",
                   "--n-boot", "50", "--jobs", "2", *common])
        leaked = evaluate._get(evaluate.evaluate_run(out, "leakypl_ctx_prior_labels", n_boot=50), "test_clean",
                               "v2")["macro_auroc"]
        for path in [*(out / "predictions").glob("leakypl_*"), *(out / "reports").glob("leakypl_*"),
                     out / "reports" / "m4b_leakypl.md", leaky]:
            path.unlink(missing_ok=True)
        assert leaked > 0.95, f"own labels scored only {leaked:.3f}: the prior-label trap has no teeth"
        assert honest < 0.75, f"honest prior labels scored {honest:.3f} on independent synthetic studies: a leak?"
        stacked = stack.main(["--pair", "test-pixels_48_probe", "tiny-siglip_32_ctx_text", "--n-boot", "50", *common])
        return (f"h0 {h0:.3f}; G4b {res['gate']['verdict'].split(':')[0]}; prior labels {honest:.3f} vs own labels "
                f"{leaked:.3f}; {len(stacked['runs'])} stack")

    stage("M4b tokens + attend + prior labels + stack", m4b)

    def m5() -> str:
        bench = work / "v2_bench"
        bench_args = ["--mimic-cxr-jpg", str(paths.mimic_root), "--population", "all", "--split-scheme", "official",
                      "--image-format", "jpg", "--image-size", "64", "--workers", "2", "--min-count", "5",
                      "--rad-dino-list", str(paths.rad_dino_list)]
        try:
            prepare_data.main([*bench_args, "--skip-images", *common])
            raise AssertionError("the benchmark replaced the v2 manifest")
        except SystemExit:
            pass
        prepare_data.main([*bench_args, "--out", str(bench)])
        m = load_manifest(bench)
        assert set(m.loc[m["in_test_clean"], "official_split"]) == {"test"}, "test_clean is not the official test split"
        assert set(m.loc[m["split"] == "val", "official_split"]) == {"validate"}, "val is not the official validate split"
        out_b = ["--out", str(bench)]
        embed.main(["--model", "test-pixels", "--tokens", "12", "--with-pooled", "--workers", "0", *out_b])
        emb = bench / "embeddings" / "test-pixels_48.npz"
        probe.main(["--embeddings", str(emb), "--jobs", "2", *out_b])
        context.main(["--reports", str(paths.reports), "--mimic-iv-ed", str(paths.mimic_iv_ed),
                      "--mimic-iv", str(paths.mimic_iv), "--mimic-cxr-jpg", str(paths.mimic_root), *out_b])
        fuse.main(["--embeddings", str(emb), "--arms", "image,image_indication,indication", "--n-boot", "50",
                   "--jobs", "2", *out_b])
        attend.main(["--tokens", str(bench / "embeddings" / "test-pixels_48_tokens12.npy"), "--arms", "h0",
                     "--seeds", "0", "--max-epochs", "20", "--patience", "20", "--batch-size", "16", "--lr", "3e-3",
                     "--n-boot", "50", "--device", "cpu", *out_b])
        stack.main(["--pair", "test-pixels_48_attn_h0", "test-pixels_48_ctx_indication", "--n-boot", "50", *out_b])
        res = lineup.main(["--n-boot", "50", *out_b])
        assert any(c["run"] == "test-pixels_48_attn_h0" for c in res["compared"]), "the attention head is not lined up"
        assert any((c["run"], c["published"]) == ("test-pixels_48_attn_h0_stack_indication", "MMBT")
                   for c in res["compared"]), "the val combiner's image + indication run is not lined up with MMBT"
        row = next(c for c in res["compared"] if c["run"] == "test-pixels_48_probe")
        assert row["ours"] >= 0.9, f"the planted signal didn't reach the lineup: {row['ours']:.3f}"
        return (f"{len(m)} studies, {int(m['in_test_clean'].sum())} in the official test split; probe "
                f"{row['ours']:.3f} vs ResNet-50's {row['published_mean']:.3f} ({row['verdict'].strip('*')})")

    stage("M5 benchmark + lineup", m5)

    def reports() -> str:
        text = evaluate.summarize(out, n_boot=100)
        assert "G1" in text and "G2" in text and "G4" in text and "G4b" in text
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
