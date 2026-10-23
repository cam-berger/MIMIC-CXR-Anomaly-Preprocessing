"""
Setup check: run this first on any machine (laptop with the external drive, or a rented GPU).

    python -m v2.doctor --mimic-cxr-jpg /Volumes/drive/mimic-cxr-jpg/2.1.0 --cohort-dir output/cohorts
    python -m v2.doctor --load-models        # also download and run each encoder on one image

Checks Python packages, the compute device, the MIMIC-CXR-JPG files (CSV
columns, a sample of JPGs, read speed), v1's cohort files, the output
directory, Hugging Face access (MedSigLIP is gated) and, with --load-models,
a one-image load test per encoder. Exits non-zero if anything required fails.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from .common import default_out_dir, load_dotenv, mimic_csv

REQUIRED = ("numpy", "pandas", "pyarrow", "PIL", "scipy", "sklearn", "joblib", "torch", "torchvision",
            "transformers", "huggingface_hub")
OPTIONAL = ("h5py", "sentencepiece", "pytest")
CSV_COLUMNS = {
    "metadata": {"dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"},
    "split": {"dicom_id", "subject_id", "split"},
    "chexpert": {"subject_id", "study_id", "Atelectasis", "Pneumothorax"},
}


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        print(f"  [ OK ] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        print(f"  [FAIL] {msg}")

    def section(self, title: str) -> None:
        print(f"\n{title}")


def check_packages(r: Report) -> None:
    r.section("Python and packages")
    if sys.version_info >= (3, 10):
        r.ok(f"Python {platform.python_version()}")
    else:
        r.fail(f"Python {platform.python_version()}: need 3.10+")
    for name in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(name)
            r.ok(f"{name} {getattr(mod, '__version__', '')}")
        except ImportError:
            hint = {"h5py": " (optional: --v1-hdf5 audit)", "sentencepiece": " (needed for MedSigLIP's tokenizer)",
                    "pytest": " (optional: tests)"}.get(name, "")
            (r.fail if name in REQUIRED else r.warn)(f"{name} not installed{hint}")


def check_device(r: Report) -> None:
    r.section("Compute device")
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        r.ok(f"CUDA: {props.name}, {props.total_memory / 1e9:.0f} GB, bf16 {'yes' if torch.cuda.is_bf16_supported() else 'no'}")
        if props.total_memory < 20e9:
            r.warn("Under 20 GB of GPU memory: use --batch-size 8 --accum 4 or --grad-checkpointing for 518 px")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        r.ok("Apple MPS GPU (fp32). Fine-tune at --size 224 locally; use a CUDA GPU for 518 px")
    else:
        r.warn(f"CPU only ({os.cpu_count()} cores): fine for M0, M1 pilots at 224 px and M4; not for M3")


def check_mimic(r: Report, root: Optional[Path], n_sample: int, cohort_dir: Optional[Path] = None) -> None:
    r.section("MIMIC-CXR-JPG")
    if root is None:
        r.warn("No --mimic-cxr-jpg and no MIMIC_CXR_JPG_PATH: skipped")
        return
    if not root.is_dir():
        r.fail(f"{root} does not exist (is the drive mounted?)")
        return
    r.ok(f"root {root}")
    import pandas as pd

    meta = None
    for name, cols in CSV_COLUMNS.items():
        path = mimic_csv(root, name)
        if not path.exists():
            r.fail(f"{path.name} missing")
            continue
        try:
            head = pd.read_csv(path, nrows=5)
            missing = cols - set(head.columns)
            if missing:
                r.fail(f"{path.name}: missing columns {sorted(missing)}")
            else:
                r.ok(f"{path.name}")
            if name == "metadata":
                meta = pd.read_csv(path, usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])
        except Exception as exc:  # noqa: BLE001
            r.fail(f"{path.name}: {exc}")
    if not (root / "files").is_dir():
        r.fail(f"{root / 'files'} missing")
        return
    if meta is None:
        return
    from PIL import Image

    from .common import jpg_path

    frontal = meta[meta["ViewPosition"].isin(["PA", "AP"])]
    r.ok(f"metadata: {len(meta):,} images, {len(frontal):,} frontal")
    what = "frontal JPGs"
    if cohort_dir is not None and (cohort_dir / "anomalous_train.parquet").exists():
        # Sample what M0 will read: a partial download is fine if it covers the cohort.
        studies = pd.concat([pd.read_parquet(cohort_dir / f"anomalous_{s}.parquet", columns=["study_id"])
                             for s in ("train", "val") if (cohort_dir / f"anomalous_{s}.parquet").exists()])
        frontal = frontal[frontal["study_id"].isin(studies["study_id"])]
        what = "frontal JPGs of cohort studies"
    if frontal.empty:
        r.fail("no frontal images to sample (do the cohort's study_ids match this MIMIC-CXR-JPG?)")
        return
    sample = frontal.sample(n=min(n_sample, len(frontal)), random_state=0)
    t0, nbytes, bad = time.time(), 0, []
    for row in sample.itertuples(index=False):
        path = jpg_path(root, row.subject_id, row.study_id, row.dicom_id)
        try:
            nbytes += path.stat().st_size
            with Image.open(path) as img:
                img.load()
        except Exception as exc:  # noqa: BLE001
            bad.append(type(exc).__name__)  # error types only: no file names or ids on the console
    elapsed = max(time.time() - t0, 1e-6)
    if bad:
        share = len(bad) / len(sample)
        kinds = {k: bad.count(k) for k in sorted(set(bad))}
        msg = (f"{len(bad)}/{len(sample)} sampled {what} unreadable {kinds}. "
               "Is the download complete? M0 drops missing images and counts them in the audit")
        (r.fail if share >= 0.05 else r.warn)(msg)
    else:
        r.ok(f"{len(sample)} sampled {what} decode; {nbytes / elapsed / 1e6:.0f} MB/s, {len(sample) / elapsed:.1f} img/s")
    if len(sample) and len(sample) / elapsed < 5:
        r.warn("Slow image reads: prepare_data reads ~30k JPGs once; expect it to take a while")


def check_cohort(r: Report, cohort_dir: Optional[Path]) -> None:
    r.section("v1 cohort files")
    if cohort_dir is None:
        r.warn("No --cohort-dir: prepare_data will use all abnormal studies and test_v1 will be empty")
        return
    import pyarrow.parquet as pq

    for name in ("train", "val"):
        path = cohort_dir / f"anomalous_{name}.parquet"
        if not path.exists():
            r.fail(f"{path} missing")
            continue
        names = set(pq.read_schema(path).names)
        if {"subject_id", "study_id"} - names:
            r.fail(f"{path.name}: needs subject_id and study_id")
            continue
        rows = pq.ParquetFile(path).metadata.num_rows
        r.ok(f"{path.name}: {rows:,} rows")


def check_context_sources(r: Report, out_dir: Path) -> None:
    """M4 inputs (docs/V2_PLAN.md §6 M4). Only M4 needs them, so a missing source is a warning."""
    r.section("M4 sources (text written before the X-ray)")
    from .context import Reports, find_csv

    value = os.environ.get("MIMIC_CXR_REPORTS_PATH")
    if not value:
        r.warn("No MIMIC_CXR_REPORTS_PATH: M4 has no indications or prior reports "
               "(mimic-cxr-reports.zip, from the MIMIC-CXR project on PhysioNet)")
    elif not Path(value).expanduser().exists():
        r.fail(f"MIMIC_CXR_REPORTS_PATH: {value} does not exist")
    else:
        reports = Reports(Path(value).expanduser())
        manifest = out_dir / "manifest.parquet"
        if len(reports) == 0:
            r.fail(f"{value}: no report files (expected files/pXX/pXXXXXXXX/sXXXXXXXX.txt)")
        elif manifest.exists():
            import pandas as pd

            m = pd.read_parquet(manifest, columns=["subject_id", "study_id"])
            sample = m.sample(n=min(200, len(m)), random_state=0)
            found = sum(reports.get(su, st) is not None for su, st in zip(sample["subject_id"], sample["study_id"]))
            (r.ok if found >= 0.95 * len(sample) else r.warn)(
                f"reports: {found} of {len(sample)} sampled manifest studies have one")
        else:
            r.ok(f"reports found in {value}")
        reports.close()

    value = os.environ.get("MIMIC_IV_ED_PATH")
    if not value:
        r.warn("No MIMIC_IV_ED_PATH: M4 has no chief complaints or triage vitals")
    else:
        root = Path(value).expanduser()
        missing = [n for n in ("edstays", "triage") if find_csv(root, n) is None]
        (r.fail if missing else r.ok)(f"MIMIC-IV-ED: {'missing ' + ', '.join(missing) if missing else 'edstays and triage found'}")

    value = os.environ.get("MIMIC_IV_PATH")
    has_patients = bool(value) and find_csv(Path(value).expanduser(), "patients") is not None
    (r.ok if has_patients else r.warn)("MIMIC-IV hosp/patients " + ("found" if has_patients else
                                                                    "not found: M4 runs without age"))


def check_m4b(r: Report, out_dir: Path, mimic_root: Optional[Path]) -> None:
    """M4b inputs (docs/V2_PLAN.md §6 M4b): prior-study labels in the context table, and room for the token cache."""
    r.section("M4b (text-steered attention, prior-study labels, val combiner)")
    from .common import mimic_csv

    if mimic_root is not None and mimic_root.exists():
        found = mimic_csv(mimic_root, "chexpert").exists()
        (r.ok if found else r.warn)("mimic-cxr-2.0.0-chexpert.csv.gz " + ("found: prior-study labels available" if found
                                                                          else "not found: no prior-study labels"))
    ctx = out_dir / "context.parquet"
    if ctx.exists():
        import pyarrow.parquet as pq

        has = "prior_labels_found" in pq.read_schema(ctx).names  # the schema only: no rows are read
        (r.ok if has else r.warn)("context.parquet " + ("has prior-study labels" if has else
                                                         "predates M4b: the m4b stage rebuilds it"))
    cached = sorted((out_dir / "embeddings").glob("*_tokens*.json")) if (out_dir / "embeddings").exists() else []
    if cached:
        r.ok("token cache: " + ", ".join(p.with_suffix(".npy").name for p in cached))
    elif out_dir.exists():
        free = shutil.disk_usage(out_dir).free / 1e9
        (r.ok if free >= 10 else r.warn)(f"no token cache yet; {free:.0f} GB free (RAD-DINO 518 px tokens need ~7 GB)")


def check_m7(r: Report, mimic_root: Optional[Path]) -> None:
    """M7 inputs (docs/V2_PLAN.md §6 M7), as counts: the radiologist's labels, the CheXpert test set, the fine-tunes."""
    r.section("M7 (a radiologist's labels; the CheXpert test set)")
    from .external import CANDIDATES, IMAGE_RE, PREFIX, RADIOLOGIST_FILE, _env_dir

    if mimic_root is not None and mimic_root.exists():
        rad = mimic_root / RADIOLOGIST_FILE
        if rad.exists():
            r.ok(f"{RADIOLOGIST_FILE}: {sum(1 for _ in rad.open()) - 1} studies")
        else:
            r.warn(f"{RADIOLOGIST_FILE} not in MIMIC_CXR_JPG_PATH: M7a can't run (it ships with MIMIC-CXR-JPG 2.1.0)")
    value = os.environ.get("CHEXPERT_TEST_PATH")
    if not value:
        r.warn("No CHEXPERT_TEST_PATH: M7b has no CheXpert test set")
    elif not Path(value).expanduser().is_dir():
        r.fail(f"CHEXPERT_TEST_PATH: {value} is not a directory")
    else:
        root = Path(value).expanduser()
        labels = sorted(root.rglob("groundtruth.csv"))
        (r.ok if len(labels) == 1 else r.warn)(f"groundtruth.csv: {len(labels)} found (need 1)")
        frontal = sum(1 for p in root.rglob("*") if (m := IMAGE_RE.search(p.as_posix())) and
                      m.group(4).lower() == "frontal" and "/test/" in f"/{p.relative_to(root).as_posix()}")
        (r.ok if frontal else r.warn)(f"{frontal} frontal test images" + ("" if frontal else
                                      ": download them from the original CheXlocalize release (v2/README.md §10)"))
    bench = _env_dir("BENCH_OUT", "v2_bench")
    for run in (f"{PREFIX}_{c}" for c in CANDIDATES if c != "probe"):
        ckpt = bench / "runs" / run / "best.pt"
        (r.ok if ckpt.exists() else r.warn)(f"{run} best checkpoint " + ("found" if ckpt.exists() else
                                                                          f"missing in {bench}: M7b applies it"))


def check_output(r: Report, out_dir: Path) -> None:
    r.section("Output directory")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        free = shutil.disk_usage(out_dir).free / 1e9
        (r.ok if free >= 10 else r.warn)(f"{out_dir} writable, {free:.0f} GB free (need ~10 GB)")
    except OSError as exc:
        r.fail(f"{out_dir}: {exc}")
        return
    manifest = out_dir / "manifest.parquet"
    if manifest.exists():
        import pandas as pd

        m = pd.read_parquet(manifest, columns=["image"])
        sample = m["image"].sample(n=min(200, len(m)), random_state=0)
        missing = sum(not (out_dir / p).exists() for p in sample)
        (r.ok if missing == 0 else r.fail)(f"manifest: {len(m):,} studies; {missing} of {len(sample)} sampled PNGs missing")


def check_hub(r: Report, load_models: bool, device: str) -> None:
    r.section("Hugging Face models")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    from .encoders import REAL_ENCODERS, REGISTRY

    for name in REAL_ENCODERS:
        spec = REGISTRY[name]
        if spec.source == "torchvision":
            continue
        try:
            hf_hub_download(spec.source, "config.json")
            r.ok(f"{name}: {spec.source} accessible")
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if spec.source == "google/medsiglip-448" and ("401" in text or "403" in text or "gated" in text.lower()):
                r.warn("medsiglip: no access. Accept the terms at https://huggingface.co/google/medsiglip-448 "
                       "and run `hf auth login` (needed for M2 only)")
            else:
                r.fail(f"{name}: {type(exc).__name__}: {text[:200]}")
    if load_models:
        import numpy as np
        import torch
        from PIL import Image

        from .common import pick_device
        from .encoders import load_encoder

        dev = pick_device(device)
        img = Image.fromarray((np.random.default_rng(0).random((300, 260)) * 255).astype(np.uint8))
        for name in REAL_ENCODERS:
            try:
                enc = load_encoder(name, 224, dev)
                with torch.inference_mode():
                    feats = enc(enc.eval_transform()(img)[None].to(dev))
                    if hasattr(enc, "encode_text"):
                        enc.encode_text(["pleural effusion", "no pleural effusion"])
                finite = bool(torch.isfinite(feats).all())
                (r.ok if finite and feats.shape == (1, enc.dim) else r.fail)(
                    f"{name}: loads and embeds one image on {dev} -> {tuple(feats.shape)}")
                del enc
            except Exception as exc:  # noqa: BLE001
                (r.warn if name == "medsiglip" else r.fail)(f"{name}: load test failed: {type(exc).__name__}: {str(exc)[:200]}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mimic-cxr-jpg", help="MIMIC-CXR-JPG root (default: MIMIC_CXR_JPG_PATH)")
    parser.add_argument("--cohort-dir", help="v1 cohort dir (default: COHORT_DIR, else $OUTPUT_PATH/cohorts if it exists)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--sample", type=int, default=50, help="JPGs to test-read")
    parser.add_argument("--load-models", action="store_true", help="Download and run each encoder on one image")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    load_dotenv()
    root = args.mimic_cxr_jpg or os.environ.get("MIMIC_CXR_JPG_PATH")
    cohort = args.cohort_dir or os.environ.get("COHORT_DIR")
    if not cohort:
        default = Path(os.environ.get("OUTPUT_PATH", "output")) / "cohorts"
        cohort = str(default) if (default / "anomalous_train.parquet").exists() else None
    out_dir = Path(args.out) if args.out else default_out_dir()

    r = Report()
    check_packages(r)
    check_device(r)
    check_mimic(r, Path(root).expanduser() if root else None, args.sample, Path(cohort) if cohort else None)
    check_cohort(r, Path(cohort) if cohort else None)
    check_output(r, out_dir)
    check_context_sources(r, out_dir)
    check_m4b(r, out_dir, Path(root).expanduser() if root else None)
    check_m7(r, Path(root).expanduser() if root else None)
    check_hub(r, args.load_models, args.device)
    print(f"\n{r.failures} failed, {r.warnings} warnings")
    sys.exit(1 if r.failures else 0)


if __name__ == "__main__":
    main()
