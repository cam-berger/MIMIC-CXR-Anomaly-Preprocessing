"""
Shared definitions for v2: labels, label policy, splits, paths, devices and
the one prediction-file format every experiment writes.

Everything that must be identical across experiments lives here, so the
evaluation protocol in docs/V2_PLAN.md §3 is defined in exactly one place.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("v2")

# The 12 CheXpert pathologies, in v1 order (src/datasets/mimic_cxr.py).
LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
)

# Manifest `split` values. Test studies are further marked by the
# `in_test_clean` / `in_test_v1` flags; the two test sets may overlap.
TRAIN, VAL, TEST = "train", "val", "test"
TEST_SETS: tuple[str, ...] = ("test_clean", "test_v1")
TEST_FLAG = {"test_clean": "in_test_clean", "test_v1": "in_test_v1"}

# Measurable-class rule (§3.2) and the looser rule used for in-training
# validation, where a missing class only weakens model selection.
MIN_COUNT = 30
MIN_COUNT_VAL = 10

DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Label policy
# ---------------------------------------------------------------------------

def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Raw CheXpert values (1 / 0 / -1 / NaN) as an (n, 12) float array."""
    return df[list(LABELS)].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)


def targets_and_mask(raw: np.ndarray, metric: str = "v2") -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a label policy to raw CheXpert values.

    v2 (training and headline metric): 1 -> positive; 0 or blank -> negative;
        -1 (uncertain) -> ignored.
    v1 (comparison with v1's reported numbers only): 1 -> positive;
        0 -> negative; -1 or blank -> ignored.

    Returns (targets float32 in {0, 1}, mask bool); targets are 0 where the
    mask is False.
    """
    raw = np.asarray(raw, dtype=np.float64)
    positive = raw == 1.0
    if metric == "v2":
        mask = raw != -1.0  # NaN != -1 is True: blanks count as negatives
    elif metric == "v1":
        mask = (raw == 1.0) | (raw == 0.0)
    else:
        raise ValueError(f"unknown label policy {metric!r} (expected 'v2' or 'v1')")
    return (positive & mask).astype(np.float32), mask


def label_counts(raw: np.ndarray, metric: str = "v2") -> pd.DataFrame:
    """Positive / negative / ignored counts per label under a policy."""
    y, m = targets_and_mask(raw, metric)
    pos = (y.astype(bool) & m).sum(axis=0)
    neg = (~y.astype(bool) & m).sum(axis=0)
    return pd.DataFrame(
        {"label": LABELS, "pos": pos, "neg": neg, "ignored": (~m).sum(axis=0)}
    )


def measurable(raw: np.ndarray, metric: str = "v2", min_count: int = MIN_COUNT) -> list[str]:
    """Labels with at least `min_count` positives and negatives (§3.2)."""
    counts = label_counts(raw, metric)
    ok = (counts["pos"] >= min_count) & (counts["neg"] >= min_count)
    return counts.loc[ok, "label"].tolist()


# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------

def load_dotenv(path: Optional[Path] = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ without overriding."""
    candidates = [path] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return


def default_out_dir() -> Path:
    """`V2_OUT`, else `$OUTPUT_PATH/v2`, else ./output/v2."""
    load_dotenv()
    if os.environ.get("V2_OUT"):
        return Path(os.environ["V2_OUT"])
    return Path(os.environ.get("OUTPUT_PATH", "output")) / "v2"


def resolve_mimic_root(cli_value: Optional[str]) -> Path:
    """The MIMIC-CXR-JPG root: --mimic-cxr-jpg, else MIMIC_CXR_JPG_PATH."""
    load_dotenv()
    value = cli_value or os.environ.get("MIMIC_CXR_JPG_PATH")
    if not value:
        raise SystemExit(
            "MIMIC-CXR-JPG location unknown: pass --mimic-cxr-jpg or set "
            "MIMIC_CXR_JPG_PATH (environment or .env)."
        )
    return Path(value).expanduser()


def mimic_csv(root: Path, name: str) -> Path:
    """`mimic-cxr-2.0.0-{name}.csv.gz`, or the uncompressed .csv if that is what exists."""
    gz = root / f"mimic-cxr-2.0.0-{name}.csv.gz"
    return gz if gz.exists() or not (root / gz.stem).exists() else root / gz.stem


def jpg_path(root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    sid = str(int(subject_id))
    return root / "files" / f"p{sid[:2]}" / f"p{sid}" / f"s{int(study_id)}" / f"{dicom_id}.jpg"


def manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / "manifest.parquet"


def load_manifest(out_dir: Path) -> pd.DataFrame:
    path = manifest_path(out_dir)
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m v2.prepare_data` first.")
    return pd.read_parquet(path)


def load_measurable(out_dir: Path) -> dict:
    path = Path(out_dir) / "measurable_classes.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m v2.prepare_data` first.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Reproducibility and devices
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(requested: str = "auto"):
    """cuda > mps > cpu, unless a device is named explicitly."""
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device, requested: str = "auto"):
    """Mixed-precision dtype: bf16 (or fp16) on CUDA, none on CPU/MPS."""
    import torch

    if requested == "fp32" or device.type != "cuda":
        return None
    if requested == "fp16":
        return torch.float16
    if requested == "bf16" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Predictions: the one format every experiment writes (§7)
# ---------------------------------------------------------------------------

PRED_COLUMNS = ("study_id", "split", "label", "score", "run")


def write_predictions(
    out_dir: Path,
    run: str,
    study_ids: Sequence[int],
    splits: Sequence[str],
    scores: np.ndarray,
    meta: Optional[dict] = None,
) -> Path:
    """
    Write `predictions/{run}.parquet` (long format: one row per study and
    label) and a `{run}.json` sidecar describing how it was produced.
    """
    scores = np.asarray(scores, dtype=np.float64)
    study_ids = np.asarray(study_ids, dtype=np.int64)
    if scores.shape != (len(study_ids), len(LABELS)):
        raise ValueError(f"scores shape {scores.shape} != ({len(study_ids)}, {len(LABELS)})")
    if not np.isfinite(scores).all():
        raise ValueError(f"run {run!r}: {np.size(scores) - np.isfinite(scores).sum()} non-finite scores")
    if len(np.unique(study_ids)) != len(study_ids):
        raise ValueError(f"run {run!r}: duplicate study_ids in predictions")

    n = len(study_ids)
    df = pd.DataFrame(
        {
            "study_id": np.repeat(study_ids, len(LABELS)),
            "split": np.repeat(np.asarray(splits, dtype=object), len(LABELS)),
            "label": np.tile(np.asarray(LABELS, dtype=object), n),
            "score": scores.reshape(-1),
            "run": run,
        }
    )
    pred_dir = Path(out_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{run}.parquet"
    df.to_parquet(path, index=False)

    sidecar = {
        "run": run,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "n_studies": int(n),
        "splits": {s: int(c) for s, c in pd.Series(list(splits)).value_counts().items()},
        **(meta or {}),
    }
    (pred_dir / f"{run}.json").write_text(json.dumps(sidecar, indent=2, default=str))
    logger.info("Wrote %s (%d studies)", path, n)
    return path


def read_predictions(path: Path) -> tuple[pd.DataFrame, dict]:
    """Predictions as a wide frame (study_id, split, one column per label) plus the sidecar."""
    path = Path(path)
    long = pd.read_parquet(path)
    missing = set(PRED_COLUMNS) - set(long.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    wide = long.pivot_table(index=["study_id", "split"], columns="label", values="score", aggfunc="first")
    absent = [label for label in LABELS if label not in wide.columns]
    if absent:
        raise ValueError(f"{path}: no scores for {absent}")
    wide = wide[list(LABELS)].reset_index()
    wide.columns.name = None
    sidecar_path = path.with_suffix(".json")
    meta = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    return wide, meta
