# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

An exploratory analysis of 12 CheXpert findings in MIMIC-CXR chest X-rays.

- **v2** (`v2/`, tests in `tests/v2/`) is the current work. It covers pretrained
  encoders (RAD-DINO, MedSigLIP, DINOv2) and text written before the X-ray. It
  scores models on the official MIMIC-CXR split against published results, and
  checks them against a radiologist's labels and on the CheXpert test set.
- **v3** (`v3/`, tests in `tests/v3/`) tests TypeSafe's Jev, a hosted
  classification model, as the final classifier and as a report labeler.
- **v1** (`src/` and the root scripts) is the original from-scratch multimodal
  pipeline. It is kept for reference.

Read first:
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md): the setup, each step (M0-M8) and the
  results.
- [`v2/README.md`](v2/README.md): how to run v2 (setup, `v2/scripts/run.sh`
  steps, the GPU session, outputs, troubleshooting).
- [`docs/V1.md`](docs/V1.md): v1.

## v3: TypeSafe's Jev

`v3/` (tests in `tests/v3/`) tests TypeSafe's Jev, a hosted classification
model, on M5's benchmark. It was tried as the final classifier, as a report
labeler and as a source of training labels, and it didn't beat v2's simpler
tools at any of them.

- [`docs/V3.md`](docs/V3.md) summarizes what we found.
- [`docs/V3_PLAN.md`](docs/V3_PLAN.md) is the pre-registration: each step's
  plan and gates, written before it ran, and the results log in §9. Add new
  steps as new sections. Don't edit a step's plan after it has run. Its
  references to V2_PLAN mean `docs/V2_PLAN.md` at commit 04f5946.
- [`v3/README.md`](v3/README.md) explains how to run each step.

Rules:
- v3 reuses v2 unchanged. It works in its own output dir `V3_OUT`, which
  links into `BENCH_OUT` and never writes there. Metrics come only from
  `v2/evaluate.py`.
- Before and after changing code, run `pytest tests/v3 -q` and
  `python -m v3.smoke`.
- Jev requests carry MIMIC text (and, in v3.1, the X-ray); the user verified
  TypeSafe's zero data retention. The data-handling rules below also cover
  `V3_OUT/jev_cache/`, `predictions/` and `logs/`.
- The radiologist's labels file sits in `output/` (gitignored) when the MIMIC
  drive is away: pass `--labels`.

## Rules for v2

- Metrics are computed only in `v2/evaluate.py`, from the one prediction format
  written by `v2.common.write_predictions`.
- Label policy and splits live in `v2/common.py` and `v2/prepare_data.py`.
- Every model choice uses the validation split, never a test set.
- Text inputs are only what was written before the X-ray: the indication, the ED
  chief complaint, the prior report. The study's own findings and impression are
  never an input; the labels come from them.
- Before and after changing code, run `pytest tests/v2 -q` and
  `python -m v2.smoke` (the synthetic end-to-end check, about 2 minutes).
  Lint with `ruff check --select F,E9 v2/ tests/v2/`.
- transformers is pinned in `v2/requirements.txt`: DINOv2 embeddings at
  non-native sizes differ between versions.

## MIMIC data handling

The PhysioNet data use agreement, and its September 2025 guidance on LLMs,
forbid sending MIMIC data to third-party services. That includes this
assistant.

- Never read, print or list raw MIMIC content into the conversation. That covers
  CSV or parquet rows, images, report text, the v2 manifest, `context.parquet`,
  embeddings (including the `*_tokens*.npy` caches, their `.json` sidecars and
  `text_*.npz`) and predictions.
- Run the v2 commands and relay only their aggregate output: `audit.md`,
  `reports/*.md`, `summary.md`. Record-level diagnostics stay in local files.
- File names under `output/` hold study IDs; check disk use with totals only.
- Never commit `output/` or anything derived from a single study: images, crops,
  reconstructions, tensors, per-study features.
- The CheXpert test set falls under the same rules.

Details: `v2/README.md` §8.

## v1 in brief

```bash
pip install -r requirements.txt && python -m spacy download en_core_sci_md
python build_cohort.py --normal-only        # or --anomalous-only for classification
python preprocess.py --leak-free --workers 8
python train_mae.py --config base
python train_classifier.py --config base
```

Classification preprocessing must use `--leak-free`: CheXpert labels are
extracted from the radiology reports, so report text would leak them. Paths come
from `.env` (`MIMIC_CXR_JPG_PATH`, `MIMIC_IV_PATH`, `MIMIC_IV_ED_PATH`,
`CXR_PRO_PATH`, `OUTPUT_PATH`). Everything else about v1 is in `docs/V1.md`.

## Git

- `main` holds the current work: v2 and v3, plus v1 for reference.
- Don't commit data files (`.h5`, `.parquet`, `.csv`, images) or `.env`.
