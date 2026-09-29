# Test Suite

## Overview

Unit, regression and end-to-end tests for the training pipeline. Tests run the
project's real code (models, losses, `train_classifier.train_epoch`,
`src/training`, the CLI `main()` functions) rather than re-implementing it. The
classifier is built at `img_size=32` so the real ViT-B + BERT stack runs on CPU
in seconds.

## Test Structure

```
tests/
├── conftest.py                    # Fixtures, incl. a tiny on-disk preprocessed cohort
├── test_training_fixes.py         # Regression tests for the September 2026 fixes
├── test_multimodal_stability.py   # Training-loop stability (skipping bad batches)
├── test_nan_handling.py           # Numerics of cross-attention, normalization, MAE
├── utils/__init__.py              # Helpers: make_classifier, make_batch, make_training
├── baseline_metrics.md            # Historical cascade-failure analysis
└── README.md                      # This file
```

## Test Coverage

### test_training_fixes.py
- **Optimizer coverage**: every trainable parameter (frozen and unfrozen MAE,
  `text_clip_proj`, CLIP logit scale) is in the optimizer. Unfreezing leaves the
  decoder and fixed position embeddings frozen.
- **Non-finite gradients**: one Inf gradient costs one skipped step. A
  non-finite update is never applied, and a long run of skips raises. MAE
  epochs skip NaN losses.
- **Schedule**: `epochs == warmup_epochs` works, the MAE encoder re-warms after
  unfreezing, and per-block LR decay matches the docs.
- **Text branch**: the encoder matches the preprocessing tokenizer, and a frozen
  BERT has no dropout.
- **Cross-attention**: query/key projections get gradients, padding is ignored,
  empty text stays finite, and NaN propagates to the loss check.
- **Structured features**: fp16-safe normalization, missingness indicators,
  saved statistics, censored lab values, procalcitonin removed.
- **Dataset / inference**: raw NaN structured values, token vocabulary check,
  full-view resize vs. legacy center crop, `detect_anomalies` preprocessing
  matches training, frontal view chosen from metadata.
- **Resume / CLI**: config round-trips through checkpoints, zero-valued
  overrides apply, resume rejects schedule changes, and resume restores
  epoch/best/history/loss state. Legacy MAE checkpoints keep center crop, and
  `cuda:N` enables AMP.
- **End to end**: `train_classifier.py` trains and then resumes using the
  documented `--resume` command; `train_mae.py` trains and `detect_anomalies.py
  --hdf5` scores its output.

### test_multimodal_stability.py
- NaN batches are skipped before backward. Inf gradients are never applied.
  Forced corruption every 10 batches doesn't stop training. Under CUDA, AMP
  overflow backs off the loss scale (skipped without CUDA).
- Dataset sanitizes NaN/Inf pixels, and missing structured values encode to
  finite features.
- Forward-pass throughput and output-collapse checks.

### test_nan_handling.py
- Cross-attention stays finite for zero/large inputs and propagates NaN/Inf.
- `safe_normalize` and CLIP loss with zero vectors (Fix #3).
- MAE patch-normalization epsilon (Fix #4).

## Running Tests

The classifier tests download `emilyalsentzer/Bio_ClinicalBERT` from the
Hugging Face Hub on first use.

```bash
# All tests
python -m pytest tests/ -v

# One file / class
python -m pytest tests/test_training_fixes.py -v
python -m pytest tests/test_training_fixes.py::TestOptimizerCoverage -v

# Coverage
python -m pytest tests/ --cov=src --cov-report=html
```

`test_amp_overflow_skips_step_and_backs_off_scale` needs CUDA and is skipped
without a GPU; `test_forced_corruption_every_10_batches` runs with mixed
precision when CUDA is available and in full precision otherwise.

## Contributing

When adding new tests:
1. Exercise the project's code, not a copy of its logic.
2. Use descriptive test names (`test_<what>_<scenario>_<expected>`).
3. Add reusable fixtures to `conftest.py` and helpers to `tests/utils`.
4. Keep large artifacts (checkpoints) out of pytest's temp dirs, or delete them
   in the test: classifier checkpoints are ~1 GB each.
