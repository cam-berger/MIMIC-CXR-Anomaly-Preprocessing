# Chest X-ray findings on MIMIC-CXR

An exploratory analysis of 12 findings in MIMIC-CXR chest X-rays (atelectasis,
cardiomegaly, effusion and nine more). It started as a from-scratch multimodal
pipeline (v1) and moved to pretrained chest X-ray encoders, text written before
the X-ray, published benchmarks and a second hospital (v2).

## What we found

| | Macro AUROC |
|---|---|
| v1: multimodal pipeline trained from scratch (its own test set) | 0.792 |
| Pretrained RAD-DINO, frozen, image only (the same test set) | 0.803 |
| Fine-tuned RAD-DINO, image only (MIMIC-CXR benchmark, official split) | 0.807 |
| Fine-tuned RAD-DINO + the indication (benchmark) | 0.812-0.819 |
| Best published model with the same inputs, MMBT (benchmark) | 0.794 |
| Fine-tuned RAD-DINO + all text written before the X-ray (benchmark) | 0.825 |
| The same image model at another hospital (CheXpert test set, five findings) | 0.904 |

- A pretrained encoder beats training from scratch, from the image alone.
- Text written before the X-ray adds about +0.02; how it's combined with the
  image matters little.
- Scored against a radiologist instead of the automatic labeler, every model
  gains about +0.03.
- At Stanford (the CheXpert test set) the model is level with CheXzero and KAD,
  two published models trained on MIMIC-CXR.
- A second training seed moves every benchmark and CheXpert score by 0.002 at
  most. Total GPU cost: about $18.

The full write-up, with every step and its numbers:
[`docs/ANALYSIS.md`](docs/ANALYSIS.md). Why each part of v1 changed, and what
the change bought: [From v1 to v2](docs/ANALYSIS.md#from-v1-to-v2).

## What's here

```
v2/                 the v2 package: data prep, encoders, probes, fine-tuning, text, evaluation, reports
v2/README.md        how to run v2: setup, steps, the GPU session, outputs, troubleshooting
tests/v2/           v2's tests and its synthetic end-to-end check
docs/ANALYSIS.md    the analysis: what we explored and what we found
docs/V1.md          v1: the original pipeline, how to run it, and its results
src/, *.py (root)   v1's code: cohort building, preprocessing, MAE pretraining, the multimodal classifier
tests/              v1's tests
```

## Getting started

```bash
git clone <repo> && cd MIMIC-CXR-Anomaly-Preprocessing
bash v2/scripts/setup.sh && source .venv-v2/bin/activate
python -m v2.smoke        # the whole pipeline on synthetic data, ~2 min, no MIMIC access needed
python -m v2.doctor       # checks the machine and your data paths
```

Then follow [`v2/README.md`](v2/README.md). For v1, see [`docs/V1.md`](docs/V1.md).

## Data

The analysis uses MIMIC-CXR-JPG 2.1.0, MIMIC-CXR's reports, MIMIC-IV-ED and
MIMIC-IV from [PhysioNet](https://physionet.org/), which require credentialed
access and a signed data use agreement, plus the CheXpert test set from Stanford
AIMI. All outputs go to `output/`, which is gitignored. If you use an AI coding
assistant on the data, read
[`v2/README.md` §8](v2/README.md#8-working-on-mimic-data-with-an-ai-assistant)
first.
