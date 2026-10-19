# v2 Plan: Modern Foundation Models for CheXpert Classification

**Status:** M0 done (G0 PASS); M1 pilot at 224 px and M2 done; G1 pending a 518 px rerun with a wider C grid (§6, §9) · how to run: [`v2/README.md`](../v2/README.md) · **Branch:** `v2` (from `main`) · **Started:** 2026-10-18

## 1. Goal

Find out how well modern pretrained chest X-ray models do on our task, at
minimal cost, with an evaluation we can trust.

v2 answers three questions, cheapest first:

1. **Zero-shot:** how good is a medical vision-language model with no training at all?
2. **Frozen embeddings:** how good is a pretrained encoder plus a linear classifier?
3. **Fine-tuning:** how much does one short GPU fine-tune add on top?

Not in v2.0: text inputs, cross-attention fusion, MAE pretraining, report
generation, anything that needs more than one GPU session. These are in the
backlog (§10).

## 2. Starting point: what v1 tells us

- v1 reported macro AUROC **0.701** over 12 classes (December 2024 model,
  `docs/RESULTS.md`).
- The per-class table in `RESULTS.md` has its **last four names shifted by
  one** (verified against MIMIC-CXR's published label statistics):

  | Name in RESULTS.md | Actually is | AUROC |
  |---|---|---|
  | No_Finding | Pleural Effusion | 0.821 |
  | Pleural_Effusion | Pleural Other | 0.326 |
  | Pleural_Other | Pneumonia | 0.751 |
  | Pneumonia | Pneumothorax | 0.812 |

- On the **8 classes with ≥30 negatives** in v1's validation set, v1 averages
  **0.792**. The other 4 rest on 2-23 negative studies and are noise.
- That 0.792 may be optimistic: labs drawn after the X-ray, discharge diagnosis
  codes in the text, and the best epoch picked on the reported set.
- The code review behind this lives on branch `claude/gifted-lovelace-1vr4rs`
  (`docs/NEXT_ITERATION_PLAN.md`, `docs/CHANGELOG.md`). v2 does not depend on
  that branch.

**So the bar is about 0.79 on those 8 classes.** Beating it is nice but
secondary. The primary goal is a clean number we believe.

## 3. The task (fixed for all of v2)

| Item | Definition |
|---|---|
| Population | v1's abnormal cohort: the studies in `output/cohorts/anomalous_{train,val}.parquet` (~32.5k studies), de-duplicated by `study_id`. It is a non-random subset of abnormal studies (see below). |
| Image | One frontal image per study: PA if present, else AP, from `ViewPosition` in `mimic-cxr-2.0.0-metadata.csv.gz` (ties: lowest `dicom_id`). Studies with no frontal image are dropped and counted. |
| Inputs | The image only. Nothing recorded after the X-ray, ever. |
| Labels | The 12 CheXpert pathologies, in v1 order: Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax |
| Label policy | `1` → positive; `0` or blank → negative; `-1` (uncertain) → ignored for that label |

Why the v1 cohort is a non-random subset: v1's ED linking
(`src/cohort/builder.py`, `_try_ed_linking`) drops every study of a patient
with ED visits unless the X-ray falls within ±24 h of one. v1 also parses
`StudyTime` with `str[:6]`. If the CSV stores times before 10:00 without
their leading zero (it stores them as numbers), that parse fails, those
studies get no datetime, and they are dropped too for patients with ED visits.
Public code that parses the same file left-pads `StudyTime` to six digits
([AEHRC CXRMate-ED, `prepare_dataset.py`](https://huggingface.co/aehrc/cxrmate-ed/blob/80559a6d52f801981e06dc8ee087a7a9230fecfd/prepare_dataset.py)),
which suggests it does. On times in that format, v1's parser returns no
datetime for 100% of studies before 10:00 (about 40% of all studies), while
v2's parser handles them.

**Confirmed by the M0 audit (2026-10-19).** The cohort holds 32,498 of
MIMIC-CXR's 138,707 abnormal studies (23.4%). Only 0.01% of its studies were
acquired before 10:00, against 51.8% of the abnormal studies outside it.
Nearly every MIMIC-CXR patient has ED visits, so v1's cohort is in effect
"abnormal studies within ±24 h of an ED visit, acquired between 10:00 and
midnight". Early-morning inpatient and ICU films are missing. Every v1 number,
and every v2.0 number, describes that slice. The audit also confirmed that
31.3% of v1's training images were non-frontal (v1 picked views by filename,
which never matched).

We keep the cohort for v2.0, which was fixed before any data was seen and is
needed for the v1 comparison. Backlog item 1 replaces it.

### 3.1 Splits (all by patient, `subject_id`)

| Split | Definition | Used for |
|---|---|---|
| `test_clean` | Cohort studies of patients in the official MIMIC-CXR `validate` or `test` split | **Headline results.** Never trained on, never used to make choices. |
| `test_v1` | v1's validation split (`anomalous_val.parquet`, 4,922 studies) | Side-by-side comparison with v1's reported numbers |
| `val` | 10% of the remaining patients (seed 42) | Model and hyperparameter selection |
| `train` | The remaining patients | Training |

Why two test sets: RAD-DINO was pretrained (without labels) on every
official-train MIMIC-CXR image and excluded the official validate/test images.
Only `test_clean` is clean for it. `test_v1` exists only to compare with v1.
The v1 checkpoint itself can't be scored on `test_clean`. v1 split its cohort
by patient at random, so it trained on most official validate/test patients.

Fallback, decided now and used only if needed: if `test_clean` has fewer than
6 measurable classes (§3.2), add every other frontal study of the same
official validate/test patients that has at least one positive pathology,
taken directly from the MIMIC-CXR CSVs.

### 3.2 Metrics

- Per-class AUROC with a 95% CI from a patient-level bootstrap (1,000 resamples).
- **Headline:** macro AUROC over each test set's *measurable classes*, those
  with ≥30 positive and ≥30 negative studies. The lists are computed in M0
  and then frozen.
- Model comparisons: paired bootstrap of the difference in macro AUROC (same
  resampled patients for both models).
- Also reported: AUPRC next to prevalence, and per-class counts.
- On `test_v1` only, the v1 metric too (blank labels ignored as well), over
  v1's 8 measurable classes, to compare with 0.792.

## 4. Models

| Model | What it is | Access | Role in v2 |
|---|---|---|---|
| RAD-DINO (`microsoft/rad-dino`) | ViT-B/14; DINOv2 continued on ~883k chest X-rays, image-only self-supervised; 518 px | Open (MIT) | Strongest expected chest X-ray encoder, frozen and fine-tuned |
| MedSigLIP (`google/medsiglip-448`) | Medical vision-language model (400M image + 400M text encoder); 448 px | Gated (Health AI Developer Foundations terms) | Zero-shot classification with text prompts; second frozen encoder |
| DINOv2-B (`facebook/dinov2-base`) | Same architecture as RAD-DINO, natural images only | Open (Apache-2.0) | Control: shows what chest X-ray pretraining adds |
| DenseNet-121, ImageNet init (torchvision) | Classic chest X-ray baseline; 0.834 mean AUROC on MIMIC-CXR (all studies, 14 labels, 256 px) in CheXclusion | Open | Fine-tuning reference point |

What each model may have seen:

- **RAD-DINO:** official-train MIMIC-CXR images. M0 checks its published
  `training_images.csv` to confirm that no `test_clean` image is in it.
- **MedSigLIP:** MIMIC-CXR images *and reports*; the split isn't documented.
  Google's MedGemma report evaluates on the MIMIC-CXR test set, which suggests
  it was held out. Report MedSigLIP on `test_clean` only, flagged.

## 5. The three techniques, briefly

- **Zero-shot (M2).** A vision-language model scores how well an image
  matches a sentence. For each class, compare "pleural effusion" with "no
  pleural effusion"; the difference is the prediction. No labels or training
  are involved, so it measures what the model already knows.
- **Frozen embeddings + linear probe (M1).** Run every image through a
  pretrained encoder once and keep its embedding vector. Then fit one logistic
  regression per class on those vectors. It's cheap (a single pass over the
  images, then seconds on CPU), hard to get wrong, and the standard way to
  compare encoders.
- **Fine-tuning (M3).** Train the whole encoder end to end on our labels at a
  low learning rate. On in-domain tasks it usually adds a few AUROC points over
  a linear probe. It's the only step that needs a GPU.

## 6. Milestones

Each milestone has a question, steps, a definition of done, and a decision.
Results go in the log (§9).

### M0: Data and audit (your machine, no GPU, under 1 hour)

**Question:** is the data right, and can the test sets measure what we care about?

1. `python -m v2.prepare_data` reads the v1 cohort files,
   `mimic-cxr-2.0.0-{metadata,split,chexpert}.csv.gz` and the JPGs.
2. It writes `output/v2/manifest.parquet`: one row per study, with `study_id`,
   `subject_id`, `dicom_id`, view, split, the 12 raw labels (1 / 0 / -1 /
   blank) and the image path.
3. It writes `output/v2/images/{study_id}.png`: 8-bit grayscale, shorter side
   518 px (about 5-8 GB in total). Every later step reads these, never the
   original JPGs or v1's HDF5.
4. It writes `output/v2/audit.md`:
   - studies and patients per split;
   - positive, negative and uncertain counts per class and split;
   - each test set's measurable classes;
   - patient overlap between splits (must be 0);
   - studies dropped for having no frontal image;
   - `test_clean` images found in RAD-DINO's `training_images.csv` (must be 0);
   - optionally (`--v1-hdf5 <dir>`), the share of lateral views among v1's
     training images.

**Done when:** the audit exists, both overlap checks are 0, and the
measurable-class lists are saved to `output/v2/measurable_classes.json`.

**Gate G0:** continue if `test_clean` has ≥6 measurable classes. Otherwise
apply the fallback in §3.1 and re-run the audit.

### M1: Frozen embeddings + linear probes (CPU pilot first)

**Question:** how good are modern encoders with no fine-tuning?

1. `python -m v2.embed --model rad-dino --size 224` on CPU for the pilot, later
   `--size 518` on the GPU. Output: `output/v2/embeddings/{model}_{size}.npz`
   (`study_id`, embedding). For the DINO models the embedding is the CLS token
   concatenated with the mean of the patch tokens; for MedSigLIP it is the
   model's own image embedding.
2. The same for `dinov2-base`, and for `medsiglip` once access is granted.
3. `python -m v2.probe --embeddings ...`: per class, standardize the features
   and fit an L2 logistic regression, choosing C from {0.0001, 0.001, 0.01, 0.1, 1, 10} by
   `val` AUROC. Output: `output/v2/predictions/{run}.parquet`.
4. `python -m v2.evaluate` writes `output/v2/reports/{run}.md`.

**Cost:** about an hour per model at 224 px on an 8-core CPU (a 10k-study
training subset halves it), or minutes on a GPU.

**Done when:** predictions and a report exist for RAD-DINO and DINOv2 on both
test sets.

**Decisions:**
- *Sanity check:* RAD-DINO's probe should beat DINOv2's. If it doesn't, suspect
  preprocessing (normalization, orientation) before anything else.
- **Gate G1:** if the best frozen probe on `test_v1` (v1 metric, 8 classes) is
  **≥ 0.77**, rent the GPU for M3. Otherwise stop and study the per-class
  results first. (RAD-DINO saw the `test_v1` images, without labels, during
  pretraining. That's acceptable for a go/no-go, but the result isn't a
  headline number.) MedSigLIP is not eligible: its `test_v1` numbers are not
  reportable (§4).
- *Amendment, 2026-10-19, after the 224 px pilot:* the C grid was widened from
  {0.01, 0.1, 1, 10}, because every class of both pilot probes chose 0.01, with
  val AUROC falling as C grew. C is chosen on val only, so the change never
  uses a test set. G1 is decided once, on the best-by-val probe of RAD-DINO or
  DINOv2 at 518 px (the plan's intended M1 resolution, run on the Mac instead of
  the GPU), with the wider grid. Whatever that gives is final.

### M2: Zero-shot with MedSigLIP (optional; CPU or GPU)

**Question:** what does a medical vision-language model know out of the box?

1. Accept the MedSigLIP terms on Hugging Face and log in (`hf auth login`).
2. `python -m v2.zero_shot`: for each class, score =
   similarity(image, "<class>") − similarity(image, "no <class>"). Choose
   between 2-3 prompt templates on `val` only, never on a test set.
3. Evaluate on `test_clean`.

**Done when:** a zero-shot report sits next to M1's.

**Decision:** none; it's a reference point. It's worth a closer look if it
comes close to the probes.

### M3: Fine-tune the best encoder (the one GPU run: ~2-3 h, ~$5-10)

**Question:** how much does fine-tuning add over the best frozen probe?

Recipe (deliberately standard; no tuning beyond what's listed):

- **Model:** the M1 winner (expected: RAD-DINO), with a linear head on CLS +
  mean patch token.
- **Input:** 518 px for RAD-DINO, 512 px for DenseNet.
- **Augmentation:** random resized crop (scale 0.8-1.0), rotation ±10°,
  brightness/contrast ±10%. No horizontal flip (it moves the heart).
- **Loss:** binary cross-entropy, ignoring uncertain labels.
- **Optimizer:** AdamW, weight decay 0.05; learning rate 1e-3 for the head
  and 2e-5 for the encoder, with layer-wise decay 0.75.
- **Schedule:** 1 warmup epoch, then cosine, for 10 epochs.
- **Batch:** 32 (16 × 2 gradient accumulation on a 40 GB GPU); bf16;
  gradient clipping 1.0.
- **Selection:** keep the epoch with the best `val` macro AUROC, then predict
  on the test sets once, at the end.
- **Runs:** 2 seeds, plus DenseNet-121 with the same recipe (learning rate
  1e-4) as the classic reference.

**Done when:** predictions, reports and the best checkpoint are downloaded and
the instance is deleted.

**Gate G2:** the fine-tuned model becomes the v2 baseline if it beats the best
frozen probe on `test_clean` (paired 95% CI of the difference above 0).
Otherwise the frozen probe is the baseline: simpler, and as good.

### M4: Add triage vitals (CPU, optional)

**Question:** do pre-imaging vitals add anything to the image?

1. Features: the image embedding (frozen by default; fine-tuned embeddings
   were fit on the training studies, which would understate what vitals add),
   plus age, sex, and triage temperature, heart rate, respiratory rate, SpO2,
   blood pressure and acuity, with missing-value indicators. These come from
   the triage columns of v1's cohort files, not `structured.parquet`, whose
   aggregates include values recorded after the X-ray.
2. Vitals count only when the ED arrival time is at or before the X-ray;
   otherwise they are missing. All studies stay in, so both models are scored
   on the same test studies.
3. Fit a logistic regression and compare paired against image-only.

**Done when:** the fusion report exists. Vitals are adopted only if the paired
CI excludes 0.

## 7. Code

A standalone package, `v2/`, on this branch. It doesn't touch the v1 pipeline.
How to run everything: [`v2/README.md`](../v2/README.md).

```
v2/
  common.py         # label list, label policy, splits, paths, devices, prediction files
  prepare_data.py   # M0: manifest, resized images, audit (incl. RAD-DINO overlap, v1 cohort coverage)
  data.py           # PyTorch dataset over the manifest's PNGs
  encoders.py       # model name -> loader, preprocessing, embeddings (+ text for SigLIP)
  embed.py          # M1: embeddings -> .npz (resumable)
  probe.py          # M1: per-class logistic regression -> predictions
  zero_shot.py      # M2
  finetune.py       # M3: plain PyTorch loop, bf16, checkpoint every epoch, --resume
  fuse.py           # M4
  evaluate.py       # metrics, bootstrap CIs, paired comparisons, reports, gates
  doctor.py         # setup check for any machine
  synth.py          # synthetic MIMIC-CXR-JPG tree with a planted signal (tests)
  smoke.py          # end-to-end wiring check on synthetic data
  scripts/setup.sh  # environment (laptop or GPU), tests, wiring check
  scripts/run.sh    # stages with logs and time limits; `run.sh gpu` is the M3 runbook
  requirements.txt  # transformers pinned (5.20.1)
tests/v2/           # CPU tests on synthetic data and tiny random models; real-model tests opt-in
```

Conventions:

- Every experiment writes predictions in one format (`study_id`, `split`,
  `label`, `score`, `run`, plus a JSON sidecar). `evaluate.py` is the only
  place metrics are computed, so every model is scored identically.
- `embed` and `finetune` have `--smoke`: a handful of real studies, minutes on
  CPU, to test a machine. `python -m v2.smoke` runs every stage on synthetic
  data in about 30 s.
- Paths come from the existing `.env` (`MIMIC_CXR_JPG_PATH`, `OUTPUT_PATH`, plus `COHORT_DIR`).
- transformers is pinned in `v2/requirements.txt`: DINOv2's position-embedding
  interpolation (any size other than 518 px) differs between 4.x and 5.x, so
  embeddings depend on the version. Each embedding and prediction file records
  the torch and transformers versions.

Tests (all must pass before any GPU time):

| Test | Checks |
|---|---|
| Manifest | Frontal selection (PA over AP, laterals never chosen); label policy; splits disjoint by patient; `test_clean` and `test_v1` definitions; drops counted; RAD-DINO overlap check; `--extend-test-clean` |
| Encoders | Embedding shape and CLS + mean-patch definition with tiny random models (no downloads); transforms; SigLIP text side |
| Probe | Recovers a planted signal; alignment of embeddings with studies |
| Evaluate | AUROC against scikit-learn with ties; known answers; whole-patient bootstrap; CIs narrow with n; paired comparison; v1 vs v2 metric |
| Fine-tune | Layer-wise lr decay and no-decay sets; schedule; masked loss; checkpoints, predictions and resume |
| Wiring check | Every stage on synthetic data through the CLI; planted signal recovered (probe ~1.0, fine-tune > 0.7); misaligned embeddings fall to chance |
| Real models (opt-in) | RAD-DINO and DINOv2 load fully; v2 preprocessing matches their image processors; SigLIP path with the real tokenizer |
| Local rehearsal (you run it) | `prepare_data --limit 200`, then `run.sh smoke` on your machine |

## 8. GPU runbook (M3)

Step-by-step commands: [`v2/README.md`](../v2/README.md) §3.

1. Rent 1× A100 or H100 (40-80 GB) with CUDA 12 (about $1.3-2.5/h).
2. `bash v2/scripts/setup.sh`: venv (keeps the provider's CUDA torch), install,
   `pytest tests/v2`, wiring check. Then `hf auth login` for MedSigLIP.
3. Upload `output/v2/manifest.parquet`, `measurable_classes.json` and
   `images/` (5-8 GB, 10-20 min).
4. `bash v2/scripts/run.sh gpu`:
   1. a 5-minute smoke test on real data (aborts on any failure);
   2. full-resolution embeddings for M1 and M2;
   3. two RAD-DINO fine-tunes (2 seeds);
   4. DenseNet-121;
   5. evaluation.

   Each stage has a time limit, and the whole run aborts at 4 hours.
5. Download `output/v2/{predictions,reports}` and the best checkpoint. Delete
   the data, then terminate the instance.

MIMIC data stays under your PhysioNet credential: use a provider you trust,
keep the data on the machine only for the run, and delete it before
terminating.

## 9. Results log

Add one row per run. Don't edit old rows; add new ones.

| Date | Milestone | Commit | Run | Test set | Macro AUROC [95% CI] | Classes | Notes / decision |
|---|---|---|---|---|---|---|---|
| 2024-12 | v1 (reference) | 8166273 | v1 multimodal | `test_v1`, v1 metric | 0.792 (8 measurable); 0.701 (all 12) | 8 / 12 | Possibly optimistic (§2) |
| 2026-10-19 | M0 | 0abc51b | audit | `test_clean` 870 studies / 341 patients; `test_v1` 4,860 / 2,947 | n/a | 9 (v2) / 8 (`test_v1`, v1 metric) | **G0 PASS.** RAD-DINO overlap with `test_clean`: 0 images, 0 patients. Splits patient-disjoint. 412 studies dropped (no frontal image), 0 unreadable. The v1-metric classes on `test_v1` are exactly v1's 8 (the other 4 have 2-23 explicit negatives, as in §2). Thin `test_clean` classes: Enlarged Cardiomediastinum 41, Lung Lesion 47 and Consolidation 60 positives, so expect wide CIs. v1 cohort bias confirmed (§3). |
| 2026-10-19 | M1 | d2a34e4 | rad-dino_224_probe | `test_clean` | 0.739 [0.719, 0.757] | 9 | Old C grid; every class chose C=0.01 (lower edge) |
| 2026-10-19 | M1 | d2a34e4 | rad-dino_224_probe | `test_v1`, v1 metric | 0.768 [0.752, 0.785] | 8 | G1 basis at 224 px: 0.002 below 0.77, CI spans it |
| 2026-10-19 | M1 | d2a34e4 | dinov2-base_224_probe | `test_clean` | 0.714 [0.696, 0.733] | 9 | Old C grid, lower edge. RAD-DINO beats it on every set (sanity check passes) |
| 2026-10-19 | M1 | d2a34e4 | dinov2-base_224_probe | `test_v1`, v1 metric | 0.720 [0.702, 0.739] | 8 | |
| 2026-10-19 | M1 | d2a34e4 | medsiglip_448_probe | `test_clean` | 0.778 [0.760, 0.793] | 9 | Flagged (§4). Best by val (0.799). `test_v1` (0.807) not reportable |
| 2026-10-19 | M2 | d2a34e4 | medsiglip_448_zeroshot | `test_clean` | 0.666 [0.646, 0.687] | 9 | Flagged (§4). Template chosen on val. Far below the probe |
| 2026-10-19 | G1 | d2a34e4 | gate (224 px) | `test_v1`, v1 metric | 0.768 | 8 | Not decided. The summary's PASS used MedSigLIP, which §4 excludes (gate code fixed). Clean probes were limited by the C grid edge. Decision deferred to the 518 px rerun with the wider grid (§6 amendment) |

## 10. Backlog (after v2.0, one at a time)

Ordered by expected value per unit of cost:

1. **More data, unbiased population:** train on all frontal MIMIC-CXR studies of
   train-split patients, not just the v1 cohort (~4× more abnormal studies,
   ~6× more studies in total). Likely the biggest single gain. It also removes
   the cohort's time-of-day and ED-window selection (§3), and grows
   `test_clean` beyond 870 studies.
2. **Higher resolution** (RAD-DINO at 1024 px with interpolated position
   embeddings) for subtle findings such as pneumothorax and fracture.
3. **Parameter-efficient fine-tuning** (LoRA) of MedSigLIP's image encoder.
4. **Both views:** use the lateral image alongside the frontal one.
5. **Anomaly detection** (normal vs. abnormal) from foundation embeddings, as
   the distance to the nearest normal studies. Uses the normal cohort and
   revisits the repo's original goal.
6. **Clinical text** (chief complaint) through a small text encoder, only if
   M4 shows that pre-imaging data helps.
7. Ensembles and test-time augmentation.

## 11. Prerequisites

- [ ] A machine with MIMIC-CXR-JPG, the v1 cohort files, Python 3.10+,
      8+ CPU cores and ~10 GB free disk
- [ ] A Hugging Face account, with the MedSigLIP terms accepted (for M2)
- [ ] AI assistants never see record-level MIMIC data (`v2/README.md` §9)
- [ ] A GPU provider account (for M3)
- [ ] Agreement on the task definition (§3), the label policy (§3) and the
      gates G0-G2 (§6)

## References

- MIMIC-CXR-JPG (label statistics, official splits): <https://arxiv.org/abs/1901.07042>
- CheXclusion (DenseNet-121 baseline on MIMIC-CXR): <https://arxiv.org/abs/2003.00827>
- RAD-DINO: <https://arxiv.org/abs/2401.10815>, <https://huggingface.co/microsoft/rad-dino>
- MedSigLIP and MedGemma: <https://arxiv.org/abs/2507.05201>, <https://huggingface.co/google/medsiglip-448>
- DINOv2: <https://arxiv.org/abs/2304.07193>
