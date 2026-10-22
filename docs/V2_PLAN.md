# v2 Plan: Modern Foundation Models for CheXpert Classification

**Status:** v2.0 done (M0-M3). **G2 PASS:** the fine-tuned RAD-DINO is the image-only v2 baseline, 0.787 [0.769, 0.803] on `test_clean`. **M4 done, G4 PASS:** text written before the X-ray adds +0.016 [0.009, 0.024] to frozen image features (§9). Exploratory follow-ups: +0.009 [0.003, 0.014] on the fine-tuned image; refitting the probe on the fine-tuned features adds +0.006 [0.001, 0.012]; best system by val so far: fine-tuned features + refit probe + text, 0.802 [0.784, 0.819]. **M4b done, G4b FAIL:** text steers an attention head's focus but adds nothing detectable over late fusion (+0.004 [-0.003, 0.011]), which argues against a joint image-text fine-tune. The same head on frozen patch tokens matches the fine-tune (0.789), and fit on val, text adds +0.016 to the fine-tuned image (§9). **M5a done** (official MIMIC-CXR split, positive vs rest, like for like): every image-only run beats the published ResNet-50 (0.781-0.797 against 0.725), image + indication ties MMBT (0.798 against 0.794), and the indication alone trails BERT (0.675 against 0.704). Best run: the attention head with all pre-imaging text, 0.818 [0.803, 0.832], not like for like (§9). **M5 done:** like for like on the official split, a fine-tuned RAD-DINO plus the indication beats MMBT, 0.815 [0.801, 0.829] against 0.794 (val combiner, a lower bound; one seed). Fine-tuning on 213k studies adds +0.010 over the frozen attention head, and after fine-tuning the attention head no longer beats mean pooling (+0.002 [-0.002, 0.006]). Best number so far: 0.825 with all pre-imaging text (exploratory). Next: M6 (§6, pre-registered 2026-10-22): BERT-family encoders for the pre-imaging text, general against medical (gate G6), then the CheXpert test set · how to run: [`v2/README.md`](../v2/README.md) · **Branch:** `v2` (from `main`) · **Started:** 2026-10-18

## 1. Goal

Find out how well modern pretrained chest X-ray models do on our task, at
minimal cost, with an evaluation we can trust.

v2 answers three questions, cheapest first:

1. **Zero-shot:** how good is a medical vision-language model with no training at all?
2. **Frozen embeddings:** how good is a pretrained encoder plus a linear classifier?
3. **Fine-tuning:** how much does one short GPU fine-tune add on top?

Not in v2.0: text inputs, cross-attention fusion, MAE pretraining, report
generation, anything that needs more than one GPU session. These are in the
backlog (§10). M4, added after v2.0, brings in text written before the X-ray
(§6).

## 2. Starting point: what v1 tells us

- v1 reported macro AUROC **0.701** over 12 classes (December 2025 model,
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
| Inputs | The image only in M0-M3; M4 adds text written before the X-ray. Nothing recorded after the X-ray, ever. |
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
training subset halves it), or minutes on a GPU. Measured on an Apple-silicon
Mac: about 6 min per model at 224 px and 53 min at 518 px; probes under a minute.

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

### M4: Pre-imaging text + image (CPU; written 2026-10-20, before any M4 run)

**Question:** does clinical text written before the X-ray add anything to the image?

v1 combined text with the image, but its text either was the report the labels
come from or included data recorded after the X-ray (labs, discharge codes).
M4 uses only text that existed before the X-ray, and measures what it adds to
the image alone, paired, on `test_clean`. This replaces the earlier vitals-only
M4; vitals become a secondary arm.

**Inputs** (`python -m v2.context`; one row per manifest study, kept local):

- **Indication:** the INDICATION, HISTORY, CLINICAL INFORMATION and REASON FOR
  EXAM sections of the study's own report, which hold the ordering clinician's
  question. Nothing else from that report is used. FINDINGS, IMPRESSION, WET
  READ, ADDENDUM and every other section are dropped, because the labels come
  from them. A guard also cuts the text at any later section header that slips
  through, and the audit counts those cuts.
- **Chief complaint:** MIMIC-IV-ED triage `chiefcomplaint` of the most recent
  ED stay that began at or before the X-ray and no more than 24 h before it.
- **Prior report:** FINDINGS and IMPRESSION of the patient's most recent
  earlier MIMIC-CXR study taken at least 24 h before. The data has no report
  sign times, and 24 h makes it near certain the prior report existed.
- **Triage vitals, acuity, sex and age** from the same ED stay and MIMIC-IV
  patients (secondary arms only). Age is optional.
- A missing source stays missing, with an indicator. Every study stays in, so
  all arms are scored on the same studies.

**Features and model:** the frozen RAD-DINO 518 px embeddings from M1. For each
text source, TF-IDF over word 1-2-grams (letters only, min_df 5) plus a
128-component truncated SVD, both fit on `train` only, and a presence indicator.
The classifier is the M1 probe: per-class L2 logistic regression with C chosen
on val.

| Arm | Features | Role |
|---|---|---|
| `image` | image | Reference; must reproduce `rad-dino_518_probe` |
| `image_text` | image + indication + chief complaint + prior report | **Primary (G4)** |
| `image_indication`, `image_complaint`, `image_prior` | image + one source | Which source carries the gain |
| `text`, `indication`, `complaint`, `prior` | text only | Controls |
| `image_vitals`, `image_text_vitals` | adds vitals, acuity, sex, age | Secondary |

**Gate G4 (fixed now):** `image_text` minus `image` on `test_clean`, with a
paired 95% CI above 0. If it passes, pre-imaging text helps, and the next step
is a joint image-text fine-tune on the GPU. Otherwise text as an input is
dropped, and the next text experiment uses reports as a training signal only
(the image-report alignment loss in §10).

**Leakage red flags, checked before G4 is read:** a text-only arm that beats
`image` on `test_clean`, or any measurable class whose text-only AUROC on
`test_clean` exceeds 0.90. Either one means stop and inspect the extraction locally before
interpreting anything.

**Exploratory, after G4 is read:** a frozen local biomedical text encoder
(PubMed-trained, no MIMIC exposure) in place of TF-IDF.

**Data:** MIMIC-CXR free-text reports (`mimic-cxr-reports.zip`, from the
MIMIC-CXR project rather than MIMIC-CXR-JPG), MIMIC-IV-ED 2.2 (`ed/edstays`,
`ed/triage`) and optionally MIMIC-IV `hosp/patients`. Everything runs locally.
No text goes to an external API, and only aggregate output reaches an AI
assistant.

**Done when:** `context_audit.md` and every arm's predictions exist, the
red flags are checked and G4 is logged in §9.

### M4b: Fusion follow-ups (CPU; written 2026-10-21, before any M4b run)

**Questions:**
1. Can text steer *where* the model looks? Late fusion (M4) adds text to each
   class's log-odds after the image has been pooled into one vector, so the
   text can shift the odds but can't change how the image is read. Does
   attention that lets the text choose image regions add to late fusion?
   (primary: gate G4b)
2. Is the prior report more useful as its CheXpert labels than as TF-IDF?
   (exploratory)
3. How much does text add to the fine-tuned image when the combination is fit
   on studies the fine-tune never trained on? (exploratory)

M4's follow-ups (§9, 2026-10-21) motivate 2 and 3: text added +0.009 on the
fine-tuned image against +0.016 on the frozen one, but that probe was fit on
training studies the fine-tune had already fit, so the image looked more
reliable than it is and the text got too little weight.

**Part 1: text-steered attention (gate G4b).**

- **Image tokens:** frozen RAD-DINO at 518 px, last hidden layer, as in M1. The
  37 x 37 patch tokens are average-pooled to a 12 x 12 grid; with the CLS token
  that is 145 tokens of 768 numbers per study, stored as float16
  (`python -m v2.embed --model rad-dino --size 518 --tokens 12`, about 7 GB). The
  CLS token must match the first half of `rad-dino_518.npz` to float16
  precision, which checks that the two passes saw the same images in the same
  order.
- **Text:** exactly M4's features (TF-IDF + 128-component SVD per source,
  fit on train, plus presence indicators), standardized on train.
- **Heads** (same code for every arm): tokens standardized per dimension (train
  statistics); keys are a linear map of each token plus a learned position
  embedding; values are another linear map; 12 learned class queries attend
  over the 145 tokens with 4 heads (width 256); each class's pooled vector goes
  through dropout 0.1 and its own linear readout to a logit.

| Arm | What it adds | Role |
|---|---|---|
| `h0` | image tokens only | Is the attention head a good image probe? |
| `h1` | `h0` + a linear text term in each class's logit | Late fusion on the same head |
| `h2` | `h1` + text-conditioned queries: each class query is shifted by an MLP of the text (387 -> 64 -> 12 x 256), its last layer initialized at 0 | Text chooses where each class looks |

  `h2` contains `h1`, and with the same seed it starts as exactly `h1`, so
  `h2` minus `h1` measures steering alone: same head, same inputs.
- **Recipe (fixed now):** masked BCE (v2 label policy); AdamW, lr 1e-3, weight
  decay 0.05 (none on biases, positions or queries); 1 warmup epoch, then
  cosine over 30 epochs; batch 256; gradient clipping 1.0; at most 30 epochs,
  stopping after 5 without a better val macro AUROC (classes with at least 10
  positives and negatives in val, as in M3); the best epoch is kept. Seeds 0-4
  for every arm; an arm's score is the mean of its 5 seeds' logits.
- **Validity check, read first:** `h0`'s val macro AUROC must be at least
  `rad-dino_518_probe`'s minus 0.02. Otherwise the head is broken: stop and
  inspect before reading G4b.
- **Gate G4b (fixed now):** `h2` minus `h1` on `test_clean`, with a paired 95%
  CI above 0. If it passes, text-steered attention adds to late fusion, and a
  joint image-text fine-tune should use text-conditioned attention over patch
  tokens. If it fails, nothing detectable is lost by fusing late, which argues
  against spending GPU time on a joint image-text fine-tune; that decision is
  taken separately (§10).
- **Reported, not gates (uncorrected):** `h0` minus `rad-dino_518_probe`
  (head capacity); `h1` minus `h0` (text on the attention head, next to M4's
  +0.016); `h2` minus `rad-dino_518_ctx_image_text` (best attention arm against
  today's late fusion; mixes the head change with the fusion change); `h2`
  minus `h1` on `test_v1` (v1 metric); per-class AUROC of `h1` and `h2` on
  `test_clean`.
- **Diagnostic:** how far text moves `h2`'s attention. Per class, the total
  variation distance between the head-averaged attention map with the study's
  text and with the text set to its train mean, averaged over `test_clean`
  studies and the 5 seeds. 0 means `h2` learned not to steer.

**Part 2: prior-study labels (exploratory).**

- **Input:** the 14 CheXpert labels (`mimic-cxr-2.0.0-chexpert.csv.gz`) of the
  same prior study M4 uses (the latest study at least 24 h earlier). They are
  the CheXpert labeler's reading of that prior report, which is already an
  allowed input, so they add no new information; they encode it differently.
  Each label is one-hot over positive, negative, uncertain and not mentioned,
  plus an indicator that a labeled prior exists: 57 features.
- **Arms** (the M1 probe, as in M4): `prior_labels` (control, alone),
  `image_prior_labels`, `image_text_prior_labels`.
- **Comparisons** (paired; `test_clean`, and `test_v1` with the v1 metric):
  `image_prior_labels` minus `image_prior` (labels against TF-IDF of the same
  report) and `image_text_prior_labels` minus `image_text` (beyond all three
  text sources).
- **Red flags:** M4's rule applies to `prior_labels`: at or above `image`, or a
  measurable class above 0.90, means stop and inspect.
- **Caveat, fixed now:** radiologists read the prior report and often carry its
  wording forward, and the same labeler reads both reports. Part of any gain
  is predicting the report rather than the patient. Read the per-class gains
  with that in mind.
- If `image_text_prior_labels` beats `image_text` on val, the prior-label
  encoding joins the text recipe for later work; test numbers don't decide it.

**Part 3: a combiner fit on val (exploratory).**

- **Method:** per class, a logistic regression (C = 1 on standardized inputs)
  on two scores: the image run's logit and the logit of `rad-dino_518_ctx_text`
  (text only, fit on train). It is fit on val, which no model here trained on.
  Val's own scores come from 5-fold cross-fitting within val, grouped by
  patient; test scores come from the fit on all of val. A class with only one
  label value in val, or in a fold, keeps the image logit unchanged.
- **Runs:** `rad-dino_518_ft_s1_stack_text` (fine-tuned image + text) and
  `rad-dino_518_probe_stack_text` (frozen image + text, the check).
- **Comparisons** (paired, `test_clean`): `ft_s1_stack_text` minus
  `rad-dino_518_ft_s1` is the text gain on the fine-tuned image without the
  bias; also on `test_v1` (v1 metric). The check: `probe_stack_text` minus
  `rad-dino_518_ctx_image_text`. On frozen features there is no bias to remove,
  so the combiner should land near the concatenation; if it is more than 0.005
  below, the two-score combiner loses information and its fine-tuned estimate
  is a lower bound.
- Known limitation: val also chose the probes' C and the fine-tune's epoch, so
  both input scores are slightly optimistic on val.

**Amendment (2026-10-21, before any M4b run): label policy for text comparisons.**
The M4 follow-ups found text gains on `test_v1` near 0 under the v1 policy but
+0.019 to +0.025 under v2. The v1 policy keeps only findings the report
mentions, and text the radiologist reads is itself a cause of a mention: asked
about X, the report addresses X whether or not it is there, so explicit
negatives are mostly studies where someone asked. Keeping only mentioned
findings selects on a common effect of the question and the disease, which
biases a text gain toward zero or below even when the text is informative
(`tests/v2/test_common.py` simulates this). Under v2, where an unmentioned
finding counts as negative, the effect cancels. So every M4b text comparison on
`test_v1` is reported under v2 first and v1 second, and text gains are judged
under v2. Neither policy can rule out reporting bias: being asked about a
finding may make a radiologist more likely to write it up. Only labels made
without seeing the text could measure that. The parts above are otherwise
unchanged, and so is G4b.

**Compute:** the token pass takes about as long as M1's 518 px embedding (52
min on the Mac) and needs about 7 GB of disk; the heads train on the cached
tokens (15 runs; wall time to be measured); parts 2 and 3 take minutes. Parts
2 and 3 need `context.parquet` rebuilt (`python -m v2.context`), which adds the
prior labels; nothing else in it changes.

**Outputs:** `reports/m4b_attention_rad-dino_518.md` (validity check, G4b,
contrasts, diagnostic), `reports/m4b_rad-dino_518.md` (prior labels; the fuse
report for those arms) and `reports/m4b_stack.md`. Everything runs locally;
only aggregate output reaches an AI assistant.

**Done when:** the three reports exist, the validity check and G4b are read in
that order, and every M4b result is logged in §9.

### M5: Line up with published results (written 2026-10-21, before any M5 run)

**Question:** how do our models compare with published MIMIC-CXR results when
scored the same way?

v2.0's cohort (abnormal studies linked to ED stays) and its test sets are not a
published benchmark, so v2.0's numbers can't be placed against the literature.
The best-documented published protocol is MMBT's (Jacenków, O'Neil and
Tsaftaris, ISBI 2022): MIMIC-CXR-JPG studies with a frontal image, the official
split (205,923 / 1,695 / 2,920 studies that have an indication or history
field), 14 CheXpert labels scored positive against the rest, and macro AUROC on
the official test split. M5 runs our models under that protocol.

**Data (backlog item 1, §10):** every MIMIC-CXR study with a CheXpert row and a
frontal image (one per study, PA over AP), in its own output directory, so
v2.0's manifest and runs stay untouched. Splits follow the official split by
patient: train is official train; val is official validate (500 patients),
used for every model choice; test is official test (293 patients, 89% of its
studies have a finding). RAD-DINO left the official validate and test images
out of pretraining, so the test set stays clean for it. Images are stored at
518 px as JPEG (quality 95), which keeps about 220k images near 15 GB.

**Labels and scoring:** our 12 pathologies; No Finding and Support Devices are
not v2 classes, so published 14-label macros are compared through their
per-class tables. Models train as in v2 (uncertain labels ignored). On the
official test split they are scored under positive vs rest (the `u0` policy:
uncertain and unmentioned count as negative), which is MMBT's policy, and
under v2's policy. Reported: macro AUROC over the 12 classes with a 95%
patient-bootstrap CI, and per-class AUROC.

**Runs:**
- **M5a (Mac):** frozen RAD-DINO 518 px embeddings and the M1 probe (C chosen
  on val). Then M4's late fusion on the same embeddings: `image`,
  `image_indication` (MMBT's inputs), `image_text`, and the text-only
  `indication` and `text` controls, with M4's red flags.
- **M5b (GPU, after M5a):** fine-tune RAD-DINO 518 px on all train studies.
  The M3 recipe with 3 epochs (1 of warm-up, then cosine), since both v2.0
  fine-tunes peaked at epoch 1 of 10. Best epoch on val, seed 0 (seed 1 if the
  budget allows). Then a probe refit on its features and M4's late fusion
  (exploratory, as in the M4 follow-ups).

**Amendment (2026-10-21, before any M5 run), after M4b:** M4b's image-only
attention head (`h0`) beat the linear probe on val (0.823 against 0.803) and
matched the fine-tune on `test_clean`. So M5a adds M4b's heads on the
benchmark's frozen tokens, with M4b's recipe and 5 seeds:
- `h0` (image), like for like with ResNet-50;
- `h1` (image + the three text sources at the logit), reported but not like
  for like, since MMBT uses the indication only.

`h2` is left out because G4b failed. One embedding pass writes both the pooled
embeddings and the 12 x 12 token cache (about 49 GB more disk). `prepare_data`
now stops when more than 0.5% of images fail to read, so a drive that drops
mid-run can't silently shrink the benchmark. Everything else in M5 is
unchanged.

**Amendment (2026-10-22, after M5a, before any M5b run):** like for like, M5a's
image-only runs beat ResNet-50, image + indication ties MMBT, and the
indication alone trails BERT (§9). The attention head beat the probe by
+0.016 again, and against v2.0's fine-tune it is +0.003 [-0.007, 0.013]. Three
changes:

1. **Text on the stronger image models goes through M4b's val combiner**
   (`v2.stack`), since a probe fit on training studies understated text on
   fine-tuned features by about half (M4b). Each image run is stacked with two
   text-only runs: `rad-dino_518_ctx_indication` (image + indication, like for
   like with MMBT and Attentive) and `rad-dino_518_ctx_text_prior_labels` (all
   pre-imaging text, item 3; reported, not like for like). The image runs are
   `rad-dino_518_attn_h0` (frozen, on the Mac before the GPU session),
   `rad-dino_518_ft_s0` and `rad-dino_518_ft_attn_s0` (item 2). The check, on
   frozen features: `rad-dino_518_probe` stacked with each text run against the
   matching concatenation (`ctx_image_indication`, `ctx_image_text_prior_labels`);
   more than 0.005 below means the combiner loses information and its
   estimates are lower bounds. The lineup compares a stack whose text run is
   `indication` as image + indication.
2. **A second fine-tune, with the attention head** (`rad-dino_518_ft_attn_s0`):
   M5b's recipe and seed, with the mean-pool linear head replaced by `h0`'s
   head (CLS plus the last layer's patch tokens averaged to 12 x 12; width 256,
   4 heads, dropout 0.1, learned positions; head lr 1e-3 like M3's head). The
   head starts at random, with its bias at the logit of each label's training
   prevalence; its token normalization is the frozen backbone's per-dimension
   mean and SD over 512 training studies, then fixed. It does not start from
   M5a's trained `h0` (LP-FT, backlog 8): that would carry weights from the
   Mac to the GPU, and the warm-up epoch (about 6,700 optimizer steps) lets the
   head settle while the encoder's learning rate is still small. The question:
   does attention pooling add to fine-tuning, when each alone gained about the
   same over the probe (+0.012 and +0.015 on `test_clean`)? Reported, paired, on
   the official test under both policies: `ft_attn_s0` minus `ft_s0`, and
   `ft_attn_s0` minus `attn_h0`. Not a gate. It is lined up as image with
   ResNet-50. The image model carried into later work is whichever of
   `attn_h0`, `ft_s0` and `ft_attn_s0` scores highest on val.
3. **Prior-study labels join the all-text recipe**, because M4b's rule fired
   (+0.002 [0.000, 0.003] on val). A text-only arm, `text_prior_labels`, is
   added; from M5b on, the all-text arms are `text_prior_labels` and
   `image_text_prior_labels`. The indication arms don't change, and M5a's runs
   are not redone.

The rest of M5b stands: `ft_s0` saves its embeddings for the refit probe and
M4's late fusion (exploratory), with `image_text_prior_labels` as the all-text
arm. Seed 1 of both fine-tunes only if the budget allows. Compute: about 1.7
A100-hours per fine-tune, so about 3.5 h (roughly $5-7) for seed 0 of both,
plus uploading about 16 GB of images.

**Published references** (AUROC, transcribed from the papers):

| Model | Inputs | Mean over our 12 classes | Paper's macro | Protocol |
|---|---|---|---|---|
| ResNet-50 (MMBT paper) | image, 224 px | 0.725 | 0.738 (14 labels) | MMBT's: official split, positive vs rest |
| BERT (MMBT paper) | indication | 0.704 | 0.711 (14 labels; its per-class table averages 0.719) | same |
| Attentive (MMBT paper) | image + indication | 0.751 | 0.766 (14 labels) | same |
| MMBT | image + indication | 0.794 | 0.806 (14 labels) | same |
| CheXclusion DenseNet-121 | image | 0.823 | 0.834 (14 labels) | context only: random patient split, frontal and lateral images |
| DenseNet-121 (Wang, Zhang and Oberst, CHIL 2026) | image | - | 0.803 (13 labels) | context only: random patient split |
| MBRANet | image | - | 0.805 | context only: split not verified |
| CaMCheX | two views + indication + vitals, 1024 px | - | 0.934 (14 labels) | context only: no leakage control described |

The 12-class means are the averages of each paper's per-class table (MMBT
paper Table 3; CheXclusion Table 2).

**How a comparison is read (fixed now):** a published 12-class mean is
*below* ours if it lies under the lower bound of our 95% CI, *above* if it
lies over the upper bound, and *level* otherwise. Published numbers carry their
own uncertainty (MMBT reports SDs of 0.1 to 2.5 points over 3 seeds), so
*level* means a tie. The like-for-like pairs are `image` with ResNet-50
(image only), `image_indication` with Attentive and MMBT (image +
indication), and `indication` with BERT (indication only). MMBT's test set
keeps only studies with an indication or history field, so each comparison is
also reported on the official-test studies whose indication is not empty. If a
red flag fires on a text-only arm, the lineup isn't read until the extraction
is inspected.

**Caveats, fixed now:** 12 of 14 labels; labels from the MIMIC-CXR-JPG
CheXpert file, where MMBT ran the CheXpert labeler on full reports itself; one
frontal image per study; RAD-DINO pretrained on the official train images
without labels, which is standard for this benchmark.

**Compute:** preparing ~220k images takes 1 to 3 h from a fast disk (9 h from
a USB hard drive in M5a, where reads were the limit) and ~16 GB; RAD-DINO
embeddings and tokens ~6 h on an Apple-silicon Mac (one pass, ~49 GB of
tokens); probes and fusion arms under 30 min; the attention heads about 2.5 h. M5b: about 1.7 h of an A100 per seed (~$2-3), plus
uploading the images.

**Done when:** `reports/lineup.md` exists for M5a (then M5b), and its rows are
logged in §9.

### M6: A language model for the pre-imaging text (written 2026-10-22, before any M6 run)

**Question:** on the benchmark (M5), does a BERT-family encoder of the text
written before the X-ray beat M4's TF-IDF features, alone and with the image?
And does pretraining on medical text ("medical context") beat a general BERT?
M5 left one like-for-like loss: our TF-IDF indication model scores 0.675
against MMBT's fine-tuned BERT at 0.704.

**Encoders** (Hugging Face, revisions pinned):

| Name | Repository @ revision | Pretrained on |
|---|---|---|
| `bert` | google-bert/bert-base-uncased @ 86b5e09 | Wikipedia and books; MMBT's backbone |
| `biomedbert` | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext @ e1354b7 | PubMed abstracts and full-text articles |
| `clinicalbert` | emilyalsentzer/Bio_ClinicalBERT @ d5892b3 | BioBERT, then MIMIC-III clinical notes |

Caveat, fixed now: MIMIC-III's notes come from the same hospital and overlap
MIMIC-CXR's years (2011-2012), so `clinicalbert` may have read some of these
patients' reports as unlabeled text in pretraining, as RAD-DINO saw the
training images. Its results carry that flag, and a large lead for it alone is
read with suspicion.

**Two ways to use each encoder:**
- **Frozen:** for each text source, the mean of the last layer's token
  embeddings (indication and chief complaint cut at 128 tokens, the prior
  report at 512), standardized on train, plus a presence indicator. It
  replaces that source's TF-IDF block in M4's probe (C chosen on val). Arms:
  `indication`, `text_prior_labels`, `image_indication`,
  `image_text_prior_labels`, and `image_indication` with both blocks (image +
  TF-IDF + BERT).
- **Fine-tuned:** the encoder and a linear head on its [CLS] token, trained on
  the indication alone to predict the 12 findings. Masked BCE (v2 policy),
  AdamW (lr 2e-5, weight decay 0.01), batch 32, 64 tokens, 3 epochs (10%
  warm-up, then cosine), best epoch on val, seed 0. This is MMBT's BERT
  baseline, retrained on our labels.

**Into the image:** every text-only run, frozen or fine-tuned, is added to the
best image model, `rad-dino_518_ft_attn_s0`, through M5b's val combiner, and to
the frozen probe for the combiner's check. Image + indication stacks are lined
up with MMBT; fine-tuned indication runs with MMBT's BERT.

**Gate G6 (medical context):** on the indication alone, the fine-tuned medical
encoder with the higher val score (`biomedbert` or `clinicalbert`) minus the
fine-tuned `bert`, macro AUROC on the official test under the v2 policy,
paired. PASS if the 95% interval is above 0.

**Reported comparisons** (paired, official test, both policies; not gates):
1. Fine-tuned `bert` minus TF-IDF, on the indication: is our text gap the
   encoder?
2. The fused image + indication run with the best val score minus
   `rad-dino_518_ft_attn_s0` + TF-IDF indication (0.815): does a better text
   model move the like-for-like result?
3. Frozen image + TF-IDF + BERT indication minus frozen image + TF-IDF
   indication: does BERT add on top of TF-IDF?
4. Per class, our fine-tuned `bert` against MMBT's BERT, Cardiomegaly first:
   if MMBT's labels draw on the indication, its BERT should lead most on the
   findings that indications name.

**Adoption rule:** the text representation for later work is the one whose
image + indication stack on `ft_attn_s0` scores highest on val; test numbers
don't decide it.

**Red flags:** M4's rule applies to every language-model text-only run: at or
above the image, or a class above 0.90, means stop and inspect.

**Compute (Mac, no GPU):** frozen embeddings about 20-30 min per encoder,
probes about 20 min, fine-tuning about 45-60 min; combiners and the lineup take
minutes. About 4-5 h in all.

**Done when:** `reports/m6_text.md` (G6 and the comparisons) and the updated
`reports/lineup.md` exist, and their rows are logged in §9.

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
  finetune.py       # M3: plain PyTorch loop, bf16, checkpoint every epoch, --resume; M5b: --head attention
  context.py        # M4 inputs: indication, ED chief complaint, prior report, vitals (+ leak guards, audit)
  fuse.py           # M4 arms: text features fit on train, late fusion with the probe, red flags, gate G4
  attend.py         # M4b part 1: attention heads over cached patch tokens, gate G4b
  stack.py          # M4b part 3: per-class combiner of two runs' scores, fit on val
  lineup.py         # M5: our runs next to published MIMIC-CXR results, scored the same way
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

1. Rent 1× A100 or H100 (40-80 GB) with CUDA 12 (about $1.3-2.5/h). x86-64 is
   simplest; on an ARM machine (GH200), check after setup that PyTorch still
   sees the GPU.
2. `bash v2/scripts/setup.sh`: venv (keeps the provider's CUDA torch), install,
   `pytest tests/v2`, wiring check. `hf auth login` only if M2 runs there.
3. Upload `output/v2/manifest.parquet`, `measurable_classes.json`, `audit.json`
   and `images/` (5-8 GB, 10-20 min).
4. `bash v2/scripts/run.sh smoke m3 eval` when M1 and M2 already ran elsewhere
   (in this project they ran on the Mac at 518 px), or `bash v2/scripts/run.sh gpu`
   to run them there too:
   1. a 5-minute smoke test on real data (aborts on any failure);
   2. `gpu` only: full-resolution embeddings and probes for M1 and M2;
   3. two RAD-DINO fine-tunes (2 seeds);
   4. DenseNet-121;
   5. evaluation.

   Each stage has a time limit, and stages after 4 hours are skipped.
5. Download the fine-tunes' predictions, best checkpoints and logs, and
   evaluate where the probe runs are (G2 compares the two). Delete the data,
   then terminate the instance.

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
| 2026-10-19 | M1 | ec3e020 | rad-dino_224_probe_wide | `test_clean` | 0.761 [0.741, 0.778] | 9 | Wider C grid on the same embeddings: +0.022 over the old grid. Most classes chose C=0.001; Consolidation and Lung Opacity sit at the new lower edge (0.0001) |
| 2026-10-19 | M1 | ec3e020 | rad-dino_224_probe_wide | `test_v1`, v1 metric | 0.800 [0.784, 0.814] | 8 | |
| 2026-10-19 | M1 | ec3e020 | dinov2-base_224_probe_wide | `test_clean` | 0.715 [0.696, 0.732] | 9 | The wider grid barely moves DINOv2 |
| 2026-10-19 | M1 | ec3e020 | dinov2-base_224_probe_wide | `test_v1`, v1 metric | 0.730 [0.713, 0.748] | 8 | |
| 2026-10-19 | M1 | ec3e020 | medsiglip_448_probe_wide | `test_clean` | 0.782 [0.765, 0.798] | 9 | Flagged (§4). `test_v1` not reportable. Pneumothorax at the lower edge |
| 2026-10-19 | M1 | ec3e020 | rad-dino_518_probe | `test_clean` | 0.775 [0.756, 0.791] | 9 | **Best eligible probe by val (0.803).** Paired on `test_clean`: vs `rad-dino_224_probe_wide` +0.014 [0.007, 0.022]; vs `dinov2-base_518_probe` +0.060 [0.041, 0.078]; `medsiglip_448_probe_wide` minus this +0.007 [-0.006, 0.021], a tie. Consolidation at the lower edge. 52 min to embed on the Mac |
| 2026-10-19 | M1 | ec3e020 | rad-dino_518_probe | `test_v1`, v1 metric | 0.803 [0.788, 0.818] | 8 | v1's 0.792 lies inside this CI: a frozen encoder on the image alone matches v1 |
| 2026-10-19 | M1 | ec3e020 | dinov2-base_518_probe | `test_clean` | 0.715 [0.695, 0.732] | 9 | No gain from 518 px on `test_clean` (+0.013 on val, +0.011 on `test_v1`) |
| 2026-10-19 | M1 | ec3e020 | dinov2-base_518_probe | `test_v1`, v1 metric | 0.741 [0.726, 0.759] | 8 | |
| 2026-10-19 | G1 | ec3e020 | gate (518 px, wider grid) | `test_v1`, v1 metric | 0.803 [0.788, 0.818] | 8 | **PASS**, on the run the §6 amendment named: `rad-dino_518_probe`, best by val among eligible probes, with its lower CI bound above 0.77. Exposure check on the same classes: RAD-DINO's `test_v1` minus `test_clean` gap is +0.010, below DINOv2's +0.016, so no sign that pretraining exposure inflates `test_v1`. Next: M3; G2 compares against this probe |
| 2026-10-19 | M3 | df4fb0b | rad-dino_518_ft_s0 | `test_clean` | 0.786 [0.770, 0.801] | 9 | Minus `rad-dino_518_probe`: +0.012 [0.003, 0.022]. Best epoch 1 of 10 (val 0.820). Val loss was lowest at epoch 2, then the model overfit while train loss kept falling; the last epoch scored below the frozen probe on val. 39.5 min on an A100-SXM4-40GB, 110 img/s |
| 2026-10-19 | M3 | df4fb0b | rad-dino_518_ft_s0 | `test_v1`, v1 metric | 0.824 [0.808, 0.839] | 8 | |
| 2026-10-19 | M3 | df4fb0b | rad-dino_518_ft_s1 | `test_clean` | 0.787 [0.769, 0.803] | 9 | **v2 baseline** (best fine-tune by val, 0.821). Minus the probe: +0.012 [0.003, 0.022]. Minus seed 0: -0.000 [-0.007, 0.006]. Same epoch-1 peak as seed 0 |
| 2026-10-19 | M3 | df4fb0b | rad-dino_518_ft_s1 | `test_v1`, v1 metric | 0.823 [0.807, 0.837] | 8 | 0.031 above v1's 0.792, which lies below this CI |
| 2026-10-19 | M3 | df4fb0b | densenet121_512_ft_s0 | `test_clean` | 0.768 [0.748, 0.785] | 9 | `rad-dino_518_ft_s0` minus this: +0.019 [0.005, 0.031]. Best epoch 5 of 10. 30.4 min, 137 img/s |
| 2026-10-19 | M3 | df4fb0b | densenet121_512_ft_s0 | `test_v1`, v1 metric | 0.796 [0.780, 0.813] | 8 | |
| 2026-10-19 | G2 | df4fb0b | gate | `test_clean` | difference +0.012 [0.003, 0.022] | 9 | **PASS**: `rad-dino_518_ft_s1` minus `rad-dino_518_probe`, so the fine-tune is the v2 baseline; both seeds agree. `rad-dino_518_ft_s0` minus `medsiglip_448_probe_wide`: +0.005 [-0.008, 0.017], a tie. Exposure check: the fine-tunes' gaps (+0.014, +0.015) match DINOv2's (+0.016). GPU session: Lambda A100-SXM4-40GB, 2.15 instance hours, about $3-4. One environment fix: numpy<2 for Lambda's preinstalled torch (now automatic in `setup.sh`) |
| 2026-10-20 | correction | 90be62e | v1 (reference) | | | | The v1 reference row's date is wrong: v1 was trained and reported in December 2025 (`docs/RESULTS.md`, added 2025-12-29 in 90be62e), not 2024-12. 8166273 is the `main` commit v2 branched from. The v1 docs' dates are corrected to 2025 |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image | `test_clean` | 0.775 [0.756, 0.791] | 9 | Reference arm. Reproduces `rad-dino_518_probe` exactly (largest score difference 0) |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image | `test_v1`, v1 metric | 0.803 [0.788, 0.818] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_text | `test_clean` | 0.791 [0.774, 0.808] | 9 | **Primary (G4).** Minus `image`: +0.016 [0.009, 0.024] |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_text | `test_v1`, v1 metric | 0.812 [0.797, 0.827] | 8 | +0.009 over `image` here (not paired) |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_indication | `test_clean` | 0.783 [0.766, 0.800] | 9 | Minus `image`: +0.009 [0.004, 0.014], the largest single source |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_indication | `test_v1`, v1 metric | 0.807 [0.791, 0.822] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_complaint | `test_clean` | 0.777 [0.760, 0.794] | 9 | Minus `image`: +0.003 [-0.001, 0.007], a tie |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_complaint | `test_v1`, v1 metric | 0.810 [0.794, 0.825] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_prior | `test_clean` | 0.781 [0.763, 0.798] | 9 | Minus `image`: +0.007 [0.002, 0.012]. Priors exist for 74.9% of `test_clean` studies vs 56.7% of train |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_prior | `test_v1`, v1 metric | 0.803 [0.788, 0.818] | 8 | No gain here (57.1% have a prior; not paired) |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_vitals | `test_clean` | 0.774 [0.756, 0.791] | 9 | Minus `image`: -0.001 [-0.003, 0.001]: triage vitals, acuity, sex and age add nothing |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_vitals | `test_v1`, v1 metric | 0.806 [0.791, 0.820] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_text_vitals | `test_clean` | 0.790 [0.772, 0.807] | 9 | Minus `image`: +0.016 [0.009, 0.023] |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_image_text_vitals | `test_v1`, v1 metric | 0.814 [0.799, 0.828] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_text | `test_clean` | 0.692 [0.671, 0.713] | 9 | Control, text only: well below `image`; no measurable class above 0.90 |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_text | `test_v1`, v1 metric | 0.674 [0.654, 0.694] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_indication | `test_clean` | 0.622 [0.600, 0.643] | 9 | Control |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_indication | `test_v1`, v1 metric | 0.620 [0.600, 0.638] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_complaint | `test_clean` | 0.568 [0.546, 0.589] | 9 | Control |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_complaint | `test_v1`, v1 metric | 0.600 [0.580, 0.621] | 8 |  |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_prior | `test_clean` | 0.660 [0.635, 0.685] | 9 | Control. 0.595 on `test_v1`: priors are commoner among `test_clean` patients |
| 2026-10-20 | M4 | f199e81 | rad-dino_518_ctx_prior | `test_v1`, v1 metric | 0.595 [0.576, 0.613] | 8 |  |
| 2026-10-20 | G4 | f199e81 | gate | `test_clean` | difference +0.016 [0.009, 0.024] | 9 | **PASS**: `rad-dino_518_ctx_image_text` minus `rad-dino_518_ctx_image`; no leakage red flags. Inputs: indication for 99.1% of train studies, chief complaint 94.2%, prior report 56.7%, vitals 88.8%; 0 reports missing. The gain comes from the indication and the prior report; chief complaint and vitals add nothing measurable. Run on the Mac in 10 min. Next, per §6: the joint image-text fine-tune |
| 2026-10-21 | M4, exploratory | a8060e1 | rad-dino_518_ft_s1_ctx_image | `test_clean` | 0.793 [0.775, 0.809] | 9 | Reference arm on fine-tuned features: the M1 probe refit on `rad-dino_518_ft_s1`'s embeddings. 0.006 above that fine-tune's own head (0.787); not yet paired |
| 2026-10-21 | M4, exploratory | a8060e1 | rad-dino_518_ft_s1_ctx_image_text | `test_clean` | 0.802 [0.784, 0.819] | 9 | Minus `ft_s1_ctx_image`: +0.009 [0.003, 0.014], about half the frozen-feature gain. Likely an underestimate: the probe is fit on training studies the fine-tune has already fit, where the image looks more reliable than it is on test, so the text gets too little weight. Minus `rad-dino_518_ft_s1`: +0.015 [0.008, 0.023], which mixes the refit with the text. Highest `test_clean` score so far, but not chosen by val and not a gate |
| 2026-10-21 | M4, exploratory | a8060e1 | rad-dino_518_ft_s1_ctx_image_indication | `test_clean` | 0.797 [0.780, 0.814] | 9 | Minus `ft_s1_ctx_image`: +0.004 [-0.000, 0.009] (+0.009 on frozen features) |
| 2026-10-21 | M4, exploratory | a8060e1 | rad-dino_518_ft_s1_ctx_image_prior | `test_clean` | 0.797 [0.779, 0.814] | 9 | Minus `ft_s1_ctx_image`: +0.004 [0.001, 0.007] (+0.007 on frozen features) |
| 2026-10-21 | M4, exploratory | a8060e1 | compare | `test_clean` | difference +0.005 [-0.006, 0.015] | 9 | `rad-dino_518_ctx_image_text` minus `rad-dino_518_ft_s1`: a tie. The frozen image plus text matches the fine-tuned image |
| 2026-10-21 | M4, exploratory | a8060e1 | compare | `test_v1`, v1 metric | difference +0.009 [0.001, 0.017] | 8 | `rad-dino_518_ctx_image_text` minus `rad-dino_518_ctx_image`, now paired: the frozen text gain holds on the second test set at about half its `test_clean` size (+0.016) |
| 2026-10-21 | M4, exploratory | a8060e1 | compare | `test_v1`, v1 metric | difference -0.000 [-0.004, 0.004] | 8 | `rad-dino_518_ctx_image_prior` minus `rad-dino_518_ctx_image`: no prior-report gain here (+0.007 on `test_clean`). Prior reports exist for 57.1% of `test_v1` studies vs 74.9% of `test_clean`, whose patients have more cohort studies each (870 / 341 = 2.6 vs 4,860 / 2,947 = 1.6). If the gain scaled with coverage alone it would be about +0.005 here, just above this CI; the classes and metric also differ. The follow-ups took 3.7 min on the Mac |
| 2026-10-21 | M4, exploratory | a8060e1 | compare | `test_clean` | difference +0.006 [0.001, 0.012] | 9 | `rad-dino_518_ft_s1_ctx_image` minus `rad-dino_518_ft_s1`: refitting the M1 probe on the fine-tuned features beats the fine-tune's own head, on val too (0.825 vs 0.821). The earlier +0.015 over the baseline splits into about +0.006 from the refit and +0.009 from text |
| 2026-10-21 | M4, exploratory | a8060e1 | compare | `test_v1`, v1 metric | difference +0.001 [-0.004, 0.006] | 8 | `rad-dino_518_ft_s1_ctx_image_text` minus `rad-dino_518_ft_s1_ctx_image`. Under the v2 policy the same gain is +0.019 on `test_v1` (unpaired: 0.836 vs 0.817) and on val (0.844 vs 0.825) |
| 2026-10-21 | M4, exploratory | a8060e1 | rad-dino_518_ft_s1_ctx_image_text | `val` | 0.844 [0.834, 0.853] | 12 | The best val score so far (`ft_s1_ctx_image` 0.825, `ft_s1` 0.821), so fine-tuned features + refit probe + text is the best system by val: 0.802 [0.784, 0.819] on `test_clean`, 0.827 [0.811, 0.842] on `test_v1` (v1 metric). Exploratory; no gate defines it. 6 min on the Mac |
| 2026-10-21 | correction | a8060e1 | M4 follow-ups | | | | Two notes in the 2026-10-21 rows above compared `test_v1` under the v1 metric with `test_clean` under v2. Under the same v2 policy, the frozen text gain on `test_v1` is +0.025 (unpaired: 0.820 vs 0.795) against +0.016 on `test_clean`, and the prior-report gain about +0.005 against +0.007. So `test_clean`'s higher prior coverage doesn't explain the gaps; the label policy does (§6 M4b amendment). Text gains are judged under v2 |
| 2026-10-21 | M4b | 9e62548 | compare | `test_v1` | difference +0.025 [0.021, 0.029] | 12 | Label policy, paired: `rad-dino_518_ctx_image_text` minus `rad-dino_518_ctx_image` under v2; on fine-tuned features +0.019 [0.016, 0.022]. On `test_clean` under the v1 policy (5 classes): -0.000 [-0.014, 0.013] frozen and -0.007 [-0.018, 0.004] fine-tuned. Text gains vanish when only mentioned findings count, as the §6 M4b amendment predicts |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_attn_h0 | `test_clean` | 0.789 [0.771, 0.805] | 9 | Head validity check passed: val 0.823 against the probe's 0.803. Minus `rad-dino_518_probe`: +0.015 [0.007, 0.023]. Attention pooling over frozen patch tokens matches the fine-tune (0.787) without one |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_attn_h0 | `test_v1`, v1 metric | 0.818 [0.802, 0.833] | 8 | |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_attn_h1 | `test_clean` | 0.798 [0.781, 0.813] | 9 | Minus `h0`: +0.009 [-0.002, 0.020]. Val 0.840 |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_attn_h1 | `test_v1`, v1 metric | 0.813 [0.796, 0.828] | 8 | |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_attn_h2 | `test_clean` | 0.802 [0.786, 0.818] | 9 | Minus `rad-dino_518_ctx_image_text`: +0.011 [0.002, 0.021], which mixes the stronger head with the fusion change. Text moves its attention (mean total variation distance 0.300), and its best epochs come earlier (2-3) than `h1`'s (5-8). Val 0.839 |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_attn_h2 | `test_v1`, v1 metric | 0.816 [0.800, 0.831] | 8 | |
| 2026-10-21 | G4b | 9e62548 | gate | `test_clean` | difference +0.004 [-0.003, 0.011] | 9 | **FAIL**: `rad-dino_518_attn_h2` minus `rad-dino_518_attn_h1`; on `test_v1` -0.002 [-0.006, 0.002]. Text steers the attention but not the scores, so nothing detectable is lost by fusing late; this argues against a joint image-text fine-tune (§6 M4b). 15 heads in 8.5 min on the Mac (MPS); the token cache took 53 min and 7.1 GB |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_ctx_image_prior_labels | `test_clean` | 0.783 [0.764, 0.799] | 9 | Minus `ctx_image_prior` (TF-IDF of the same prior report): +0.001 [-0.004, 0.007]; on `test_v1` +0.003 [0.001, 0.005]. The prior study's labels add next to nothing over its report text |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_ctx_image_prior_labels | `test_v1`, v1 metric | 0.806 [0.790, 0.821] | 8 | |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_ctx_image_text_prior_labels | `test_clean` | 0.794 [0.776, 0.811] | 9 | Minus `ctx_image_text`: +0.003 [-0.000, 0.006]; on `test_v1` +0.002 [0.001, 0.003]. `prior_labels` alone: 0.643 [0.619, 0.667] (`test_v1`, v1 metric: 0.582); no red flags |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_ctx_image_text_prior_labels | `test_v1`, v1 metric | 0.814 [0.799, 0.828] | 8 | |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_ft_s1_stack_text | `test_clean` | 0.802 [0.784, 0.819] | 9 | Minus `rad-dino_518_ft_s1`: +0.016 [0.009, 0.023]; on `test_v1` +0.022 [0.018, 0.025]. Fit on val, the text gain on the fine-tuned image is close to the frozen one (+0.019, next row). The earlier +0.009 came mostly from fitting on training studies; fine-tuning absorbs little of the text |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_ft_s1_stack_text | `test_v1`, v1 metric | 0.826 [0.810, 0.841] | 8 | |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_probe_stack_text | `test_clean` | 0.793 [0.776, 0.810] | 9 | The check: minus `rad-dino_518_ctx_image_text` +0.002 [-0.002, 0.007], so the two-score combiner loses little. Minus `rad-dino_518_probe`: +0.019 [0.011, 0.027]. Standardized weights: image 1.12, text 0.48 |
| 2026-10-21 | M4b | 9e62548 | rad-dino_518_probe_stack_text | `test_v1`, v1 metric | 0.811 [0.795, 0.826] | 8 | The whole M4b stage took 71 min on the Mac. The external drive dropped off the bus twice while idle; the doctor caught the stale mount before the run, which then used `caffeinate -ims` |
| 2026-10-22 | M4b, follow-up | 46afb0a | compare | `val` | difference +0.002 [0.000, 0.003] | 12 | `rad-dino_518_ctx_image_text_prior_labels` minus `rad-dino_518_ctx_image_text`. The pre-registered rule (§6 M4b part 2) fires: the prior-label encoding joins the text recipe for later work (§6 M5b amendment). The gain is negligible |
| 2026-10-22 | M4b, follow-up | 46afb0a | compare | `test_clean` | difference +0.003 [-0.007, 0.013] | 9 | `rad-dino_518_attn_h0` minus `rad-dino_518_ft_s1`; on val +0.002 [-0.003, 0.007]. The attention head on frozen tokens is within 0.007 below and 0.013 above the fine-tune: the match is now measured |
| 2026-10-22 | M5 | 46afb0a | audit (benchmark) | official test 3,041 studies / 289 patients; val 1,733 / 487; train 213,357 / 63,169 | n/a | 12 (v2 and positive vs rest) | **G0 PASS** on the official split. 9,696 studies dropped (no frontal image), 0 unreadable. RAD-DINO overlap with val and test: 0. 96.1% of test studies have an indication, about MMBT's 2,920. Prepare took 9 h 10 min, bound by the USB hard drive (about 6.6 images/s); embeddings and tokens 6.1 h on MPS; 62 GB |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_probe | official test, positive vs rest | 0.781 [0.763, 0.796] | 12 | ResNet-50 (0.725) is below ours. v2 policy 0.788 [0.770, 0.803]; val 0.815. Best C at the grid's edge for Consolidation and Pneumonia (1e-4) and Fracture (10) |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_attn_h0 | official test, positive vs rest | 0.797 [0.781, 0.811] | 12 | ResNet-50 below ours. Minus the probe +0.016 [0.010, 0.022] (v2 policy), the same as in M4b with 9x the training studies. Head check passed (val 0.825 against 0.815). v2 policy 0.804 [0.788, 0.818]. 10 heads in 139 min on MPS |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_ctx_image_indication | official test, positive vs rest | 0.798 [0.783, 0.812] | 12 | MMBT's inputs: **MMBT (0.794) level**, Attentive (0.751) below ours; the same on the studies with an indication. Minus `image` +0.018 [0.013, 0.024] (v2 policy). Val 0.849. Lower than MMBT on Cardiomegaly (0.795 against 0.826) and Pneumonia; higher on Lung Lesion, Fracture and Pneumothorax |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_ctx_indication | official test, positive vs rest | 0.675 [0.659, 0.688] | 12 | **BERT (0.704) above ours**, our one like-for-like loss: TF-IDF + SVD against a fine-tuned BERT. No red flags. Post hoc, not tested: MMBT ran the labeler on full reports, which contain the indication, so some of its labels can be read off its own text input; ours come from the impression |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_ctx_image_text | official test, positive vs rest | 0.806 [0.791, 0.819] | 12 | All three text sources, so not like for like. Minus `image` +0.026 [0.020, 0.032] (v2 policy). A prior report exists for 89% of test studies against 68% of train |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_ctx_text | official test, positive vs rest | 0.732 [0.716, 0.746] | 12 | Text only, all sources. No red flags |
| 2026-10-22 | M5 | 46afb0a | rad-dino_518_attn_h1 | official test, positive vs rest | 0.818 [0.803, 0.832] | 12 | Best M5a run (val 0.857); not like for like. Minus `h0` +0.023 [0.016, 0.030] (v2 policy). v2 policy 0.827 [0.812, 0.840]. The whole M5a run took about 18 h of compute, 9 of them reading the drive |
| 2026-10-22 | M5b | a7a6af1 | rad-dino_518_attn_h0_stack_indication | official test, positive vs rest | 0.807 [0.792, 0.821] | 12 | Image + indication through the val combiner (M5b amendment): **MMBT (0.794) level**, 0.002 above our lower bound; Attentive below ours. Minus `attn_h0` +0.011 [0.005, 0.016] (v2 policy). All three image + indication runs (0.794-0.807) are level with MMBT. Val 0.834 (cross-fitted) |
| 2026-10-22 | M5b | a7a6af1 | rad-dino_518_probe_stack_indication | official test, positive vs rest | 0.794 [0.777, 0.808] | 12 | The check: minus `ctx_image_indication` -0.005 [-0.009, -0.001]. On the benchmark the two-score combiner loses about 0.005 against fitting all features jointly (on the v2 cohort it lost nothing), so its estimates on fine-tuned features are lower bounds (§6 M5b amendment). Standardized weights: image 1.21, text 0.49 |
| 2026-10-22 | M5b | a7a6af1 | rad-dino_518_probe_stack_text_prior_labels | official test, positive vs rest | 0.805 [0.790, 0.819] | 12 | The check with all text: minus `ctx_image_text_prior_labels` -0.005 [-0.010, -0.001], the same reading |
| 2026-10-22 | M5b | a7a6af1 | rad-dino_518_attn_h0_stack_text_prior_labels | official test, positive vs rest | 0.816 [0.801, 0.830] | 12 | All pre-imaging text, so not like for like. Minus `attn_h0` +0.019 [0.013, 0.026] (v2 policy) |
| 2026-10-22 | M5b | a7a6af1 | rad-dino_518_ctx_image_text_prior_labels | official test, positive vs rest | 0.810 [0.796, 0.823] | 12 | Minus `ctx_image_text` +0.004 [0.002, 0.007] under both policies: the prior study's labels add more here than on the v2 cohort (+0.001 to +0.003), where fewer studies have a prior |
| 2026-10-22 | M5b | a7a6af1 | rad-dino_518_ctx_text_prior_labels | official test, positive vs rest | 0.741 [0.725, 0.756] | 12 | Text only. No red flags. The Mac part took 10 min |
| 2026-10-22 | M5b | 98896de | bug | | | | `finetune --head attention` cast tokens to float64 on the device, which MPS can't do: the Mac's tests and wiring check failed at M3 (CUDA is unaffected). Fixed in the next commit, with a test that rejects float64 before `.cpu()`. The Mac part ran on a7a6af1 at the user's direction, since `bench_stack` never runs `finetune.py` |
| 2026-10-22 | M5b | 7ffaee3 | rad-dino_518_ft_s0 | official test, positive vs rest | 0.804 [0.788, 0.818] | 12 | Mean-pool fine-tune, 3 epochs on 213k studies (Lambda GH200, 52.5 min with the embeddings, 223 img/s). Val by epoch 0.820, 0.827, 0.830: best at the last epoch with val loss still falling, so 3 epochs didn't overfit and may be short. v2 policy 0.812 [0.796, 0.826]. ResNet-50 below ours |
| 2026-10-22 | M5b | 7ffaee3 | rad-dino_518_ft_attn_s0 | official test, positive vs rest | 0.807 [0.791, 0.820] | 12 | The attention-head fine-tune (M5b amendment). Minus `ft_s0` +0.002 [-0.002, 0.006] (v2 policy; +0.002 [-0.001, 0.006] positive vs rest): after fine-tuning, attention pooling adds nothing detectable, against +0.016 on frozen features. Minus `attn_h0` +0.010 [0.005, 0.015] (both policies): at 213k studies fine-tuning adds over the frozen head, where at 24k it only matched it. Val by epoch 0.818, 0.822, 0.830, best at the last. 49.7 min |
| 2026-10-22 | M5b | 7ffaee3 | rad-dino_518_ft_attn_s0_stack_indication | official test, positive vs rest | 0.815 [0.801, 0.829] | 12 | **MMBT (0.794) below ours**, like for like (image + indication through the val combiner, which the check marks as a lower bound); the same on the studies with an indication, 0.815 [0.800, 0.827]. Minus `ft_attn_s0` +0.009 [0.004, 0.014] (v2 policy). One seed |
| 2026-10-22 | M5b | 7ffaee3 | rad-dino_518_ft_s0_stack_indication | official test, positive vs rest | 0.813 [0.797, 0.827] | 12 | **MMBT below ours** as well. Minus `ft_s0` +0.008 [0.003, 0.014] (v2 policy). With all text: `ft_s0_stack_text_prior_labels` 0.819, `ft_attn_s0_stack_text_prior_labels` 0.822 [0.808, 0.836] |
| 2026-10-22 | M5b | 7ffaee3 | rad-dino_518_ft_s0_ctx_image_indication | official test, positive vs rest | 0.820 [0.805, 0.833] | 12 | Exploratory, not lined up (M4's fusion on the fine-tuned features). Minus `ft_s0_ctx_image` (the refit probe, 0.805: +0.001 over the fine-tune's own head) +0.016 [0.012, 0.021] (v2 policy). Here the probe fit on training studies gives the larger text gain, the reverse of the v2 cohort: the fine-tunes didn't memorize 213k studies, and the combiner loses about 0.005. Text adds about the same to frozen and fine-tuned images again (+0.018 and +0.016) |
| 2026-10-22 | M5b | 7ffaee3 | rad-dino_518_ft_s0_ctx_image_text_prior_labels | official test, positive vs rest | 0.825 [0.811, 0.837] | 12 | Best number so far; exploratory and not like for like (all pre-imaging text). Minus `ft_s0_ctx_image` +0.021 [0.015, 0.027] (v2 policy). The GPU part took 2 h 16 min of instance time (about $5); the Mac part 32 min |

## 10. Backlog (after v2.0, one at a time)

Ordered by expected value per unit of cost:

1. **More data, unbiased population:** train on all frontal MIMIC-CXR studies of
   train-split patients, not just the v1 cohort (~4× more abnormal studies,
   ~6× more studies in total). Likely the biggest single gain. It also removes
   the cohort's time-of-day and ED-window selection (§3), and grows
   `test_clean` beyond 870 studies. Declare the fine-tune schedule before
   running it: in v2.0 both RAD-DINO seeds peaked at epoch 1 of 10 and then
   overfit, so use about 3 epochs (1 of warm-up) and keep best-epoch selection.
2. **Higher resolution** (RAD-DINO at 1024 px with interpolated position
   embeddings) for subtle findings such as pneumothorax and fracture.
3. **Parameter-efficient fine-tuning** (LoRA) of MedSigLIP's image encoder.
4. **Both views:** use the lateral image alongside the frontal one.
5. **Anomaly detection** (normal vs. abnormal) from foundation embeddings, as
   the distance to the nearest normal studies. Uses the normal cohort and
   revisits the repo's original goal.
6. **Reports as a training signal only:** fine-tune RAD-DINO with an extra
   loss that aligns each image with its own report (findings and impression).
   Reports are used in training only and the test is image-only, so nothing
   leaks. It runs after M4 whichever way G4 goes (§6 M4); if G4 passes, it
   shares the GPU session with the joint image-text fine-tune.
7. Ensembles and test-time augmentation.
8. **LP-FT:** start the fine-tune's head from the frozen probe instead of from
   scratch. It tends to distort pretrained features less and to transfer
   better under distribution shift. Compare it paired against the v2 baseline.
9. **A stronger indication encoder:** fine-tune a BERT-family model on the
   indication with our labels, then add it with the val combiner. On the
   benchmark the indication alone trails MMBT's BERT (0.675 against 0.704); this
   tells whether the gap is our TF-IDF features or MMBT's labels (next item).
10. **Relabel from full reports:** MMBT ran the CheXpert labeler on full
   reports, indication included, and the labeler marks a finding positive if
   any sentence asserts it. Labeling the official test split both ways, on the
   Mac, would measure how much of a published text gain comes from labels the
   indication itself produces.

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
- LP-FT (fine-tuning can distort pretrained features): <https://arxiv.org/abs/2202.10054>
