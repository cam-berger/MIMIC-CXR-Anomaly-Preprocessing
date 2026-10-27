# Chest X-ray findings on MIMIC-CXR: an exploratory analysis

We explored how far a modern pretrained chest X-ray encoder gets on 12 findings
in MIMIC-CXR, what text written before the X-ray adds, how the results line up
with published models, and whether they hold at another hospital. Everything
here comes from the `v2/` package; [`v2/README.md`](../v2/README.md) explains
how to run it.

## Findings

- **A pretrained encoder beats training from scratch.** RAD-DINO with a linear
  classifier matches our original multimodal pipeline (v1) on v1's own test set,
  0.803 against 0.792, from the image alone. Fine-tuned, it reaches 0.823 there.
- **On the standard MIMIC-CXR benchmark** (official split, 3,041 test studies),
  the fine-tuned model scores 0.807 from the image and 0.812-0.819 with the
  indication. The best published model with the same inputs, MMBT, scores 0.794;
  the published image-only ResNet-50, 0.725.
- **Text written before the X-ray helps a little, and consistently.** The
  indication, the ED chief complaint and the prior report add +0.016 to +0.026.
  Vitals and demographics add nothing. With all of it, the benchmark score is
  0.825.
- **How the text is used matters little.** Adding it at the end works as well as
  letting it steer where the image model looks. General BERT matches medical
  BERTs. Fine-tuning BERT on the indication lifts text alone by +0.033 but moves
  the image + text system by at most 0.004.
- **The labels hold the scores down.** Scored against a radiologist's labels
  instead of the automatic labeler's, the same model gains +0.03.
- **The model transfers.** On the CheXpert test set (another hospital, labeled by
  five radiologists) it scores 0.904 on the five standard findings without
  seeing a CheXpert label, level with CheXzero (0.889) and KAD (0.905).
- **The results are stable.** A second training seed moves every benchmark and
  CheXpert score by 0.002 at most.
- **It's cheap.** About $18 of rented GPU time in all (8.5 hours); v1's single
  training run cost $54.

## From v1 to v2

v1 trained everything from scratch: a ViT image encoder, pretrained as a masked
autoencoder on normal X-rays, fused with labs, vital signs and text about the
visit. A review of its code and evaluation
([V1.md](V1.md#what-v2-found-about-v1s-evaluation)) found problems that made its
numbers hard to trust and impossible to compare with published work. v2 kept
the question and the 12 findings and changed the rest. Each change below
answers something v1 showed.

**Measurement first.** v1 reported 0.701, but it picked its best epoch on the
same set it reported, and four of its findings had only 2 to 23 negative
studies there, too few to measure. Without those four, v1 scores 0.792. v2
scores a finding only when it has at least 30 positive and 30 negative studies,
gives every score a 95% interval, makes every choice on a separate validation
split and scores each test set once. Every later change was judged this way.

**The studies.** v1's cohort held 23% of MIMIC-CXR's abnormal studies, and a
time-parsing bug dropped almost every study taken before 10:00. v2 kept that
cohort at first, to compare with v1 on the same patients, then moved to all
218k frontal studies on MIMIC-CXR's official split, the split published results
use.

**The image.** v1 meant to pick the frontal view by file name, but MIMIC-CXR
names its files by DICOM ID, so the rule never matched and 31% of its training
images were side views. It also fed the encoder a 1024 px center crop, about
13% of a 3,056 × 2,544 X-ray, so the model never saw the whole chest. v2 reads
the view from MIMIC-CXR's metadata (PA, else AP), shrinks the image to 518 px
on its shorter side and trims the longer side to a square, which keeps about
83% of it.

**The image encoder.** Both versions use a ViT-Base. v1 pretrained its own as a
masked autoencoder on about 20,000 normal X-rays; the recipe called for 800
epochs at 1024 px, weeks of GPU time, and training the classifier on top took
36 hours and $54. v2 starts from RAD-DINO, a ViT-Base already pretrained on
883,000 chest X-rays. With the encoder frozen and only a linear classifier on
top, it matches v1's whole multimodal model on v1's own test set from the image
alone (0.803 against 0.792); fine-tuned, it reaches 0.823. The same
architecture pretrained on natural images (DINOv2) scores 0.060 lower, so the
gain comes from pretraining on chest X-rays, not from the architecture. A
fine-tune on all 213,000 benchmark training studies takes about 50 minutes.

**What goes in.** The labels are extracted from the report, so anything written
after the X-ray can carry the answer. v1's text was either that report or a
description of the visit that included data recorded after the X-ray, such as
labs and discharge diagnosis codes. v2 uses only what existed before the X-ray:
the indication, the ED chief complaint, the prior report and the triage vitals.
It uses no labs. As a check, no finding scores above 0.90 from the text alone.
The text adds about +0.02, while the triage vitals, age and sex add nothing
(-0.001), so v2 keeps the text and drops v1's structured branch.

**Combining text and image.** v1 used cross-attention, but over one image
vector and one text vector, so each attention step had a single item to look
at and acted as a plain linear layer. v2 puts the text features next to the
image features in one classifier. It also tested real attention: a head over
the image's patch tokens helps the frozen image (+0.015), but letting the text
steer it adds nothing over adding the text at the end (+0.004 [-0.003, 0.011]).

**The text model.** v1 tokenized its text with Bio_ClinicalBERT's vocabulary
and encoded it with PubMedBERT, which reads the same IDs as different tokens.
v2 started with TF-IDF, then tried BERT, BiomedBERT and ClinicalBERT, each with
its own tokenizer, frozen and fine-tuned. With the image, all seven text models
land between 0.812 and 0.819, so TF-IDF, the simplest, is enough.

**The labels.** v1 left out every finding a report doesn't mention. In an
abnormal-only cohort that leaves some findings with almost no negatives:
Atelectasis had 1,329 labeled validation studies, 98.3% of them positive. v2
counts an unmentioned finding as negative, as published work does, and leaves
out only the uncertain ones.

**Training length.** v1 trained for 50 epochs. RAD-DINO's validation score
peaks after one or two epochs and then falls, so the benchmark fine-tunes run
for 3 epochs and keep the best on validation. Six epochs add only +0.003 on the
benchmark.

**Findings, not anomaly scores.** v1 also scored anomalies by how poorly its
autoencoder reconstructed an image. That score can't say which finding is
present, so v2 works on the 12 labeled findings, which can be checked against
labels, a radiologist and published models.

**New checks.** v2 lines its models up with published results on the official
split, scores them against a radiologist's labels on 596 test studies, tests
the image model on the CheXpert test set at Stanford, and repeats the
fine-tunes with a second seed. The results are under [Findings](#findings).

## Setup

| | |
|---|---|
| Data | MIMIC-CXR-JPG 2.1.0; MIMIC-CXR reports for the indication and prior report; MIMIC-IV-ED for the chief complaint and vitals |
| Image | One frontal image per study: PA if there is one, else AP |
| Findings | Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax |
| Labels | MIMIC-CXR-JPG's CheXpert labels: 1 is positive, 0 or blank negative, -1 (uncertain) left out. Comparisons with published results use their convention instead: positive against everything else |
| Text | Only what was written before the X-ray. The study's own report is never an input |
| Metric | Macro AUROC over the findings with at least 30 positive and 30 negative studies in the test set, with a 95% interval from a patient-level bootstrap (1,000 resamples); differences between models are paired |
| Choices | Every choice (model, epoch, regularization, combination) is made on the validation split. Test sets are scored once, at the end |

**Test sets**
- **The benchmark:** every frontal study in MIMIC-CXR, on the official split:
  213,357 training, 1,733 validation and 3,041 test studies (289 patients).
- **The v2 cohort:** v1's abnormal cohort (32.5k studies), split by patient.
  `test_clean` holds its 870 studies from official validation and test patients;
  `test_v1` is v1's own validation split, kept to compare with v1.
- **A radiologist's labels:** 596 official test studies that one radiologist
  labeled from the reports (shipped with MIMIC-CXR-JPG).
- **The CheXpert test set:** 500 studies from Stanford, labeled by the majority
  vote of five radiologists.

**Models:** RAD-DINO (a ViT-B pretrained without labels on 883k chest X-rays),
MedSigLIP (a medical image-text model), DINOv2 (the same architecture, natural
images only) and DenseNet-121. For text: TF-IDF, and BERT, BiomedBERT and
ClinicalBERT, frozen or fine-tuned.

## What we did, step by step

### M0: data audit

v1's cohort turned out to be a narrow slice of MIMIC-CXR: abnormal studies
within 24 hours of an ED visit, taken between 10:00 and midnight. v1's time
parser dropped earlier studies, and 31% of its training images were side views.
The v2 cohort keeps the same studies, with one frontal image each and test
patients that no model has seen. The clean test set has 870 studies and 9
measurable findings.

### M1-M3: encoders, frozen, zero-shot and fine-tuned

On `test_clean`, at 518 px:

| Model | Macro AUROC |
|---|---|
| RAD-DINO, frozen + linear classifier | 0.775 [0.756, 0.791] |
| MedSigLIP, frozen + linear classifier | 0.782 [0.765, 0.798] |
| DINOv2, frozen + linear classifier | 0.715 [0.695, 0.732] |
| MedSigLIP, zero-shot | 0.666 [0.646, 0.687] |
| DenseNet-121, fine-tuned | 0.768 [0.748, 0.785] |
| **RAD-DINO, fine-tuned** | **0.787 [0.769, 0.803]** |

Chest X-ray pretraining is worth +0.060 over DINOv2. Fine-tuning adds +0.012
over the frozen encoder, and two seeds agree. MedSigLIP trained on MIMIC-CXR
images and reports with an undocumented split, so it stayed a reference. On v1's
test set, under v1's metric, frozen RAD-DINO scores 0.803 and the fine-tune
0.823, against v1's 0.792.

### M4: text written before the X-ray

Text features (TF-IDF) joined the frozen image features in one classifier.

| Added to the image | Change |
|---|---|
| All text (indication, chief complaint, prior report) | +0.016 [0.009, 0.024] |
| Indication | +0.009 [0.004, 0.014] |
| Prior report | +0.007 [0.002, 0.012] |
| ED chief complaint | +0.003 [-0.001, 0.007] |
| Triage vitals, sex, age | -0.001 [-0.003, 0.001] |

Text alone scores 0.692, and no text-only finding scores above 0.90, the
check for text that gives the answer away.

### M4b: attention and combining

- An attention head over frozen patch tokens matches the fine-tune (0.789
  against 0.787) without fine-tuning the encoder.
- Letting the text steer that attention adds nothing over adding the text at
  the end (+0.004 [-0.003, 0.011]).
- The prior study's labels add nothing beyond its report text.
- A simple combiner fit on validation (image score plus text score) carries the
  text gain over to the fine-tuned image: +0.016, to 0.802.

### M5: the standard benchmark

All 218k frontal studies, on the official split, scored like the published
results:

| System | Inputs | Macro AUROC | Published, same inputs |
|---|---|---|---|
| Frozen RAD-DINO + linear classifier | image | 0.781 [0.763, 0.796] | ResNet-50 0.725 |
| Attention head on frozen tokens | image | 0.797 [0.781, 0.811] | |
| Fine-tuned, mean pooling | image | 0.804 [0.788, 0.818] | |
| **Fine-tuned, attention head** | image | **0.807 [0.791, 0.820]** | |
| Fine-tuned + indication (TF-IDF) | image + indication | 0.815 [0.801, 0.829] | MMBT 0.794, Attentive 0.751 |
| Indication alone (TF-IDF) | indication | 0.675 [0.659, 0.688] | BERT 0.704 |
| Fine-tuned features + all pre-imaging text | image + indication, complaint, prior report | 0.825 [0.811, 0.837] | none |

Fine-tuning on 213k studies adds +0.010 over the attention head on frozen
tokens. Each fine-tune took about 50 minutes on a GH200.

### M6: language models for the text

| Indication model | Indication alone |
|---|---|
| TF-IDF | 0.675 [0.659, 0.688] |
| BERT, frozen | 0.676 |
| BERT, fine-tuned | 0.707 [0.691, 0.720] |
| BiomedBERT, fine-tuned | 0.713 [0.695, 0.726] |
| ClinicalBERT, fine-tuned | 0.708 [0.692, 0.720] |

Fine-tuned BERT matches MMBT's BERT (0.704). Medical pretraining adds nothing
measurable (+0.001 [-0.005, 0.006]). With the fine-tuned image, all seven text
models (TF-IDF, and the three BERTs frozen or fine-tuned) land between 0.812
(frozen BERT) and 0.819 (fine-tuned BERT or BiomedBERT); TF-IDF gives 0.815. Validation picked the frozen-BERT system (0.812), the
lowest of the seven on test: differences that small sit inside validation's
noise. From then on a change was adopted only if its paired validation gain was
clearly above zero.

### M7: two external checks

**A radiologist's labels.** On the 596 official test studies a radiologist
labeled, the image model scores +0.029 [0.012, 0.046] higher against the
radiologist than against the automatic labeler. The largest moves are Pneumonia
(+0.116) and Cardiomegaly (+0.067, to 0.817). The labeler calls more positives
than the radiologist on these findings.

**The CheXpert test set.** The same image model, applied unchanged:

| Model | Trained on | Five-finding mean |
|---|---|---|
| **Ours, fine-tuned RAD-DINO** | MIMIC-CXR labels | **0.904 [0.886, 0.919]** |
| Ours, frozen RAD-DINO + linear classifier | MIMIC-CXR labels | 0.900 [0.885, 0.915] |
| CheXzero (Tiu et al. 2022) | MIMIC-CXR images and reports | 0.889 |
| KAD (2023) | MIMIC-CXR images and reports | 0.905 |
| CARZero (CVPR 2024) | MIMIC-CXR images and reports | 0.923 |
| DenseNet-121 | CheXpert's own labels | 0.902 |
| DAM (2021) | CheXpert's own labels | 0.931 |

Per finding: Atelectasis 0.889, Cardiomegaly 0.897, Consolidation 0.871, Edema
0.911, Pleural Effusion 0.953. The lead over CheXzero is all Atelectasis (0.889
against 0.816). Fine-tuning adds little here (0.900 frozen against 0.904), and
CheXpert scores sit well above MIMIC's (0.826 on the same five findings), which
fits the cleaner radiologist labels. KAD's and CARZero's numbers are their own;
an independent re-evaluation of their public weights found 0.699 and 0.824.
RAD-DINO's pretraining included CheXpert's training images, without labels and
without any test patients.

### M8: a second seed, and longer training

| Result | Seed 0 | Seed 1 |
|---|---|---|
| Benchmark, image only | 0.807 | 0.806 |
| Benchmark, image + indication (frozen BERT) | 0.812 | 0.811 |
| CheXpert, five findings | 0.904 | 0.904 |
| Radiologist minus labeler | +0.029 | +0.033 |

Seed 1 minus seed 0 is 0.002 or less on every set, about a tenth of a test
interval's half-width. Training the attention head for 6 epochs instead of 3 adds +0.004
[-0.001, +0.008] on validation and +0.003 on test, mostly on the small findings
(Fracture +0.021, Lung Lesion +0.009, Pneumothorax +0.007). The fine-tunes
stay at 3 epochs.

## Compute and cost

| Step | Where | Time | Cost |
|---|---|---|---|
| M3 fine-tunes (v2 cohort) | Lambda A100 | 2.15 h | about $4 |
| M5 fine-tunes (benchmark) | Lambda GH200 | 2.3 h | about $5 |
| M8 seed 1 and 6 epochs | Lambda GH200 | 4.0 h | about $9 |
| Everything else | Apple M5 Pro laptop | | |

Preparing the benchmark took 9 hours, most of it reading 218k JPEGs from a USB
drive; its embeddings and patch tokens took 6 hours on the laptop.

## Limitations

- The training and test labels come from an automatic labeler, which, judged by
  M7, understates every model by about 0.03.
- One frontal image per study; lateral views are unused.
- RAD-DINO's pretraining saw MIMIC-CXR's and CheXpert's training images, never
  their test images.
- Published comparisons use each paper's reported numbers.

## Next directions

- **Image + clinical history at another hospital:** CheXpert Plus releases
  Stanford's reports, whose clinical-history section plays the indication's
  role.
- **Higher resolution** for the small findings that longer training helped.
- **Paired CheXpert comparisons:** score CheXzero's and CARZero's public weights
  on the same 500 images.
- **A hospital RAD-DINO never saw**, such as VinDr-CXR.

## Method notes

Each step started from a written question and decision rule. The full
milestone plan, its decision rules and the run-by-run results log are in git
history: `docs/V2_PLAN.md` at commit 04f5946. Every run's predictions,
reports and the summary table regenerate from the output directories with
`python -m v2.evaluate --all`.
