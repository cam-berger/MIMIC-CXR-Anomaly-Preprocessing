# v3 Plan: Jev as the combiner (text first, the image later)

**Status:** pre-registered 2026-10-26, before any Jev request on MIMIC data; val chose the yes/no format. **v3.0 done, J1 FAIL:** Jev with the estimates and all the text scores 0.803 on the official test, -0.024 [-0.033, -0.016] below v2's val combiner of the same inputs (0.827). Adding text to Jev adds nothing detectable (-0.002 [-0.010, 0.006]), re-reading the estimates costs -0.010, and text alone trails TF-IDF by -0.043 (§9). **v3.1 done, J2: no detectable gain** from showing Jev the X-ray (-0.001 [-0.005, 0.005]); on its own Jev reads X-rays at 0.716, against RAD-DINO's 0.814 (§9). **v3.2 (exploratory):** as an input to v2's val combiner, Jev's text reading adds +0.010 [0.005, 0.014] to the image, about what TF-IDF adds, but nothing detectable on top of v2's best system (+0.002 [-0.000, 0.004]); its X-ray reading hurts (§9). **v3.3 done, L1 FAIL:** as a report labeler Jev's chosen answers score 0.914 against the radiologist, the CheXpert labeler 0.940 (-0.026 [-0.041, -0.010]), almost all of it on Lung Opacity (§9). **v3.5 done, L2: no detectable difference:** a probe trained on 20k studies with Jev's soft labels scores +0.009 [-0.007, 0.027] over the same probe with CheXpert labels, judged by the radiologist (§9). **v3.6 done, L3 PASS (+0.065) but definitional:** on the CheXpert test set the gain sits in Lung Opacity and Enlarged Cardiomediastinum, which CheXpert's radiologists label hierarchically, as Jev's broad questions do; on the five standard findings it is +0.001 (§9) · how to run: [`v3/README.md`](../v3/README.md) · **Branch:** `v3` (from `v2`) · **Started:** 2026-10-26

## 1. Question

v2's best like-for-like systems combine an image model's 12 estimates with the
text written before the X-ray through a per-class logistic regression fit on val
(`v2/stack.py`). v3 asks two things:

1. **The ladder:** how does a frontier classification model, TypeSafe's Jev,
   change as it is given more context? First the image model's estimates alone,
   then the indication, then the chief complaint and the prior report, and in
   v3.1 the image itself.
2. **The gate:** reading the same estimates and the same text, does Jev combine
   them better than v2's logistic regression?

v3.0 is a proof of concept: text only, one stage-1 model, one seed, no training.

## 2. Fixed from v2

| Item | Definition |
|---|---|
| Population and splits | M5's benchmark (`BENCH_OUT`): all frontal MIMIC-CXR studies, official split. Val 1,733 studies / 487 patients; official test 3,041 / 289 (V2_PLAN §9, M5 audit). |
| Labels and policy | The 12 CheXpert findings. Headline under the v2 policy (§3 of V2_PLAN), which is also how text gains are judged (M4b amendment); positive vs rest (`u0`) reported next to it. |
| Metrics | Computed only by `v2/evaluate.py`, from files written by `v2.common.write_predictions`. Macro AUROC over the frozen measurable classes, patient-level bootstrap, paired comparisons. |
| Text | `context.parquet` from `v2.context`: `indication`, `chief_complaint`, `prior_report`. Nothing written after the X-ray, and never the current report's findings or impression. |
| Stage 1 | `rad-dino_518_ft_attn_s0`: image only, the model val chose in M7a and G7. Its logits are shown to Jev as probabilities: sigmoid, as percentages with one decimal (below 0.1% shown as "<0.1%"). Amended before any request: two decimals of a probability would tie most rare findings at 0.00-0.02. |

## 3. Jev

- `POST https://api.typesafe.ai/v1/systemone`, one request per study. `state`
  is the study's context as text (`v3/prompt.py`), with 12 questions, one per
  finding, in one of two formats:
  - **`noul`** (yes/no): the pre-registered prompt, unchanged (prompt hash
    `e99b96a72ad9` for `jev_est_text`). Each answer's `noul`, the probability of
    yes, is that finding's score.
  - **`score`** (10 likelihood bands, `0-10%` to `90-100%`): Jev returns the
    expected band, 0 to 9. The finding's score is that band's likelihood,
    (band + 0.5) / 10, which is monotone, so AUROC is the answer's own.
- **Amendment (2026-10-26, after the pilot, before any metric):** the pilot
  showed Jev's probabilities on a 0.01 grid, which ties most studies on rare
  findings (§9). A score comes on the same grid but over 0-9, about 9 times
  finer. Val picks the format (§7 step 4), the user's call: "whichever works
  best". Cost: on synthetic text a score request takes about 2,830 input tokens
  against 1,450 for yes/no, since each of the 120 levels costs about 11 tokens.
  Five bands would take about 2,170, and digits as labels about 2,520.
- **No training and no examples:** nothing from MIMIC goes into the prompt
  beyond the study's own context. The prompt is fixed in `v3/prompt.py`, and its
  hash goes in every run's sidecar.
- **Model pinned:** a version name (not `jev-latest`), chosen at the pilot from
  `GET /v1/models` and recorded in every sidecar. A run that gets answers from a
  second model fails.
- **Data handling:** the user verified on 2026-10-26 that TypeSafe keeps no data
  (zero data retention). Answers are cached locally in `V3_OUT/jev_cache/`
  (record-level: never printed, listed or committed). Only aggregate reports
  reach an AI assistant (`v2/README.md` §9).

## 4. Arms

| Arm | Jev's `state` holds | Role |
|---|---|---|
| `jev_est` | the 12 estimates | Does Jev keep what the image model knows? |
| `jev_est_indication` | + the indication | Like for like with `rad-dino_518_ft_attn_s0_stack_indication` (MMBT's inputs) |
| `jev_est_text` | + the chief complaint and the prior report | **Primary (J1)** |
| `jev_text` | the three text sources, no estimates | Control: Jev reading the text alone, against v2's TF-IDF `rad-dino_518_ctx_text` |
| `jev_img`, `jev_img_est`, `jev_img_est_text` | the frontal X-ray, alone, + the estimates, + the estimates and the text | v3.1 (§8) |

A missing source is written as "none recorded", so every arm scores every study.
A run is named for its arm and question format, e.g. `jev_est_text_score`; the
ladder and J1 are read in the format val chose.

**Baselines** (v2 runs on the same studies):

- `rad-dino_518_ft_attn_s0`: stage 1 on its own.
- `rad-dino_518_ft_attn_s0_stack_indication`: the val combiner with the indication.
- `rad-dino_518_ft_attn_s0_stack_text`: the val combiner with all three text sources, built by `v3.setup` with `v2.stack`. This is the like-for-like baseline for `jev_est_text`.
- `rad-dino_518_ctx_text`: v2's text-only arm (TF-IDF).

## 5. Gate J1 (fixed now)

**J1:** `jev_est_text`, in the format val chose, minus
`rad-dino_518_ft_attn_s0_stack_text`, official test, v2 policy, paired 95% CI
from 1,000 patient resamples.

- **CI above 0: PASS.** Jev combines the estimates and the text better than the logistic regression.
- **CI includes 0:** no detectable gain over the cheap combiner.
- **CI below 0: FAIL.** Jev loses information the logistic regression keeps.

v3.1 (the image) goes ahead whichever way J1 goes; J1 decides only what the
write-up claims.

**The ladder** (paired on the official test, v2 policy; reported, not gates):

| Step | Comparison |
|---|---|
| Jev keeps the estimates | `jev_est` − `rad-dino_518_ft_attn_s0` |
| + indication | `jev_est_indication` − `jev_est` |
| + complaint and prior report | `jev_est_text` − `jev_est_indication` |
| All the text | `jev_est_text` − `jev_est` |
| Jev against TF-IDF, text alone | `jev_text` − `rad-dino_518_ctx_text` |
| Like for like with MMBT's inputs | `jev_est_indication` − `rad-dino_518_ft_attn_s0_stack_indication` |

## 6. Red flags (checked before J1 is read)

- **Leakage** (M4's flags): `jev_text` beats `rad-dino_518_ft_attn_s0` on the
  official test, or any measurable class has `jev_text` AUROC above 0.90 there.
  Either one: stop and inspect locally before reading anything.
- **Failures:** more than 1% of an arm's studies lack an answer after retries:
  stop.
- **Lost estimates:** `jev_est` more than 0.02 below `rad-dino_518_ft_attn_s0`.
  Jev then doesn't keep the image's information, and the ladder steps measure
  Jev's reading of the text on top of a weaker start. This is reported with the
  ladder; it doesn't stop the run.
- **Coarse scores:** each arm reports its median number of distinct scores per
  class. Few distinct values mean tied ranks, which cap AUROC.

## 7. Order and cost control

1. **Offline smoke** (`python -m v3.smoke`): synthetic benchmark, a fake Jev,
   every arm, the report. Nothing leaves the machine.
2. **Live smoke** (`python -m v3.smoke --live`): the same synthetic data
   through the real API. This checks the request format; no MIMIC data is sent.
3. **Pilot:** 40 val studies, `jev_est_text`, in each format. This measures
   latency, tokens per request, the failure rate and the number of distinct
   scores. Only aggregates are printed.
4. **The format, on val:** `jev_est_text` on all of val in both formats
   (2 × 1,733 requests). The format with the higher val macro AUROC (v2 policy)
   wins; an exact tie goes to `score`, the finer one. Only the point estimate
   decides, and the choice is logged in §9 before anything else runs.
5. **Val, the other three arms**, in the chosen format (3 × 1,733 requests).
   Nothing else is tuned on val. If `jev_est_text` falls more than 0.03 below
   the val stack on val, the prompt is revised and val re-run, at most twice.
   Each revision is logged in §9 with its val score.
6. **Official test, all four arms**, in the chosen format only (4 × 3,041
   requests), once, with the final prompt. `v3.run` refuses test in the other
   format, or before the arm's val run.

About 20,800 requests in all. Answers are cached, so a crash or a re-run never
pays twice for the same (arm, study, prompt, model).

## 8. v3.1: the X-ray itself (amendment, 2026-10-26, after v3.0, before any v3.1 request on MIMIC data)

**Question:** Jev already has the image model's estimates and the text. Does
seeing the X-ray itself help? And how well does Jev read a chest X-ray on its
own?

**Fixed now:**

| Item | Definition |
|---|---|
| Model | `jev-1.14.0`, as in v3.0. On synthetic JPEGs it accepts 384 to 1100 px on a side (256 px and 2048 px are "Invalid image dimensions") and sees the image: P("almost entirely white") 0.98 for a white image, 0.02 for a black one (§9). |
| Image | The benchmark's prepared frontal JPEG (`images/{study_id}.jpg`, the manifest's `image` column), sent as is: 518 px on the shorter side; val and test images run 518-848 px, all 4,774 readable. It goes as a multipart asset behind `TypeSafe-Danger-Zone-Images: true`, and its placeholder opens `state` ("The frontal chest X-ray: ..."). `v3.run` checks every image's size before any request. |
| Questions | The yes/no format val chose in v3.0, unchanged. `v3.run` refuses an image arm in any other format. |
| Data | The X-ray is MIMIC data like the text. It goes to TypeSafe under the zero-retention terms the user verified (§3), which must cover image assets too. |

**Arms** (run names `{arm}_noul`):

| Arm | Jev's request holds | Role |
|---|---|---|
| `jev_img` | the X-ray only | Jev's own reading, against RAD-DINO's (`rad-dino_518_ft_attn_s0`) |
| `jev_img_est` | the X-ray + the 12 estimates | Does the X-ray add to the estimates? (against `jev_est_noul`) |
| `jev_img_est_text` | the X-ray + the estimates + all three text sources | **Primary (J2)** |

**Gate J2 (fixed now):** `jev_img_est_text_noul` minus `jev_est_text_noul`,
official test, v2 policy, paired 95% CI from 1,000 patient resamples.

- **CI above 0: PASS.** The X-ray adds to Jev's estimates and text.
- **CI includes 0:** no detectable gain from the X-ray.
- **CI below 0: FAIL.** The X-ray makes Jev worse.

Reported, not gates (official test, paired): `jev_img` − `rad-dino_518_ft_attn_s0`;
`jev_img_est` − `jev_est_noul`; and `jev_img_est_text` −
`rad-dino_518_ft_attn_s0_stack_text`, which asks whether the X-ray closes J1's gap.

**Red flags:** those of §6, plus **image unread:** `jev_img` below 0.60 on the
official test means Jev barely reads the X-ray, so J2 measures little. That is
reported; it doesn't stop the run.

**Order:** (1) the live image smoke on synthetic images (`python -m v3.smoke
--live-image`), done; (2) the pilot: 40 val studies, `jev_img_est_text`, the
same sample as v3.0's pilots; (3) val, all three arms; (4) the official test,
all three arms, once. Nothing is chosen on val. More than 1% of a split
unanswered stops a run (§6).

**Cost:** an image adds about 900 input tokens per request (about 2,400 on
synthetic studies, against about 1,480 without it). Three arms on 4,774 studies
come to about 14,300 requests and roughly 33M input tokens.

## 8b. v3.2 (exploratory, written 2026-10-26 after v3.1, before any v3.2 run): Jev's answers as inputs

**Question:** Jev can't combine the inputs better than a logistic regression,
but are its readings worth anything as inputs? Here v2's val combiner
(`v2/stack.py`: per class, a logistic regression on two runs' scores, fit on val,
cross-fitted for val's own scores) takes a Jev run as its second input. No new
Jev requests: only the cached answers are used.

**Inputs:** Jev's probabilities are turned into log-odds, clipped at 0.005 and
0.995 because Jev's answers come rounded to 0.01 and can be 0 or 1. The
transform is monotone, so each Jev run's own AUROC is unchanged.

| Stack | Inputs | Read against |
|---|---|---|
| A | image run + `jev_text_noul` | `rad-dino_518_ft_attn_s0_stack_text` (the same with TF-IDF's text), and the image run |
| B | image run + `jev_img_noul` | the image run: does Jev's own X-ray reading add to RAD-DINO's? |
| C | `rad-dino_518_ft_attn_s0_stack_text` + `jev_text_noul` | `stack_text`: does Jev's text reading add on top of v2's best system? |
| D | `stack_text` + `jev_img_noul` | `stack_text`: does Jev's X-ray reading add on top of it? |
| E | `stack_text` + `jev_img_est_text_noul` | `stack_text`: Jev's all-inputs answer on top of it |

C to E feed the combiner's own output into a second combiner. That's nested:
the first combiner's val scores are cross-fitted, and the second is fit on those
same val studies, so its val score is slightly optimistic. The official test
was used for no fit at all.

**Reading:** all comparisons are paired on the official test (v2 policy, u0
next to it), with 95% CIs. There is no gate: an interval above 0 says Jev's
readings carry information the combiner can use. A result here would be a
lead, to be confirmed on a second seed or another test set before anyone
relies on it.

## 8c. v3.3: Jev as a report labeler (written 2026-10-26, before any v3.3 request)

**Question:** every model here learns from, and is scored against, the
CheXpert labeler's reading of each report. In M7a a radiologist's labels on 596
official-test studies raised the image model by +0.029 [0.012, 0.046], so the
labeler's errors are measurable. Reading the same report, does Jev label the 12
findings closer to the radiologist? This is about the labels (the targets),
not about the image model's inputs. The current report is the thing being
labeled, and it never becomes an input to a classifier.

**Fixed now:**

| Item | Definition |
|---|---|
| Studies | M7a's: the radiologist's studies in the benchmark's official test (`mimic-cxr-2.1.0-test-set-labeled.csv`), 596 studies, 268 patients |
| Text | Each study's own report, FINDINGS and IMPRESSION (`v2.context.prior_text`: those sections, else the report up to any addendum; 4,000 characters at most) |
| Questions | 12 `choice` questions, one per finding, with three options in the labeler's terms: **present**; **uncertain** (possible, probable, suggestive of, cannot be excluded); **absent** (said to be absent, or not mentioned). The wording is in `v3/label.py`, its hash in every sidecar. Nothing is tuned: radiologist labels exist only on the official test |
| Model | `jev-1.14.0` |

**Runs** (v2 prediction files on those studies, scored by `v2/evaluate.py`
against the radiologist's labels, with M7a's manifest and the classes
measurable under both sets of labels):

| Run | Score per finding | Role |
|---|---|---|
| `labeler_chexpert` | the CheXpert labeler's label: positive 1, uncertain 0.5, negative or blank 0 | Comparator |
| `labeler_jev` | Jev's chosen option on the same three levels | **Primary (L1)** |
| `labeler_jev_prob` | P(present) + 0.5 × P(uncertain), from Jev's probabilities | Secondary |

The AUROC of a three-level score is the area under its three-point ROC curve.
It rewards calling the radiologist's positives present and the negatives
absent, with uncertain in between.

**Gate L1 (fixed now):** `labeler_jev` minus `labeler_chexpert`, both scored
against the radiologist's labels on those 596 studies, v2 policy (the
radiologist's uncertain labels ignored, blanks negative), paired 95% CI from
1,000 patient resamples. Positive vs rest (u0) is reported next to it.

- **CI above 0: PASS.** Jev labels closer to the radiologist. The next step,
  relabeling the training reports, gets its own plan and budget.
- **CI includes 0:** no detectable difference.
- **CI below 0: FAIL.** The CheXpert labeler is closer.

**Red flag:** `labeler_chexpert` below 0.70 against the radiologist means the
studies or labels are misaligned (a labeler reading the same report should be
far above chance): stop and inspect locally.

**Order:** offline smoke (synthetic reports and a fake Jev); live smoke on
synthetic reports; the pilot on 40 val reports (mechanics only: there are no
radiologist labels there, and no metric is computed); the 596 studies, once;
the report.

**Cost and data:** about 640 requests, about 1.5M input tokens. The current
report's findings and impression go to TypeSafe: MIMIC text, under the
zero-retention terms the user verified (§3).

## 8d. v3.4 (exploratory, written 2026-10-26 after L1, before any v3.4 run): closing the labeler's gap

**Why:** L1 failed on two gaps. One is Lung Opacity (Jev's answer 0.682,
its probabilities 0.846, the CheXpert labeler 0.914). The other is the rest:
on the other nine findings, Jev's three-level answer is level with the labeler
(0.939 against 0.943), while its probabilities reach 0.962. v3.4 tests two
fixes on the cached v3.3 answers, with no new Jev requests. Both were prompted
by results on the same 596 studies, so v3.4 is exploratory: a lead for a
pre-registered test, not a revision of L1.

**1. Hierarchy (decided on the training split only).** In CheXpert's scheme,
Lung Opacity is a parent of Atelectasis, Consolidation, Edema, Lung Lesion and
Pneumonia, and Enlarged Cardiomediastinum is a parent of Cardiomegaly. On the
benchmark's training split, measure how often the CheXpert labeler marks each
parent positive when any of its children is positive. A parent whose rate is
at least **0.90** gets the rule: it is present when any child is present. On
Jev's answers that means max(parent, children) per study, applied to both the
chosen answer and the probability score. A parent below 0.90 gets no rule. The
radiologist's own rate on the 596 studies is reported afterwards as a
diagnostic, never used to decide.

**2. Cut-offs (cross-fitted).** Per finding, two cut-offs on Jev's probability
score, P(present) + 0.5 × P(uncertain), turn it back into present (1),
uncertain (0.5) or absent (0). The cut-offs are fit on the radiologist's
labels (v2 policy) by 5-fold cross-fitting grouped by patient: each study's
label comes from cut-offs fit on the other folds. Each fit picks, from 41
quantiles of the training folds' scores, the pair that maximizes that
finding's AUROC (v2's `auroc`). A finding with only one label value in a
training fold keeps Jev's own answer.

**Runs** (scored against the radiologist exactly as in §8c, paired against
`labeler_chexpert` and `labeler_jev`): `labeler_jev_cut`; and, for each
parent that gets the rule, `labeler_jev_hier` (Jev's answer + hierarchy) and
`labeler_jev_cut_hier` (cut-offs on the hierarchy-adjusted probabilities).

**Fairness note:** the cut-offs learn from radiologist labels (cross-fitted)
and the CheXpert labeler learns from none. A three-level score's AUROC doesn't
change under any monotone relabeling of its levels, though, so the labeler
couldn't gain from cut-offs of its own. Lung Opacity's confusion counts (Jev's
answer by the radiologist's label) are reported as a diagnostic.

## 8e. v3.5: Jev's labels as training targets (written 2026-10-26, before any v3.5 request)

**Question:** do Jev's report labels train a better image classifier than the
CheXpert labeler's? v3.3 compared the labels themselves, where a continuous
score can't be judged fairly against three levels. Here each label source
trains the same model on the same studies, and the radiologist judges the
models.

**Fixed now:**

| Item | Definition |
|---|---|
| Training studies | 20,000 studies sampled from the benchmark's training split with a fixed seed (`v3.label`'s sample, seed 0). Every arm trains on the same studies: those Jev answered |
| Val | All 1,733 val studies, labeled by each arm's own source, used only to choose each finding's C |
| Model | v2's probe: frozen RAD-DINO 518 px embeddings (the benchmark's), standardized on the training studies, one L2 logistic regression per finding, C from v2's grid (1e-4 to 10) by val AUROC |
| Jev's labels | The v3.3 labeler prompt, unchanged (hash `f2cb79b46c1c`, `jev-1.14.0`), on each study's own findings and impression. The narrowed Lung Opacity wording (§9, v3.4) is not used: it was prompted by the test labels |
| Data and cost | About 21,700 new requests (20,000 train, 1,693 val; 40 cached), about 41M input tokens; MIMIC report text to TypeSafe under the verified zero-retention terms (§3) |

**Arms** (predictions on val and the official test):

| Arm | Training targets | Val targets (for C) |
|---|---|---|
| `probe20k_chexpert` | the CheXpert labeler (v2 policy: uncertain ignored) | the CheXpert labeler |
| `probe20k_jev_hard` | Jev's answer: present 1, absent 0, uncertain ignored | Jev's answer |
| `probe20k_jev_soft` | Jev's probabilities as soft targets, P(present) + 0.5 × P(uncertain): each study enters twice, as positive and negative, weighted p and 1 − p | Jev's answer |

The reference is v2's `rad-dino_518_probe`, the same model trained on all
213k training studies with the CheXpert labeler's labels.

**Gate L2 (fixed now):** `probe20k_jev_soft` minus `probe20k_chexpert`,
scored against the radiologist's labels on M7a's 596 official-test studies
(v2 policy, the classes measurable under both sets of labels), paired 95% CI
from 1,000 patient resamples.

- **CI above 0: PASS.** At 20k training studies, Jev's soft labels train a
  better classifier, as judged by a radiologist.
- **CI includes 0:** no detectable difference. With 596 studies, a
  difference under about 0.01 is unlikely to be detectable.
- **CI below 0: FAIL.**

Reported, not gates: `probe20k_jev_hard` minus `probe20k_chexpert` on the same
studies, and every arm on the full official test against the CheXpert
labeler's labels. That second view favours the CheXpert-trained arm, which
learns the same labeler it is scored against.

**Order:** offline smoke; the labels (training sample, then val), run by the
user; `python -m v3.downstream` (local: the probes and the report).

## 8f. v3.6: the v3.5 probes on the CheXpert test set (written 2026-10-26 after L2, before any v3.6 run)

**Why:** L2 came out +0.009 [-0.007, 0.027]: in Jev's favour, but too small for
596 studies to settle. The CheXpert test set gives a second judge with no new
Jev requests. It was prepared by v2's M7b (`EXT_OUT`): 500 Stanford studies
from 500 patients, one frontal image each, labeled by the majority vote of 5
radiologists reading the images. It is a different hospital, with
image-based ground truth.

**Fixed now:**

- **Probes:** the three v3.5 probes, refit exactly as v3.5 fit them. The fit
  is deterministic, and the refit must reproduce their stored predictions to
  within 1e-6 or the run stops. They are applied to M7b's RAD-DINO 518 px
  embeddings of the CheXpert test images (same encoder, size and transformers
  version as the benchmark's). Nothing is trained or chosen on this set.
- **Where it runs:** `V3_OUT/chexpert/` links `EXT_OUT`'s manifest, frozen
  classes and audit, and nothing is written into `EXT_OUT`.
- **Reference:** M7b's `rad-dino_518_probe` (all 213k training studies,
  CheXpert labels).

**Gate L3 (fixed now):** `probe20k_jev_soft` minus `probe20k_chexpert` on the
CheXpert test set, v2 policy, over the classes frozen as measurable there at
M7b (Atelectasis, Cardiomegaly, Edema, Enlarged Cardiomediastinum, Lung
Opacity, Pleural Effusion), paired 95% CI from 1,000 patient resamples. CI
above 0: PASS; includes 0: no detectable difference; below 0: FAIL.
Reported, not gates: the same on the five standard CheXpert findings (M7b's
set), and `probe20k_jev_hard` minus `probe20k_chexpert`. L2 and L3 are read
side by side. A gain that holds on both judges is the result; one that holds
on one of them is a lead.

**Caveat:** RAD-DINO's encoder saw 223k unlabeled CheXpert images in
pretraining (M7b). That affects every probe alike, so it doesn't bias the
comparison between them.

## 8g. v3.7 (exploratory, written 2026-10-26 after L3, before any v3.7 run): the hierarchy control

**Why:** L3's gain sits in Lung Opacity and Enlarged Cardiomediastinum. The
CheXpert test set labels both hierarchically (P(parent | a child positive) =
1.0); Jev nearly does (0.88, 0.95); the CheXpert labeler doesn't (0.33, 0.04).
The control asks whether the gain is Jev's reading of the reports or only the
definition. It applies the same hierarchy to the CheXpert labeler's own labels,
with no Jev involved. It is exploratory, because the L3 diagnostic prompted it.

**The arm:** `probe20k_chexpert_hier` is v3.5's `probe20k_chexpert` with its
labels changed in one way. Before fitting, on the training studies and on val
(used for C), a parent becomes positive whenever any of its children is
positive in the CheXpert labeler's labels. That is Lung Opacity over
Atelectasis, Consolidation, Edema, Lung Lesion and Pneumonia, and Enlarged
Cardiomediastinum over Cardiomegaly, as in §8d. Otherwise the parent keeps its
own label, uncertain or blank. The studies, features, model, C grid and
selection are v3.5's own. No Jev requests.

**Reading** (paired 95% CIs, 1,000 patient resamples, v2 policy):

- **Primary:** `probe20k_jev_soft` minus `probe20k_chexpert_hier` on the
  CheXpert test set, over L3's six measurable classes. CI including 0 or below
  it: the L3 gain is the definition, and the hierarchy rule recovers it
  without Jev. CI above 0: Jev's labels add something beyond the definition.
- **Also reported:** `probe20k_chexpert_hier` minus `probe20k_chexpert` (the
  definition alone); the same comparisons on the five standard findings; and
  on M7a's 596 MIMIC-CXR studies against their radiologist, whose labels are
  not hierarchical.

## 9. Results log

| Date | Step | Commit | Run | Test set | Macro AUROC [95% CI] | Classes | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | setup | d1bfba8 | `rad-dino_518_ft_attn_s0_stack_text` | official test | 0.827 [0.811, 0.840] | 12 | The J1 baseline, built by `v3.setup` (val combiner, image + all three text sources). Minus the image run +0.012 [0.005, 0.019]. The rebuilt `stack_indication` matches the benchmark's copy exactly (largest difference 0.0), 0.824 [0.809, 0.837] |
| 2026-10-26 | live smoke | d1bfba8 | `jev_est_text`, 12 synthetic studies | none | n/a | n/a | Request format checked on synthetic text: 12 of 12 answered. `GET /v1/models` lists only the aliases `jev-latest` and `jev-preview`; `jev-1.14.0` (named in LiteLLM's docs) is accepted, and every answer reports it, so it is the pinned model (§3) |
| 2026-10-26 | pilot | d1bfba8 | `jev_est_text`, 40 val studies | none | n/a | n/a | 40 of 40 answered, 0 failed; 1,521 input tokens per study (mean); latency 0.21 s median, 0.27 s p90, at 4 in flight. **Jev's probabilities come on a 0.01 grid** (480 of 480 answers): 69 distinct values overall, 8-30 per finding over 40 studies, fewest for Fracture (8) and Pleural Other (9). Tied ranks cap AUROC on rare findings |
| 2026-10-26 | amendment | c0f3871 | `score` format | none | n/a | n/a | §3 and §7 amended before any metric: a 10-band `score` format next to `noul`, val picks. On 12 made-up studies, `score` answers came on a 0.01 grid over 0-9 (56 distinct values of 144, against 35 for `noul`). Live smoke on synthetic text, both formats: 12 of 12 answered by `jev-1.14.0` in each; 1,477 against 2,857 input tokens per request |
| 2026-10-26 | pilot | c0f3871 | `jev_est_text_score`, 40 val studies | none | n/a | n/a | The same 40 studies as the yes/no pilot: 40 of 40 answered, 0 failed; 2,901 input tokens per study (mean), against 1,521; latency 0.18 s median, 0.23 s p90. Distinct scores per finding, median 27.5 against 19.5; Fracture 18 against 8, Pleural Other 19 against 9; 162 distinct values over all findings against 69 |
| 2026-10-26 | format choice (§7 step 4) | c3c5242 | `jev_est_text_noul` against `jev_est_text_score` | val (1,733 studies) | 0.840 [0.819, 0.857] against 0.827 [0.803, 0.847] | 12 | **Val chose `noul` (yes/no).** Score minus noul -0.013 [-0.024, -0.005]: the finer scores rank worse despite 240 distinct values per finding against 71. Both formats answered 1,733 of 1,733; 2.64M against 5.03M input tokens. Val prompt rule: `jev_est_text_noul` minus `rad-dino_518_ft_attn_s0_stack_text` on val -0.004 [-0.019, 0.010], within the 0.03 margin, so the prompt stands. For reference on val: the image run 0.830, the stack 0.844. No red flags |
| 2026-10-26 | J1 | e4cd26e | `jev_est_text_noul` | official test | 0.803 [0.787, 0.816] | 12 | **J1 FAIL:** minus `rad-dino_518_ft_attn_s0_stack_text` (0.827) -0.024 [-0.033, -0.016] (u0 -0.023 [-0.032, -0.015]): Jev loses information the logistic regression keeps. On val the same comparison was -0.004 [-0.019, 0.010]; Jev's text arms drop more from val to test (-0.032 to -0.037) than the image run (-0.016) or the stack (-0.017). No red flags. Every arm answered 1,733 of 1,733 val and 3,041 of 3,041 test studies |
| 2026-10-26 | ladder | e4cd26e | `jev_est_noul`, `jev_est_indication_noul`, `jev_text_noul` | official test | 0.805 [0.788, 0.818], 0.808 [0.791, 0.822], 0.695 [0.675, 0.711] | 12 | **What context does to Jev:** re-reading the image's estimates costs -0.010 [-0.015, -0.005] against the image run (0.814), all of it on rare findings (Pleural Other -0.050, Lung Lesion -0.026, Enlarged Cardiomediastinum -0.022, Fracture -0.015; the common findings within 0.002), which fits the 0.01 grid. The indication adds +0.003 [-0.003, 0.009], the complaint and prior report -0.005 [-0.012, 0.002], all the text -0.002 [-0.010, 0.006]: nothing detectable, where the same text adds +0.012 [0.005, 0.019] through the val combiner. Text alone, Jev trails TF-IDF by -0.043 [-0.061, -0.026]; Enlarged Cardiomediastinum is at chance (0.498 against 0.687), and adding the text costs it -0.083. Text helps Fracture (+0.044), Lung Lesion (+0.024) and Pneumothorax (+0.023). Like for like with MMBT's inputs, -0.016 [-0.022, -0.010] against the indication stack. Per-finding numbers are test diagnostics, not choices. About 32M input tokens in all |
| 2026-10-26 | v3.1 live image smoke | 94ad9b9 | synthetic images | none | n/a | n/a | `jev-1.14.0` sees images: P("almost entirely white") 0.98 for a white JPEG, 0.02 for a black one, 0.03 for mid-grey. Sizes: 48-256 px and 2048 px rejected (HTTP 400, "Invalid image dimensions."); 384-1100 px accepted. 6 of 6 image-arm requests on 518 px synthetic studies answered, 2,396 input tokens per request (mean). The benchmark's val and test images: 4,774 readable, 518 px on the shorter side, 848 px at most on the longer |
| 2026-10-26 | v3.1 pilot | 94ad9b9 | `jev_img_est_text_noul`, 40 val studies | none | n/a | n/a | The same 40 studies as v3.0's pilots, run by the user: 40 of 40 answered, 0 failed; 2,413 input tokens per study (mean), against 1,521 without the image; latency 0.24 s median, 0.50 s p90. Distinct scores per finding, median 23 against 19.5 |
| 2026-10-26 | J2 | 4cd45b7 | `jev_img_est_text_noul` | official test | 0.802 [0.787, 0.816] | 12 | **J2: no detectable gain from the X-ray.** Minus `jev_est_text_noul` -0.001 [-0.005, 0.005] (u0 the same). Against the val combiner -0.024 [-0.034, -0.015]: the X-ray doesn't close J1's gap. Every image arm answered 1,733 of 1,733 val and 3,041 of 3,041 test studies, 0 failed. No red flags |
| 2026-10-26 | v3.1 ladder | 4cd45b7 | `jev_img_noul`, `jev_img_est_noul` | official test | 0.716 [0.698, 0.733], 0.801 [0.786, 0.814] | 12 | **Jev's own reading of the X-ray:** 0.716, -0.099 [-0.112, -0.084] below RAD-DINO's fine-tune (0.814), above the 0.60 "image unread" line. Best on Pleural Effusion 0.856, Edema 0.836, Pneumothorax 0.831; Enlarged Cardiomediastinum 0.530 (near chance, as from text), Pneumonia 0.636, Lung Opacity 0.641. The X-ray added to the estimates: -0.003 [-0.010, 0.004]; it helps Fracture (+0.025) and Lung Lesion (+0.024) and costs Enlarged Cardiomediastinum (-0.029), Lung Opacity (-0.033) and Pneumonia (-0.023), which net to zero. About 33M input tokens for v3.1 (about 2,100-2,400 per request) |
| 2026-10-26 | v3.2 (exploratory) | 4d29571 | stacks A-E (§8b) | official test | A 0.824, B 0.809, C 0.829, D 0.821, E 0.828 | 12 | **Jev's text reading is a usable input, not a complementary one.** A (image + Jev's text answers, fit on val) minus the image run +0.010 [0.005, 0.014], against +0.012 for TF-IDF's text (A minus `stack_text` -0.003 [-0.009, 0.003]). On top of v2's best system: Jev's text +0.002 [-0.000, 0.004] (C), Jev's all-inputs answer +0.001 [-0.002, 0.005] (E), nothing detectable. Jev's X-ray reading hurts, against the image run -0.005 [-0.010, -0.002] (B) and against `stack_text` -0.006 [-0.009, -0.003] (D), although the combiner gives it little weight (0.07-0.11 against 1.56). Val scores of the nested stacks C-E (0.804-0.820) came out below `stack_text`'s own val score (0.844), not above it as §8b expected: pooling cross-fitted fold scores through a second combiner distorts them, so read C-E on the test only. No Jev requests |
| 2026-10-26 | v3.3 live smoke | 03cce7d | labeler, 6 synthetic reports | none | n/a | n/a | `choice` questions through the real API: 6 of 6 answered by `jev-1.14.0`, 1,842 input tokens per request |
| 2026-10-26 | v3.3 pilot | 03cce7d | labeler, 40 val reports | none | n/a | n/a | Run by the user: 40 of 40 answered, 0 failed, 0 empty reports; 1,891 input tokens per report (mean), latency 0.18 s median. Over the 40 x 12 answers Jev chose present 17.5%, uncertain 8.7%, absent 73.8%, against the CheXpert labeler's 13.3% positive, 4.0% uncertain, 82.7% negative or blank on the same studies: Jev calls more findings present or uncertain. No metric (§8c) |
| 2026-10-26 | L1 | 296c0a1 | `labeler_jev` | the radiologist's 596 official-test studies (268 patients) | 0.914 [0.904, 0.923] against the CheXpert labeler's 0.940 [0.924, 0.953] | 10 | **L1 FAIL:** minus `labeler_chexpert` -0.026 [-0.041, -0.010] (u0 -0.021 [-0.036, -0.005]): the CheXpert labeler is closer to the radiologist, so no relabeling follows. 596 of 596 answered by the user's run, 0 failed, 0 empty reports; 1,899 input tokens per report; Jev chose present 22.1%, uncertain 7.5%, absent 70.3%. No red flags. Per finding (test diagnostics, not choices): nearly all of the gap is Lung Opacity, 0.682 against 0.914 (about -0.023 of the -0.026); without it the two labelers' point means differ by about 0.003. Jev is ahead on Lung Lesion (0.965 against 0.921), Fracture (0.972 against 0.947) and Pneumonia (0.975 against 0.958), behind on Atelectasis, Cardiomegaly, Edema and Enlarged Cardiomediastinum (-0.02 to -0.03 each). Secondary, `labeler_jev_prob` (Jev's probabilities): 0.950 [0.941, 0.958], minus the CheXpert labeler +0.010 [-0.004, 0.027] (u0 +0.019 [0.005, 0.036]); not like for like, since a continuous score ranks with more resolution than the labeler's three levels. Jev against the CheXpert labeler's own labels: 0.866 [0.852, 0.879]. About 1.2M input tokens |
| 2026-10-26 | v3.4 (exploratory) | d5d0324 | `labeler_jev_cut` | the radiologist's 596 official-test studies | 0.931 [0.919, 0.942] | 10 | **1. Hierarchy: not applied.** On train the CheXpert labeler marks Lung Opacity positive in 0.332 of the 79,871 studies with a positive child, and Enlarged Cardiomediastinum in 0.037 of the 41,735 with Cardiomegaly, both far below 0.90. The radiologist's own rates (diagnostic) are 0.247 and 0.097: neither labeler uses the hierarchy, so it doesn't explain Lung Opacity. **2. Cut-offs:** minus the CheXpert labeler -0.009 [-0.025, 0.008] (u0 -0.003 [-0.018, 0.015]), no longer detectable; minus Jev's answer +0.017 [0.010, 0.024], about half of the +0.037 between Jev's answer and its probabilities. Lung Opacity 0.682 -> 0.819, still below the labeler's 0.914; the other findings move little. **The diagnostic shows the real problem:** Jev answers Lung Opacity present in 404 of the 596 studies, against the radiologist's 146 positives. It finds 138 of those 146 but also marks 266 radiologist negatives present. It over-calls: the question's definition ("any area of increased lung density ... whatever its cause") is broader than the radiologist's and the labeler's mention-based category. Exploratory throughout: prompted by v3.3's results on these studies |
| 2026-10-26 | L2 | 552d4c2 | `probe20k_jev_soft` | the radiologist's 596 official-test studies | 0.767 [0.743, 0.790] against `probe20k_chexpert` 0.757 [0.733, 0.779] | 10 | **L2: no detectable difference.** Minus `probe20k_chexpert` +0.009 [-0.007, 0.027] (u0 +0.010 [-0.006, 0.028]); Jev's answers as hard targets +0.010 [-0.006, 0.028]; soft and hard come out the same. For scale, ten times the data with CheXpert labels (`rad-dino_518_probe`, 213k studies) adds +0.024 [0.011, 0.037]. Against the CheXpert labeler's labels on the whole official test the order flips: the Jev-trained probes trail by -0.028 [-0.040, -0.017] and -0.028 [-0.042, -0.016], since a probe scores best against the labeler it learned from. Per finding against the radiologist (test diagnostics): Jev's labels help Consolidation (0.733 hard, 0.720 soft, against 0.662), Fracture (0.779, 0.793 against 0.750) and Cardiomegaly (+0.014 to +0.016); they cost Lung Lesion (-0.022, -0.030) and, as soft targets, Pneumonia (-0.034); Lung Opacity is unchanged despite Jev's over-calling. Labeling, run by the user: 20,000 training and 1,733 val reports, 0 failed, 0 empty; about 41M input tokens; Jev chose present 15.6%, uncertain 5.4%, absent 79.0% on train |
| 2026-10-26 | L3 | 07b395b | `probe20k_jev_soft` | CheXpert test (500 studies, 5 radiologists) | 0.892 [0.878, 0.907] against `probe20k_chexpert` 0.827 [0.811, 0.844] | 6 | **L3 PASS: +0.065 [0.054, 0.077]** (hard targets +0.061 [0.049, 0.073]); the refit reproduced v3.5's predictions exactly (largest difference 0.0). **But the gain is a definition effect, not a general one:** on the five standard findings it is +0.001 [-0.010, 0.012]; it sits in the two parent categories, Enlarged Cardiomediastinum (0.900 against 0.621) and Lung Opacity (0.928 against 0.854). Post-hoc diagnostic (aggregates, not used to choose anything): the CheXpert test set labels both hierarchically, P(parent | a child positive) 1.000 for each (prevalences 0.53 and 0.51); Jev, through the questions' broad definitions, nearly does too (0.878 Lung Opacity, 0.947 Enlarged Cardiomediastinum on the 20k training reports); the CheXpert labeler (0.330, 0.038) and the MIMIC radiologist (0.247, 0.097, v3.4) don't. That also explains v3.3's Lung Opacity "over-calling" and L2's small gain on MIMIC. v2's 213k-study probe is weak on Enlarged Cardiomediastinum here too (0.686), so more CheXpert-labeled data doesn't fix it. Per §8f, a gain on one judge only is a lead: here, that label definitions decide transfer to CheXpert. The control that would separate Jev from the definition, the hierarchy applied to the CheXpert labeler's own labels, has not been run |
