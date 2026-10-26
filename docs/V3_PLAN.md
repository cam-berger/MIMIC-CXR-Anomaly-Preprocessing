# v3 Plan: Jev as the combiner (text first, the image later)

**Status:** pre-registered 2026-10-26, before any Jev request on MIMIC data; val chose the yes/no format. **v3.0 done, J1 FAIL:** Jev with the estimates and all the text scores 0.803 on the official test, -0.024 [-0.033, -0.016] below v2's val combiner of the same inputs (0.827). Adding text to Jev adds nothing detectable (-0.002 [-0.010, 0.006]), re-reading the estimates costs -0.010, and text alone trails TF-IDF by -0.043 (§9). **v3.1 done, J2: no detectable gain** from showing Jev the X-ray (-0.001 [-0.005, 0.005]); on its own Jev reads X-rays at 0.716, against RAD-DINO's 0.814 (§9) · how to run: [`v3/README.md`](../v3/README.md) · **Branch:** `v3` (from `v2`) · **Started:** 2026-10-26

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
