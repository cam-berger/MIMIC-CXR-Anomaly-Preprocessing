# v3 Plan: Jev as the combiner (text first, the image later)

**Status:** pre-registered 2026-10-26, before any Jev request on MIMIC data · how to run: [`v3/README.md`](../v3/README.md) · **Branch:** `v3` (from `v2`) · **Started:** 2026-10-26

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
  is the study's context as text (`v3/prompt.py`), and there are 12 yes/no
  (`noul`) questions, one per finding. Each answer's `noul`, the probability of
  yes, is that finding's score.
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
| `jev_img_est_text` | + the frontal image | v3.1: request support is built, but not run in v3.0 (§8) |

A missing source is written as "none recorded", so every arm scores every study.

**Baselines** (v2 runs on the same studies):

- `rad-dino_518_ft_attn_s0`: stage 1 on its own.
- `rad-dino_518_ft_attn_s0_stack_indication`: the val combiner with the indication.
- `rad-dino_518_ft_attn_s0_stack_text`: the val combiner with all three text sources, built by `v3.setup` with `v2.stack`. This is the like-for-like baseline for `jev_est_text`.
- `rad-dino_518_ctx_text`: v2's text-only arm (TF-IDF).

## 5. Gate J1 (fixed now)

**J1:** `jev_est_text` minus `rad-dino_518_ft_attn_s0_stack_text`, official
test, v2 policy, paired 95% CI from 1,000 patient resamples.

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
3. **Pilot:** 40 val studies, `jev_est_text`. This measures latency, tokens per
   request and the failure rate. Only aggregates are printed.
4. **Val, all four arms** (4 × 1,733 requests). Nothing is tuned on val. If
   `jev_est_text` falls more than 0.03 below the val stack on val, the prompt is
   revised and val re-run, at most twice. Each revision is logged in §9 with its
   val score.
5. **Official test, all four arms** (4 × 3,041 requests), once, with the final
   prompt.

About 19,000 requests in all. Answers are cached, so a crash or a re-run never
pays twice for the same (arm, study, prompt, model).

## 8. v3.1: adding the image (built, not run)

The request builder already supports an `image` block. It sends the study's
prepared frontal JPEG (`BENCH_OUT/images/{study_id}.jpg`, the manifest's `image`
column) as a base64 asset, with the `TypeSafe-Danger-Zone-Images: true` header
and a placeholder in `state`. `jev_img_est_text` refuses to run without
`--allow-image`. Before it runs, a short amendment here fixes three things: the
model (it must be a vision-language Jev), the image size, and the gate
(`jev_img_est_text` − `jev_est_text`).

## 9. Results log

| Date | Step | Commit | Run | Test set | Macro AUROC [95% CI] | Classes | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | setup | d1bfba8 | `rad-dino_518_ft_attn_s0_stack_text` | official test | 0.827 [0.811, 0.840] | 12 | The J1 baseline, built by `v3.setup` (val combiner, image + all three text sources). Minus the image run +0.012 [0.005, 0.019]. The rebuilt `stack_indication` matches the benchmark's copy exactly (largest difference 0.0), 0.824 [0.809, 0.837] |
| 2026-10-26 | live smoke | d1bfba8 | `jev_est_text`, 12 synthetic studies | none | n/a | n/a | Request format checked on synthetic text: 12 of 12 answered. `GET /v1/models` lists only the aliases `jev-latest` and `jev-preview`; `jev-1.14.0` (named in LiteLLM's docs) is accepted, and every answer reports it, so it is the pinned model (§3) |
| 2026-10-26 | pilot | d1bfba8 | `jev_est_text`, 40 val studies | none | n/a | n/a | 40 of 40 answered, 0 failed; 1,521 input tokens per study (mean); latency 0.21 s median, 0.27 s p90, at 4 in flight. **Jev's probabilities come on a 0.01 grid** (480 of 480 answers): 69 distinct values overall, 8-30 per finding over 40 studies, fewest for Fracture (8) and Pleural Other (9). Tied ranks cap AUROC on rare findings |
