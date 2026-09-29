# Next Iteration Plan

Written 2026-09-29, after the full code review and the fixes in
[CHANGELOG.md](CHANGELOG.md) ("Unreleased"). This document covers what the
review means for the December 2024 result, and what to do next.

## Summary

1. **Don't resume from the December checkpoint.** Its fusion, text and
   structured-data weights were trained through bugs that are now fixed, so
   they don't match the corrected pipeline.
2. **Treat macro AUROC 0.701 as a weak, optimistic baseline.** It is not a
   measure of what this architecture can do (section 1).
3. **Fix the evaluation protocol before training anything** (section 3).
   Without that, no comparison between models is interpretable.
4. **Make an image-only model with a pretrained backbone the main line**
   (section 4, E1). Keep MAE pretraining as a controlled comparison arm (E2),
   not as the default encoder.
5. **Add structured data, then text, one at a time, with ablations**
   (E3, E4). Leak-free text now carries little beyond the structured data.

## 1. What the review means for the December 2024 result

| Component | What the December model actually got |
|-----------|--------------------------------------|
| Image | MAE encoder **frozen** for all 50 epochs (RESULTS.md). The MAE was pretrained on ~20k normal studies, and training took a 1024×1024 **center crop of a ~3056×2544 image: ~13% of the radiograph**, without the costophrenic angles, apices and lateral fields. |
| Text | Token ids from the Bio_ClinicalBERT tokenizer fed to PubMedBERT: every id mapped to an unrelated wordpiece. The text branch read noise. |
| Leakage (latent) | The "leak-free" text contained ED and hospital **discharge ICD codes** (e.g. 486 = pneumonia), procedures and disposition. The scrambled tokens happened to neutralize this. With the tokenizer fixed, the same text would have leaked labels (both fixed now). |
| Fusion | Cross-attention over single pooled vectors is a linear map; ~2.4M of its 5.9M parameters never trained. |
| Structured | Raw values: NT-proBNP (up to 70,000) overflowed fp16, so batches were skipped, and it dominated the embedding. "Procalcitonin" was total protein. Censored results were missing. Labs up to 24 h *after* the study were included. |
| Losses | `text_clip_proj` was never in the optimizer (a random projection) and the CLIP temperature never trained. The CLIP loss aligns the *fused image+text* embedding with the text embedding, so it can be solved through the text path (still true; see E4). |
| Evaluation | A random 85/15 subject-level split, with **no test set**; the best epoch was selected on the same validation set that is reported. Blank CheXpert labels are masked rather than treated as negative, which, in the abnormal-only cohort, gives extreme positive rates. |

The per-class numbers show the evaluation problem directly (RESULTS.md):

- **Pleural Effusion: AUROC 0.326 from 71 labeled validation studies with 2
  negatives.** An AUROC computed from 2 negatives is noise, not a finding.
  (It is still suggestive that the class most dependent on the costophrenic
  angles did worst under a center crop that removes them.)
- Atelectasis has 23 negatives and Lung Opacity about 42. Five of the 12
  classes are at least 91% positive in validation.
- **Macro AUPRC 0.899 mostly reflects prevalence.** AUPRC's chance level
  equals the positive rate, so 0.95+ is chance for these classes.
- The per-class table lists `No_Finding`, which the 12-label classifier does
  not predict. Confirm which model and evaluation script produced that table.

**Reading of 0.701:** CLS features from a frozen MAE looking at the central
13% of the image, plus an MLP on raw labs, scored with an optimistic
protocol. It is not evidence for or against multimodal fusion or MAE
pretraining.

## 2. Should you fine-tune a pretrained vision model?

Yes, as the main line, and as a controlled comparison rather than a swap:

- The image is where the signal is, and the current encoder was never
  fine-tuned. The first thing to establish is how far a well-trained
  image-only model gets under a sound protocol.
- MAE pretraining on ~20k normal-only images is small by MAE standards. MAE
  recipes use orders of magnitude more images, and nothing here showed it
  beats a public CXR or general-purpose encoder.
- "Does in-domain MAE pretraining beat off-the-shelf encoders at equal
  fine-tuning budget?" is a clean, publishable question. You can only answer
  it with E1 as the reference, so keep the MAE as an arm (E2).

## 3. Step 0: evaluation protocol (before any model comparison)

1. **Split: use the official MIMIC-CXR split** (`mimic-cxr-2.0.0-split.csv.gz`,
   patient-level train/validate/test; loader: `MIMICCXRLoader.split`). The
   cohort builder currently ignores it. Select models on validation, and
   evaluate on test once, at the end.
2. **Population:** decide what question the model answers.
   - *All frontal ED studies* (normal + abnormal): the standard setting,
     comparable with the literature, and with enough negatives for every
     class. **Recommended as primary.**
   - *Abnormal-only* (current): a different, less standard task in which
     AUROC is driven by a handful of negatives. Keep it as a secondary
     analysis if it matters to your research question.
3. **Label policy:** treat blank CheXpert labels as negative (standard
   practice) and mask uncertain (-1) labels. Fix the policy before looking at
   results, and report a sensitivity analysis (e.g. uncertain → positive for
   the classes where the CheXpert paper found that better).
4. **Metrics:**
   - Per-class AUROC with patient-level bootstrap 95% CIs.
   - Macro AUROC over classes with at least ~30 positives *and* 30 negatives
     in the test set.
   - AUPRC reported next to prevalence.
   - For model comparisons, a paired bootstrap (or DeLong) on the same test
     patients, and ≥3 seeds for the final arms.
   - Always report per-class positive/negative counts.
5. **Leakage audit:** every feature must be known at image acquisition, and
   independent of the image's findings.
   - Already fixed: discharge diagnoses, procedures, disposition, post-study labs.
   - Still open: ED vitals are aggregated over the whole stay
     (`MIMICIVEDLoader.get_vitals_summary`), which includes vitals after the
     X-ray. Filter by `charttime <= study time`.
6. **Re-baseline the December checkpoint** under the new protocol with the
   code at commit `8166273` (its architecture no longer loads in the fixed
   code). This gives E1 a reference point on the same test set.

## 4. Experiments

Each experiment answers one question under the protocol above. Run them in
order: each later one depends on the answer before it.

### E1: image-only, pretrained backbones (primary baseline)

- Frontal image, full-view resize (`--image-mode resize`), 512px (518 for
  ViT/14), mixed precision.
- For each backbone:
  - a linear probe (frozen features; measures representation quality);
  - a full fine-tune with layer-wise LR decay, reusing `src/training`
    (`backward_and_step`, per-group warmup/unfreeze schedule).
- Loss: BCE, with the asymmetric focal loss as a variant.

| Backbone | Why | Caveat |
|----------|-----|--------|
| DenseNet-121 or ConvNeXt-T, ImageNet init | Classic CXR baseline, cheap | None |
| DINOv2 ViT-B/14 (`facebook/dinov2-base`) | Strong general SSL features; natural-image pretraining (no MIMIC-CXR) | Domain gap |
| RAD-DINO (`microsoft/rad-dino`) | DINOv2 continued on 883k CXRs; likely strongest | Its pretraining included **368,960 MIMIC-CXR images** (it excluded MAIRA's validation/test images). Check your test images against its published `training_images.csv`; self-supervised exposure to test images inflates results. |
| torchxrayvision DenseNets | Supervised CXR weights | Several variants were trained on MIMIC-CXR *labels*. Use a variant without MIMIC. |

The best E1 model is the bar every later experiment has to clear with a CI
that doesn't overlap it (or a significant paired test).

### E2: does in-domain MAE pretraining help?

Same fine-tuning budget as E1:

1. The existing MAE checkpoint, fine-tuned in its native `center_crop` mode
   (fair to it) and in `resize` mode (shows the field-of-view mismatch).
2. MAE re-pretrained on full-view images from **all train-split images**,
   normal and abnormal (labels unused, so no leakage). Tune the LR:
   `base_lr=1.5e-4` is used as-is, while the MAE paper defines it per 256
   images, so at batch size 4 it is 64× the linearly scaled value.
3. MAE initialized from ImageNet MAE weights (`facebook/vit-mae-base`) with
   continued pretraining on CXRs. This is usually cheaper and stronger than
   training from scratch.

If none of these beats the best E1 backbone, drop the MAE from the main line.
That is a legitimate negative result.

### E3: + structured data

- Late fusion: best E1 image embedding + normalized structured embedding
  (the fixed `StructuredEncoder` with missingness indicators) → head.
  Compare paired against image-only.
- Variants: triage vitals only; triage + pre-study labs.
- Data fix first: "troponin" mixes Troponin I and Troponin T (different
  assays and scales). Split them or keep one.

### E4: + text

- Leak-free text is now demographics, chief complaint, triage vitals and
  labs. Only the chief complaint is not already in the structured data.
  Test the chief complaint alone (short, cheap, maybe as categories) before
  a BERT branch.
- CLIP objective: it currently aligns the fused (image+text) embedding with
  text, which the model can satisfy through the text path. If you want
  image-text alignment, align the *image-only* embedding with the text.
  Otherwise drop CLIP and SupCon unless an ablation shows they help.

### E5: fusion architecture (only if E3/E4 show that the other modalities add signal)

Compare late fusion (concat), the corrected token-level cross-attention, and
FiLM-style conditioning. Keep the simplest unless the CIs separate.

## 5. Engineering to do first

- **Downsample images at preprocessing** (e.g. 1024 on the long side). The
  review measured ~190-200 ms per sample to decode a full-resolution image,
  versus a few ms for a small crop. With the documented 4 workers, the
  December run was likely limited by data loading.
- **Use the official split** in the cohort builder (section 3).
- **Filter ED vitals to before the study** (section 3.5).
- **Exclude frozen modules from checkpoints.** Each classifier checkpoint
  carries ~440 MB of frozen BERT weights.
- **Fit the anomaly detector on non-augmented images.**
  `train_mae.fit_anomaly_detector` uses the augmented, shuffled training
  loader.
- **GPU smoke test before any long run.** The mixed-precision path of the new
  training loop has only been exercised on CPU (the review environment had no
  GPU). Run `--config debug --epochs 1` on a small cohort and check the
  per-epoch counters:
  - `skipped_grad`: a few at the start (GradScaler calibrating), then 0;
  - `skipped_loss`: about 0.

## 6. Decisions for you

1. **Population:** all frontal ED studies (recommended) or abnormal-only.
2. **Label policy** for blank and uncertain labels.
3. **Is MAE pretraining a research claim?** If yes, E2 needs equal compute and
   E1 as the reference. If it is only a means to an end, take the best E1
   backbone and move on.
4. **Is text still in scope?** With leakage removed, it overlaps heavily
   with the structured data.

## 7. Known issues not fixed in this iteration

| Issue | Where | Why not fixed |
|-------|-------|---------------|
| CLIP loss aligns fused (image+text) with text | `MultimodalClassifier.forward`, `MultiTaskLoss` | Design decision (E4) |
| ED vitals include post-study measurements | `MIMICIVEDLoader.get_vitals_summary` | Needs a time join; decide with the leakage audit |
| Blank labels masked; abnormal-only cohort | `classification_dataset.py`, `cohort/builder.py` | Protocol decision (section 3) |
| Random split, no test set | `cohort/builder.py:split_cohort` | Protocol decision (section 3) |
| MAE LR not scaled with batch size | `MAEConfig.base_lr` | Tuning (E2) |
| Troponin I and T mixed | `MIMICIVLoader.PRIORITY_LAB_IDS` | Data decision (E3) |
| Anomaly detector fit on augmented data | `train_mae.fit_anomaly_detector` | Outside the classifier scope |
| Full-resolution decode per sample; ~15-20 host syncs per step | datasets, `losses.py` | Efficiency; do with downsampling (section 5) |
| Frozen BERT saved in every checkpoint | `train_classifier.save_checkpoint` | Format change (section 5) |
