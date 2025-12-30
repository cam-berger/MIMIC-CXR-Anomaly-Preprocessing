# Production Model Results

Comprehensive evaluation report for the MIMIC-CXR Multimodal Classifier trained on Lambda Cloud GH200 GPU (December 2024).

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Macro AUROC** | **0.701** |
| **Macro AUPRC** | **0.899** |
| Training Samples | 27,576 |
| Validation Samples | 4,922 |
| Epochs | 50 |
| Training Time | ~36 hours |
| Training Cost | ~$54 (GH200 @ $1.50/hr) |

**Key Findings:**
- Strong performance on Edema (0.878 AUROC), Consolidation (0.840), and Pneumonia (0.812)
- High AUPRC across most classes reflects the anomalous cohort's pathology prevalence
- Classes with extreme imbalance (>95% positive) show inflated AUPRC but low AUROC
- Model favors high recall over precision (clinical safety prioritized)

---

## Model Architecture

### Multimodal Fusion Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MAE Encoder   │    │   TextEncoder   │    │ StructuredEncoder│
│   (ViT-Base)    │    │ (ClinicalBERT)  │    │     (MLP)       │
│   [B, 768]      │    │   [B, 768]      │    │   [B, 256]      │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────┬───────────┘                      │
                    ▼                                  │
         ┌─────────────────┐                          │
         │ CrossAttention  │                          │
         │     Fusion      │                          │
         └────────┬────────┘                          │
                  │                                   │
                  └─────────────┬──────────────────────┘
                                ▼
                    ┌───────────────────┐
                    │  Classification   │
                    │   Head [B, 12]    │
                    └───────────────────┘
```

### Components

| Component | Architecture | Parameters |
|-----------|--------------|------------|
| Image Encoder | ViT-Base (MAE pretrained) | 86M |
| Text Encoder | ClinicalBERT (frozen) | 110M |
| Structured Encoder | 2-layer MLP | 0.5M |
| Cross-Attention | 8 heads, 768 dim | 4.7M |
| Classification Head | Linear + Sigmoid | 6K |

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| Asymmetric Focal | 1.0 | Multi-label classification with class imbalance handling |
| CLIP | 0.3 | Image-text contrastive alignment |
| SupCon | 0.3 | Supervised contrastive learning |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 1024 × 1024 |
| Batch Size | 16 |
| Learning Rate | 3e-5 |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Warmup Epochs | 2 |
| MAE Encoder | Frozen |
| Mixed Precision | FP16 (AMP) |
| Gradient Clipping | 1.0 |

---

## Dataset Statistics

### Validation Set Composition

| Class | Samples | Positive Rate | Imbalance |
|-------|---------|---------------|-----------|
| Lung_Opacity | 2,022 | 97.9% | Extreme |
| Atelectasis | 1,329 | 98.3% | Extreme |
| No_Finding | 1,777 | 75.3% | High |
| Cardiomegaly | 1,334 | 88.3% | High |
| Edema | 1,237 | 68.1% | Moderate |
| Pleural_Other | 987 | 61.4% | Moderate |
| Pneumonia | 730 | 23.6% | Low (rare) |
| Consolidation | 599 | 42.9% | Moderate |
| Lung_Lesion | 318 | 95.0% | Extreme |
| Enlarged_Cardiomediastinum | 271 | 60.5% | Moderate |
| Fracture | 230 | 91.3% | High |
| Pleural_Effusion | 71 | 97.2% | Extreme |

**Note:** The anomalous cohort is enriched for pathology, causing high positive rates. This differs from general population prevalence.

---

## Per-Class Performance

### Ranked by AUROC

| Rank | Class | AUROC | AUPRC | Precision | Recall | F1 |
|------|-------|-------|-------|-----------|--------|-----|
| 1 | **Edema** | **0.878** | 0.934 | 0.804 | 0.953 | 0.872 |
| 2 | **Consolidation** | **0.840** | 0.825 | 0.578 | 0.895 | 0.702 |
| 3 | No_Finding | 0.821 | 0.928 | 0.787 | 0.979 | 0.872 |
| 4 | Pneumonia | 0.812 | 0.672 | 0.429 | 0.756 | 0.547 |
| 5 | Cardiomegaly | 0.808 | 0.965 | 0.905 | 0.958 | 0.930 |
| 6 | Pleural_Other | 0.751 | 0.837 | 0.652 | 0.967 | 0.779 |
| 7 | Lung_Opacity | 0.717 | 0.990 | 0.980 | 0.995 | 0.987 |
| 8 | Enlarged_Cardiomediastinum | 0.708 | 0.793 | 0.608 | 0.976 | 0.749 |
| 9 | Fracture | 0.651 | 0.948 | 0.913 | 1.000 | 0.955 |
| 10 | Lung_Lesion | 0.580 | 0.960 | 0.958 | 0.983 | 0.971 |
| 11 | Atelectasis | 0.524 | 0.984 | 0.983 | 0.998 | 0.991 |
| 12 | Pleural_Effusion | 0.326 | 0.953 | 0.972 | 1.000 | 0.986 |

### Performance Tiers

**Excellent (AUROC > 0.8):**
- Edema, Consolidation, No_Finding, Pneumonia, Cardiomegaly

**Good (AUROC 0.7-0.8):**
- Pleural_Other, Lung_Opacity, Enlarged_Cardiomediastinum

**Moderate (AUROC 0.6-0.7):**
- Fracture, Lung_Lesion

**Poor (AUROC < 0.6):**
- Atelectasis, Pleural_Effusion (extreme class imbalance)

---

## Confusion Matrix Analysis

### High-Performing Classes

**Edema** (AUROC: 0.878)
| | Predicted - | Predicted + |
|---|-------------|-------------|
| **Actual -** | 198 (TN) | 196 (FP) |
| **Actual +** | 40 (FN) | 803 (TP) |

- Specificity: 50.3%
- Strong discriminative ability despite moderate positive rate (68%)

**Cardiomegaly** (AUROC: 0.808)
| | Predicted - | Predicted + |
|---|-------------|-------------|
| **Actual -** | 37 (TN) | 119 (FP) |
| **Actual +** | 50 (FN) | 1,128 (TP) |

- Specificity: 23.7%
- High recall (96%) ensures few missed cases

### Challenging Classes

**Pleural_Effusion** (AUROC: 0.326)
| | Predicted - | Predicted + |
|---|-------------|-------------|
| **Actual -** | 0 (TN) | 2 (FP) |
| **Actual +** | 0 (FN) | 69 (TP) |

- Only 71 samples, 97% positive rate
- Model predicts positive for all samples (no discrimination)
- AUROC < 0.5 indicates worse than random for this class

**Atelectasis** (AUROC: 0.524)
| | Predicted - | Predicted + |
|---|-------------|-------------|
| **Actual -** | 0 (TN) | 23 (FP) |
| **Actual +** | 2 (FN) | 1,304 (TP) |

- 98% positive rate makes negative class nearly invisible
- Model learns to always predict positive

---

## Threshold Analysis

Optimal thresholds vary by class based on precision-recall tradeoffs:

### Default Threshold (0.5)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Edema | 0.804 | 0.953 | 0.872 |
| Consolidation | 0.578 | 0.895 | 0.702 |
| Cardiomegaly | 0.905 | 0.958 | 0.930 |
| Pneumonia | 0.429 | 0.756 | 0.547 |

### High-Precision Threshold (0.7)

For clinical settings requiring fewer false positives:

| Class | Precision | Recall | F1 | Change |
|-------|-----------|--------|-----|--------|
| Edema | 0.886 | 0.835 | 0.860 | +8% precision, -12% recall |
| Consolidation | 0.769 | 0.700 | 0.733 | +19% precision, -19% recall |
| Cardiomegaly | 0.935 | 0.862 | 0.897 | +3% precision, -10% recall |
| Pneumonia | 0.689 | 0.488 | 0.571 | +26% precision, -27% recall |

### High-Recall Threshold (0.3)

For screening applications (minimize missed cases):

| Class | Precision | Recall | F1 | Change |
|-------|-----------|--------|-----|--------|
| Edema | 0.707 | 0.996 | 0.827 | -10% precision, +4% recall |
| Consolidation | 0.439 | 0.988 | 0.608 | -14% precision, +9% recall |
| Cardiomegaly | 0.889 | 0.998 | 0.940 | -2% precision, +4% recall |
| Pneumonia | 0.261 | 0.994 | 0.414 | -17% precision, +24% recall |

---

## Key Observations

### 1. Class Imbalance Effects

Classes with >95% positive rate (Atelectasis, Lung_Opacity, Pleural_Effusion, Fracture, Lung_Lesion) show:
- **High AUPRC** (0.95+) due to baseline positive rate
- **Low AUROC** (<0.7) due to insufficient negative samples for discrimination
- **Near-perfect recall** (model always predicts positive)
- **Zero specificity** (no true negatives)

**Interpretation:** High AUPRC for these classes reflects the dataset's pathology prevalence, not model discrimination. AUROC is the more reliable metric.

### 2. Model Behavior

The model prioritizes **high recall** over precision:
- 10 of 12 classes have recall > 0.9
- Only 4 classes have precision > 0.9
- This is appropriate for clinical screening (minimize missed diagnoses)

### 3. Multimodal Contribution

Cross-attention fusion between image and text features enables:
- Contextual understanding from clinical notes
- Disambiguation of ambiguous imaging findings
- Integration of patient history and demographics

### 4. Stability Validation

Production training validated all NaN/Inf stability fixes:
- 50/50 epochs completed (zero cascade failures)
- <0.5% NaN batch rate (properly handled)
- Zero weight corruptions
- Circuit breaker never triggered

---

## Visualizations

Research artifacts generated and available in `output/research_artifacts/`:

| File | Description |
|------|-------------|
| `confusion_matrices.png` | 4×3 grid of per-class confusion matrices |
| `roc_curves.png` | ROC curves for all 12 classes |
| `pr_curves.png` | Precision-Recall curves for all 12 classes |
| `auroc_by_class.png` | Bar chart of AUROC scores |
| `auprc_by_class.png` | Bar chart of AUPRC scores |
| `auroc_vs_positive_rate.png` | Scatter plot showing imbalance effects |
| `f1_vs_threshold.png` | F1 score across threshold values |
| `specificity_vs_recall.png` | Trade-off visualization |
| `label_cooccurrence_conditional.png` | Multi-label correlation heatmap |
| `score_dist_Edema.png` | Score distribution for Edema |
| `score_dist_Pleural_Effusion.png` | Score distribution for Pleural_Effusion |

---

## Recommendations

### Short-Term Improvements

1. **Class-Specific Thresholds**: Deploy different thresholds per class based on clinical requirements
2. **Focal Loss Tuning**: Increase `gamma_neg` for rare classes (Pneumonia, Consolidation)
3. **Oversampling**: Balance training set for underrepresented negative cases

### Medium-Term Improvements

1. **MAE Fine-Tuning**: Gradually unfreeze encoder layers for task-specific adaptation
2. **Ensemble Methods**: Combine multiple models with different seeds
3. **Test-Time Augmentation**: Multiple crops and flips for robust inference

### Long-Term Research Directions

1. **External Validation**: Test on CheXpert, NIH ChestX-ray14, PadChest
2. **Longitudinal Analysis**: Track disease progression across sequential studies
3. **RAG Integration**: Retrieve similar cases for explainable predictions
4. **Uncertainty Quantification**: Calibrated confidence scores for clinical deployment

---

## Reproducibility

### Training Command

```bash
python train_classifier.py \
    --config base \
    --train-dir output/preprocessed/anomalous_train \
    --val-dir output/preprocessed/anomalous_val \
    --chexpert-csv /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
    --mae-checkpoint output/models/mae_final.pt \
    --epochs 50 \
    --batch-size 16 \
    --img-size 1024
```

### Model Checkpoints

| File | Description | Size |
|------|-------------|------|
| `classifier_best.pt` | Best validation AUROC | 942 MB |
| `classifier_final.pt` | Final epoch checkpoint | 890 MB |
| `mae_final.pt` | MAE pretrained encoder | 447 MB |

---

## References

- [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/)
- [CheXpert Labeler](https://arxiv.org/abs/1901.07031)
- [MAE: Masked Autoencoders](https://arxiv.org/abs/2111.06377)
- [Asymmetric Loss for Multi-Label Classification](https://arxiv.org/abs/2009.14119)

---

*Report generated: December 2024*
*Training infrastructure: Lambda Cloud GH200 (480GB)*
*Total compute cost: ~$54*
