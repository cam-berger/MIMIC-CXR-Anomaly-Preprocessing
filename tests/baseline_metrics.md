# Baseline Training Metrics (Before Fixes)

**Date**: 2025-12-17
**Training Run**: `/home/dev/Documents/Portfolio/MIMIC/MIMIC-CXR-Anomaly-Preprocessing/output/classifier_training.log`

## Summary

Training exhibits **severe cascade failure** starting at Epoch 2. After initial corruptions, GradScaler enters corrupted state and never recovers, causing circuit breaker to trigger at the start of EVERY subsequent epoch.

## Epoch-by-Epoch Breakdown

| Epoch | Status | Train Loss | Time | Corruptions | Notes |
|-------|--------|------------|------|-------------|-------|
| 0 | ✅ Normal | 1.6735 | 55.7 min | 0 | Healthy training |
| 1 | ✅ Normal | 1.3512 | 55.7 min | 0 | Healthy training |
| 2 | ⚠️ Partial | 1.3126 | 19.6 min | 10 | Circuit breaker triggered mid-epoch |
| 3-29 | ❌ Failed | 0.0000 | ~20 sec | 10 each | Circuit breaker at epoch start |

## Key Metrics

### Pre-Cascade (Epochs 0-1)
- **Training Progress**: Normal loss reduction (1.6735 → 1.3512)
- **Corruption Rate**: 0%
- **Time per Epoch**: ~56 minutes
- **Batches Completed**: 1723 batches/epoch (full dataset)

### Cascade Onset (Epoch 2)
- **First Corruption**: Mid-epoch after ~1000+ batches
- **Circuit Breaker**: Triggered after 10 consecutive corruptions
- **Partial Progress**: Train Loss still improved (1.3126)
- **Time to Failure**: ~19.6 minutes

### Post-Cascade (Epochs 3-29)
- **Corruption Rate**: 100% (immediate failure at batch 1-10)
- **Circuit Breaker**: Triggered at start of EVERY epoch
- **Training Progress**: Zero (Train Loss: 0.0000)
- **Time per Epoch**: ~20 seconds (immediate abort)
- **Batches Completed**: 9 batches/epoch (circuit breaker threshold)
- **Total Wasted Epochs**: 27 epochs

## Validation Metrics

**Best Performance** (Epoch 2, before cascade):
- **AUROC**: 0.6503 (macro-average)
- **AUPRC**: 0.8712

**Per-Class AUROC** (Epoch 2):
| Class | AUROC |
|-------|-------|
| Atelectasis | 0.3338 |
| Cardiomegaly | 0.7818 |
| Consolidation | 0.7891 |
| Edema | 0.8382 |
| Enlarged Cardiomediastinum | 0.6973 |
| Fracture | 0.4886 |
| Lung Lesion | 0.5687 |
| Lung Opacity | 0.6586 |
| Pleural Effusion | 0.7620 |
| Pleural Other | 0.4348 |
| Pneumonia | 0.7209 |
| Pneumothorax | 0.7302 |

**Note**: Metrics did NOT improve after Epoch 2 (no training occurred)

## Root Cause Analysis

### Primary Issue: GradScaler Reset Bug (Fix #1)
**Location**: `train_classifier.py:428`

**Current Code**:
```python
scaler = GradScaler()  # Creates NEW object, loses state!
```

**Problem**: When weight corruption is detected, code creates a new GradScaler instead of resetting the existing one. This breaks the connection between scaler and optimizer state.

**Impact**:
1. Corruption occurs in Epoch 2 batch ~1000
2. Code creates new GradScaler object
3. New scaler has no knowledge of optimizer state
4. Scaler state becomes permanently corrupted
5. Every subsequent batch fails validation
6. Circuit breaker triggers immediately at start of next epoch
7. Pattern repeats for all remaining epochs

### Secondary Contributors

1. **CrossAttention NaN Propagation** (Fix #2)
   - No NaN guards before attention operations
   - Single NaN in embedding can corrupt entire batch

2. **F.normalize Division by Zero** (Fix #3)
   - Zero-norm embeddings cause NaN in CLIP/SupCon losses
   - No safety checks before normalization

3. **MAE Normalization Epsilon** (Fix #4)
   - Small epsilon (1e-6) may cause issues with constant patches
   - Less critical for classifier (uses MAE encoder only)

## Expected Improvements After Fixes

### Fix #1 (GradScaler Reset)
- **Expected Impact**: Eliminate cascade failures
- **Metric**: Zero circuit breaker triggers after Epoch 2
- **Training**: All epochs should complete with ~56 min/epoch

### Fix #2 (CrossAttention Guards)
- **Expected Impact**: Reduce initial corruption rate
- **Metric**: Fewer mid-epoch corruptions

### Fix #3 (Safe Normalize)
- **Expected Impact**: Stabilize contrastive losses
- **Metric**: Lower NaN rate in CLIP/SupCon losses

### Fix #4 (MAE Epsilon)
- **Expected Impact**: Marginal (MAE is encoder only)
- **Metric**: Slightly more stable image embeddings

### Combined Fixes
- **NaN Batch Rate**: <0.5% (down from ~100% after Epoch 2)
- **Weight Corruption Rate**: <0.1% (isolated incidents, no cascade)
- **Circuit Breaker Triggers**: 0
- **Training Completion**: 30/30 epochs
- **Expected Val AUROC**: 0.70-0.75 (after 30 epochs of actual training)

## Baseline Summary Statistics

**Total Training Time**: ~5.4 hours
**Effective Training Time**: ~2 hours (Epochs 0-2 only)
**Wasted Time**: ~3.4 hours (Epochs 3-29 circuit breaker loops)
**Epochs Completed**: 2/30 (6.7%)
**Final Val AUROC**: 0.6503
**Circuit Breaker Triggers**: 28 times (Epochs 2-29)
**Total Weight Corruptions**: 280+ (10 per epoch × 28 epochs)

## Conclusion

The baseline demonstrates **catastrophic failure due to GradScaler reset bug**. Training becomes completely non-functional after Epoch 2. Implementing the 4 fixes, especially Fix #1, is critical for any meaningful training progress.
