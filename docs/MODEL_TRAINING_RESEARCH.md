# Model Training Research: MAE, Fine-Tuning, and Anomaly Detection

This document explores implementation approaches for training anomaly detection models on chest X-rays using the preprocessed MIMIC-CXR data.

## Table of Contents
- [Overview: The Training Pipeline](#overview-the-training-pipeline)
- [Phase 1: Self-Supervised Pretraining](#phase-1-self-supervised-pretraining)
- [Phase 2: Fine-Tuning for Classification](#phase-2-fine-tuning-for-classification)
- [Phase 3: Anomaly Detection](#phase-3-anomaly-detection)
- [Multimodal Integration](#multimodal-integration)
- [Implementation Recommendations](#implementation-recommendations)

---

## Overview: The Training Pipeline

Our approach follows a three-phase training strategy:

```
Phase 1: Self-Supervised Pretraining (Normal cohort ~33k)
    │
    │   Train on unlabeled normal X-rays
    │   Model learns "what normal looks like"
    │
    ▼
Phase 2: Fine-Tuning (Anomalous cohort ~200k)
    │
    │   Add classification head
    │   Train on labeled abnormal examples
    │
    ▼
Phase 3: Anomaly Detection (Inference)
    │
    │   Detect deviations from "normal"
    │   Flag potential abnormalities
    │
    ▼
Output: Anomaly scores + classifications
```

---

## Phase 1: Self-Supervised Pretraining

Self-supervised learning allows models to learn meaningful representations from unlabeled data. For medical imaging, this is crucial because labeled data is scarce and expensive to obtain.

### Option 1: Masked Autoencoder (MAE)

**How it works:**
1. Randomly mask 75-90% of image patches
2. Encode only the visible patches with a Vision Transformer
3. Decode to reconstruct the full image
4. Loss = MSE between reconstruction and original

**Architecture:**
```
Input Image (224×224) → Patch Embedding (14×14 patches)
    │
    ▼
Random Masking (keep 10-25% of patches)
    │
    ▼
Encoder (ViT) [only processes visible patches]
    │
    ▼
Decoder (smaller ViT) [reconstructs all patches]
    │
    ▼
Reconstructed Image → MSE Loss with original
```

**Medical Imaging Adaptations:**
- [Research shows](https://arxiv.org/abs/2210.12843) medical images require **smaller visible ratios (10-25%)** compared to natural images (25%)
- Use **moderate crop ranges (0.5-1.0)** vs aggressive crops for natural images
- Higher resolution inputs preserve diagnostic details

| Aspect | Pros | Cons |
|--------|------|------|
| **Data efficiency** | Works well with limited data; comparable to 30% labeled data | Requires careful hyperparameter tuning |
| **Scalability** | Efficient training (only encodes visible patches) | Long pretraining required (800-1600 epochs typical) |
| **Transfer** | Excellent transfer to downstream tasks | May not capture fine-grained anomalies |
| **Medical imaging** | [SOTA on CheXpert, MIMIC-CXR, ChestX-ray14](https://github.com/lambert-x/medical_mae) | Optimal mask ratio varies by dataset |

**Key hyperparameters:**
```python
# Recommended settings for chest X-rays (from medical_mae)
mask_ratio = 0.75  # 75% masked (vs 90% for natural images)
decoder_depth = 2  # Shallow decoder sufficient
encoder = "vit_base_patch16"  # ViT-B/16
input_size = 224  # Or 384 for higher resolution
epochs = 800  # Longer pretraining helps
```

**Reference implementations:**
- [medical_mae](https://github.com/lambert-x/medical_mae) - WACV 2023
- [AMAE](https://arxiv.org/abs/2307.12721) - MICCAI 2023 (anomaly-specific)

---

### Option 2: Contrastive Learning (SimCLR/MoCo)

**How it works:**
1. Create two augmented views of each image
2. Encode both views to embeddings
3. Pull same-image embeddings together
4. Push different-image embeddings apart

**Architecture:**
```
Image → Augmentation 1 ──┐
                        ├──→ Encoder → Projection Head → z1
Image → Augmentation 2 ──┘                              → z2

Loss: InfoNCE = -log(exp(sim(z1,z2)/τ) / Σ exp(sim(z1,z_neg)/τ))
```

**Methods Comparison:**

| Method | Key Innovation | Batch Size Requirement | Memory |
|--------|---------------|----------------------|--------|
| **SimCLR** | Simple framework, strong augmentations | Very large (4096+) | High |
| **MoCo v2** | Momentum encoder, queue of negatives | Small (256) | Medium |
| **BYOL** | No negative samples needed | Medium (512) | Medium |
| **SwAV** | Clustering-based, multi-crop | Medium (256) | Medium |

| Aspect | Pros | Cons |
|--------|------|------|
| **Representations** | Learns semantically meaningful features | May focus on global features, miss local anomalies |
| **Augmentations** | Domain knowledge can improve pairs | Medical augmentations need care (no extreme color changes) |
| **Adoption** | [Most popular in medical imaging](https://www.nature.com/articles/s41746-023-00811-0) (44/79 studies) | Large batch sizes expensive |
| **Downstream** | Good for classification | Less suited for pixel-level reconstruction |

**Medical-specific augmentations:**
```python
# Safe augmentations for chest X-rays
transforms = [
    RandomResizedCrop(224, scale=(0.5, 1.0)),
    RandomHorizontalFlip(p=0.5),  # Anatomically valid
    RandomRotation(degrees=15),
    GaussianBlur(kernel_size=23),
    # Avoid: extreme color jitter, vertical flip (anatomically invalid)
]
```

**Reference implementations:**
- [MICLe](https://research.google/blog/self-supervised-learning-advances-medical-image-classification/) - Google Research
- [MoCo v2 for medical](https://github.com/facebookresearch/moco)

---

### Option 3: Hybrid Approach (Recommended)

[Recent research](https://www.nature.com/articles/s41598-023-46433-0) shows combining MAE and contrastive learning can outperform either alone:

```
Stage 1: MAE Pretraining
    │   Learn to reconstruct normal images
    │   Captures local structure
    │
    ▼
Stage 2: Contrastive Fine-tuning
    │   Learn discriminative features
    │   Captures semantic similarities
    │
    ▼
Encoder with both reconstruction and discriminative capabilities
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Best of both** | Local (MAE) + global (contrastive) features | Longer training time |
| **Flexibility** | Can use encoder for reconstruction or classification | More complex pipeline |
| **Performance** | [Shown to outperform single methods](https://www.nature.com/articles/s41598-023-46433-0) | Hyperparameter search harder |

---

## Phase 2: Fine-Tuning for Classification

After pretraining, we add a classification head and fine-tune on the anomalous cohort.

### Fine-Tuning Strategies

#### Strategy 1: Linear Probing
Freeze encoder, train only classification head.

```python
# Freeze all encoder layers
for param in encoder.parameters():
    param.requires_grad = False

# Add and train classification head
classifier = nn.Linear(embed_dim, num_classes)
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Speed** | Fast training | Limited adaptation |
| **Data** | Works with small datasets | May underperform |
| **Use case** | Quick evaluation of pretrained model | Not final solution |

#### Strategy 2: Full Fine-Tuning
Unfreeze all layers, train end-to-end.

```python
# All parameters trainable
for param in model.parameters():
    param.requires_grad = True

# Use smaller learning rate for pretrained layers
optimizer = AdamW([
    {'params': encoder.parameters(), 'lr': 1e-5},
    {'params': classifier.parameters(), 'lr': 1e-4}
])
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Performance** | Best downstream performance | Requires more data |
| **Adaptation** | Fully adapts to new domain | Risk of catastrophic forgetting |
| **Compute** | Standard | More epochs needed |

#### Strategy 3: Layer-wise Learning Rate Decay (LLRD)
Different learning rates for different depths.

```python
# Higher LR for later layers, lower for earlier
def get_layer_lr(layer_idx, base_lr=1e-4, decay=0.9):
    return base_lr * (decay ** (num_layers - layer_idx))

# Example: ViT-B with 12 layers
# Layer 0:  1e-4 * 0.9^12 = 2.8e-5
# Layer 6:  1e-4 * 0.9^6  = 5.3e-5
# Layer 11: 1e-4 * 0.9^1  = 9e-5
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Preservation** | Preserves low-level features | More hyperparameters |
| **Adaptation** | Adapts high-level features | Requires tuning decay rate |
| **Evidence** | [Shown effective for ViTs](https://link.springer.com/article/10.1007/s10278-022-00666-z) | Implementation complexity |

#### Strategy 4: Gradual Unfreezing
Progressively unfreeze layers during training.

```python
# Epoch 1-3: Only classifier
# Epoch 4-6: Last 3 encoder layers
# Epoch 7+: All layers
def unfreeze_schedule(epoch, model):
    if epoch < 3:
        freeze_until = len(model.encoder.layers)
    elif epoch < 6:
        freeze_until = len(model.encoder.layers) - 3
    else:
        freeze_until = 0

    for i, layer in enumerate(model.encoder.layers):
        layer.requires_grad = (i >= freeze_until)
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Stability** | Stable training, less forgetting | Longer training |
| **Features** | Preserves pretrained features well | Requires schedule tuning |
| **Performance** | Often best results | More complex |

### Multi-Label Classification

Chest X-rays often have multiple findings. Use appropriate loss:

```python
# Binary Cross-Entropy for multi-label
criterion = nn.BCEWithLogitsLoss()

# Or weighted BCE for class imbalance
pos_weight = (num_negatives / num_positives)  # Per-class
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Focal loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

---

## Phase 3: Anomaly Detection

After training, use the model to detect anomalies in new images.

### Method 1: Reconstruction-Based (MAE)

**Approach:** Anomalies should have higher reconstruction error.

```python
def compute_anomaly_score(model, image):
    # Mask and reconstruct
    reconstruction = model.reconstruct(image, mask_ratio=0.75)

    # Compute per-pixel error
    error = (image - reconstruction) ** 2

    # Aggregate to single score
    score = error.mean()  # or error.max(), or weighted
    return score
```

**Threshold Methods:**

| Method | Formula | Use Case |
|--------|---------|----------|
| **Percentile** | threshold = np.percentile(train_errors, 95) | Simple, robust |
| **Max training error** | threshold = np.max(train_errors) | Conservative |
| **Statistical** | threshold = mean + k * std | Assumes normal distribution |
| **IQR-based** | threshold = Q3 + 1.5 * IQR | Robust to outliers |
| **Learned** | Optimize on validation set | Best if labels available |

```python
# Percentile-based threshold
def set_threshold(model, normal_images, percentile=99):
    errors = [compute_anomaly_score(model, img) for img in normal_images]
    threshold = np.percentile(errors, percentile)
    return threshold

# Statistical threshold
def set_statistical_threshold(errors, k=3):
    mean = np.mean(errors)
    std = np.std(errors)
    return mean + k * std
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Interpretability** | Can visualize where reconstruction fails | Threshold selection tricky |
| **Localization** | Natural anomaly heatmaps | May miss subtle anomalies |
| **No labels needed** | Truly unsupervised | Higher false positive rate |

### Method 2: Embedding Distance

**Approach:** Anomalies should be far from normal embeddings.

```python
def compute_embedding_score(model, image, normal_embeddings):
    # Get embedding for new image
    embedding = model.encode(image)

    # Distance to nearest normal neighbor
    distances = cdist([embedding], normal_embeddings, metric='cosine')
    score = distances.min()  # or mean of k-nearest
    return score
```

**Variants:**

| Variant | Description | Pros | Cons |
|---------|-------------|------|------|
| **k-NN** | Distance to k nearest normals | Simple, effective | Slow at inference |
| **Centroid** | Distance to mean of normals | Fast | Assumes unimodal |
| **GMM** | Fit Gaussian mixture to normals | Handles multimodality | Complex |
| **One-class SVM** | Learn boundary around normals | Classical approach | Scalability issues |

```python
# k-NN based anomaly detection
from sklearn.neighbors import NearestNeighbors

class KNNAnomalyDetector:
    def __init__(self, k=10):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric='cosine')

    def fit(self, normal_embeddings):
        self.nn.fit(normal_embeddings)

    def score(self, embedding):
        distances, _ = self.nn.kneighbors([embedding])
        return distances.mean()  # Average distance to k neighbors
```

### Method 3: AMAE (Adaptation of MAE for Anomaly Detection)

[AMAE](https://arxiv.org/abs/2307.12721) is specifically designed for chest X-ray anomaly detection:

**Two-stage approach:**
1. **Stage 1:** Pretrain MAE on all data (normal + unlabeled)
2. **Stage 2:** Adapt using pseudo-labels from reconstruction error

```
Normal images: low reconstruction error → pseudo-label "normal"
High-error images: may be anomalous → use for contrastive learning
```

| Aspect | Pros | Cons |
|--------|------|------|
| **SOTA** | Best results on RSNA, NIH-CXR, VinDr-CXR | More complex pipeline |
| **Dual distribution** | Handles unlabeled anomalous data | Requires careful tuning |
| **Medical-specific** | Designed for chest X-rays | Less generalizable |

### Method 4: Classification Confidence

Use fine-tuned classifier's uncertainty as anomaly signal:

```python
def compute_confidence_score(model, image):
    logits = model(image)
    probs = torch.sigmoid(logits)

    # High confidence in any pathology = anomalous
    max_pathology_prob = probs[1:].max()  # Exclude "normal" class

    # Or: entropy of predictions
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))

    return max_pathology_prob, entropy
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Interpretability** | Can explain which pathology | Needs labeled training |
| **Specificity** | Detects specific abnormalities | May miss novel anomalies |
| **Calibration** | Can calibrate probabilities | Overconfident predictions |

### Ensemble Approach (Recommended)

Combine multiple methods for robustness:

```python
class EnsembleAnomalyDetector:
    def __init__(self, mae_model, classifier, normal_embeddings):
        self.mae = mae_model
        self.classifier = classifier
        self.knn = KNNAnomalyDetector(k=10)
        self.knn.fit(normal_embeddings)

    def score(self, image, weights=[0.4, 0.3, 0.3]):
        # Method 1: Reconstruction error
        recon_score = self.compute_reconstruction_error(image)

        # Method 2: Embedding distance
        embedding = self.mae.encode(image)
        embed_score = self.knn.score(embedding)

        # Method 3: Classification confidence
        class_score = self.classifier(image).sigmoid().max()

        # Weighted combination
        scores = [recon_score, embed_score, class_score]
        final_score = sum(w * s for w, s in zip(weights, scores))
        return final_score, {'recon': recon_score, 'embed': embed_score, 'class': class_score}
```

---

## Multimodal Integration

Our preprocessed data includes images, structured data, and text. Here's how to leverage all modalities:

### Architecture Options

#### Option 1: Late Fusion
Process each modality separately, combine at decision level.

```
Image ──→ ViT Encoder ──→ [CLS] token ──┐
                                        │
Structured ──→ MLP ──→ embedding ───────┼──→ Concat ──→ Classifier
                                        │
Text ──→ BERT ──→ [CLS] token ──────────┘
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Simplicity** | Easy to implement | Limited cross-modal learning |
| **Flexibility** | Can use pretrained unimodal encoders | May miss interactions |
| **Debugging** | Easy to analyze per-modality | Suboptimal fusion |

#### Option 2: Early Fusion
Combine inputs before processing.

```
Image patches ──┐
                │
Structured ─────┼──→ Unified Transformer ──→ Classification
                │
Text tokens ────┘
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Integration** | Deep cross-modal interactions | Needs lots of data |
| **Representations** | Joint multimodal embeddings | Training complexity |
| **Performance** | Potentially best | Harder to pretrain |

#### Option 3: Cross-Attention Fusion
Use attention to selectively combine modalities.

```
Image features ──→ Cross-Attention ←── Text features
                         │
                         ▼
                  Fused representation
                         │
Structured features ──→ Concat ──→ Classifier
```

| Aspect | Pros | Cons |
|--------|------|------|
| **Selective** | Learns which modality to attend | More parameters |
| **Interpretable** | Can visualize attention | Training stability |
| **SOTA** | Used in medical VLMs | Implementation complexity |

### Leveraging Clinical Context in Summarization

Our text preprocessing includes clinical context. This can be used:

1. **As input feature:** Include context in text encoder
2. **For attention guidance:** Use clinical features to guide image attention
3. **For anomaly context:** Different thresholds based on patient risk

```python
# Risk-stratified anomaly detection
def get_anomaly_threshold(patient_features):
    base_threshold = 0.5

    # Higher threshold for low-risk patients (more permissive)
    if patient_features['age'] < 40 and patient_features['no_comorbidities']:
        return base_threshold * 1.2

    # Lower threshold for high-risk (more sensitive)
    if patient_features['age'] > 70 or patient_features['high_acuity']:
        return base_threshold * 0.8

    return base_threshold
```

---

## Implementation Recommendations

### Recommended Pipeline

Based on the research, here's a recommended implementation path:

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: MAE Pretraining (2-3 weeks GPU time)                   │
├─────────────────────────────────────────────────────────────────┤
│ • Model: ViT-Base/16                                            │
│ • Data: Normal cohort (~33k images)                             │
│ • Mask ratio: 0.75                                              │
│ • Epochs: 800-1600                                              │
│ • Input size: 224 (or 384 for higher quality)                   │
│ • Augmentation: RandomCrop, HorizontalFlip, GaussianBlur        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Fine-Tuning (1 week GPU time)                          │
├─────────────────────────────────────────────────────────────────┤
│ • Strategy: Gradual unfreezing with LLRD                        │
│ • Data: Anomalous cohort (~200k images)                         │
│ • Loss: Focal loss (for class imbalance)                        │
│ • Epochs: 20-50                                                 │
│ • Multi-label: BCE with class weights                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Anomaly Detection (Ensemble)                           │
├─────────────────────────────────────────────────────────────────┤
│ • Method 1: MAE reconstruction error                            │
│ • Method 2: k-NN in embedding space                             │
│ • Method 3: Classification confidence                           │
│ • Threshold: 99th percentile on validation normal images        │
│ • Risk stratification: Adjust threshold by patient features     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Libraries

```python
# Core
import torch
import timm  # Vision Transformers
from transformers import AutoModel  # Text encoders

# MAE implementations
# - https://github.com/facebookresearch/mae (original)
# - https://github.com/lambert-x/medical_mae (medical-specific)

# Anomaly detection
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
```

### Compute Requirements

| Phase | GPU Memory | Time (V100) | Time (A100) |
|-------|------------|-------------|-------------|
| MAE Pretraining (33k images) | 16GB | 2-3 weeks | 1 week |
| Fine-tuning (200k images) | 16GB | 1 week | 2-3 days |
| Inference | 8GB | Real-time | Real-time |

### Evaluation Metrics

```python
# For anomaly detection
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_anomaly_detection(scores, labels):
    """
    scores: anomaly scores (higher = more anomalous)
    labels: 0 = normal, 1 = anomalous
    """
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return {'AUROC': auroc, 'AUPRC': auprc}

# For multi-label classification
def evaluate_classification(predictions, labels):
    """Per-class and mean metrics"""
    aurocs = {}
    for i, class_name in enumerate(CLASS_NAMES):
        aurocs[class_name] = roc_auc_score(labels[:, i], predictions[:, i])
    aurocs['mean'] = np.mean(list(aurocs.values()))
    return aurocs
```

---

## References

### Key Papers
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) - Original MAE
- [Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification](https://arxiv.org/abs/2210.12843) - Medical MAE
- [AMAE: Adaptation of MAE for Anomaly Detection in Chest X-Rays](https://arxiv.org/abs/2307.12721) - MICCAI 2023
- [Self-Supervised Learning for Medical Image Analysis](https://www.nature.com/articles/s41746-023-00811-0) - Comprehensive review
- [Adapting Visual-Language Models for Generalizable Anomaly Detection](https://github.com/MediaBrain-SJTU/MVFA-AD) - CVPR 2024

### Code Resources
- [medical_mae](https://github.com/lambert-x/medical_mae) - Medical MAE implementation
- [MAE (Facebook)](https://github.com/facebookresearch/mae) - Original implementation
- [MoCo v2](https://github.com/facebookresearch/moco) - Contrastive learning
- [timm](https://github.com/huggingface/pytorch-image-models) - Vision Transformer models

### Datasets & Benchmarks
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/) - Our primary dataset
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) - Stanford chest X-ray
- [NIH ChestX-ray14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) - NIH dataset
- [VinDr-CXR](https://vindr.ai/datasets/cxr) - Vietnamese dataset with annotations
