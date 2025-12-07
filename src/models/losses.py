"""
Loss functions for multimodal classification with contrastive learning.

Includes:
- CLIPLoss: CLIP-style image-text contrastive loss (InfoNCE)
- SupConLoss: Supervised contrastive loss for multi-label classification
- AsymmetricFocalLoss: Focal loss with asymmetric weighting for imbalanced data
- MultiTaskLoss: Combined loss for multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss (symmetric InfoNCE).

    Aligns image and text embeddings in a shared space by:
    1. Computing similarity matrix between all image-text pairs
    2. Using InfoNCE loss in both directions (image->text and text->image)
    3. Learning temperature for scaling similarities

    Args:
        temperature: Initial temperature for scaling (learned)
        label_smoothing: Label smoothing factor
    """

    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.label_smoothing = label_smoothing

    def forward(
        self,
        img_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss.

        Args:
            img_emb: Normalized image embeddings [B, D]
            text_emb: Normalized text embeddings [B, D]

        Returns:
            Scalar loss value
        """
        # Ensure embeddings are normalized
        img_emb = F.normalize(img_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # Compute similarity matrix
        # [B, B] where element (i,j) is similarity between image i and text j
        logits = torch.matmul(img_emb, text_emb.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = img_emb.shape[0]
        labels = torch.arange(batch_size, device=img_emb.device)

        # Symmetric loss: image->text and text->image
        loss_i2t = F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            logits.T, labels, label_smoothing=self.label_smoothing
        )

        return (loss_i2t + loss_t2i) / 2


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multi-label classification.

    Extends SupCon to handle multi-label scenarios by:
    1. Computing label similarity between all pairs (Jaccard-like)
    2. Weighting positive pairs by label similarity
    3. Pulling together samples with similar labels

    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)
    Extended for multi-label by using soft positives based on label overlap.

    Args:
        temperature: Temperature for scaling similarities
        base_temperature: Base temperature for normalization
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Normalized embeddings [B, D]
            labels: Multi-hot labels [B, num_classes]
            mask: Label validity mask [B, num_classes] (1 = valid, 0 = invalid)

        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Apply mask to labels if provided
        if mask is not None:
            labels = labels * mask

        # Compute label similarity (soft positives)
        # Using Jaccard-like similarity: |intersection| / |union|
        intersection = torch.matmul(labels, labels.T)  # [B, B]
        union = (
            labels.sum(dim=1, keepdim=True)
            + labels.sum(dim=1).unsqueeze(0)
            - intersection
        )
        label_sim = intersection / (union + 1e-8)  # [B, B]

        # Compute embedding similarity
        embeddings = F.normalize(embeddings, dim=-1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]

        # Mask out self-similarity
        self_mask = torch.eye(batch_size, device=device).bool()
        sim = sim.masked_fill(self_mask, -1e9)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Compute log softmax
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Weight by label similarity (excluding self)
        positive_weight = label_sim.masked_fill(self_mask, 0)

        # Compute mean of log-likelihood weighted by positive similarity
        # Only consider pairs with some label overlap
        has_positive = positive_weight.sum(dim=1) > 0

        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (positive_weight * log_prob).sum(dim=1) / (
            positive_weight.sum(dim=1) + 1e-8
        )

        # Average over samples with positives, scale by temperature
        loss = -(self.temperature / self.base_temperature) * mean_log_prob[has_positive]

        return loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification.

    Addresses class imbalance by:
    1. Asymmetric focusing: different gamma for positives and negatives
    2. Hard negative mining: focuses on hard false positives
    3. Probability clipping: reduces contribution of easy negatives

    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2020)

    Args:
        gamma_pos: Focusing parameter for positives (usually 0)
        gamma_neg: Focusing parameter for negatives (usually 4)
        clip: Probability clipping threshold for negatives
        disable_focal: Whether to disable focal weighting
    """

    def __init__(
        self,
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        clip: float = 0.05,
        disable_focal: bool = False,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.disable_focal = disable_focal

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute asymmetric focal loss.

        Args:
            logits: Raw logits [B, num_classes]
            targets: Multi-hot targets [B, num_classes]
            mask: Label validity mask [B, num_classes] (1 = valid, 0 = invalid)

        Returns:
            Scalar loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric clipping for negatives
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Compute positive and negative losses
        pos_loss = targets * torch.log(probs_pos.clamp(min=1e-8))
        neg_loss = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8))

        # Apply focal weighting
        if not self.disable_focal:
            if self.gamma_neg > 0:
                neg_loss = neg_loss * (probs ** self.gamma_neg)
            if self.gamma_pos > 0:
                pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)

        loss = -(pos_loss + neg_loss)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            # Average over valid labels only
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Standard Focal Loss (symmetric).

    For comparison with asymmetric version.

    Args:
        gamma: Focusing parameter
        alpha: Class weight (for positive class)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute focal loss."""
        probs = torch.sigmoid(logits)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Focal weighting
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Combines:
    1. Classification loss (asymmetric focal)
    2. CLIP contrastive loss
    3. Supervised contrastive loss

    Supports learnable loss weights (uncertainty weighting).

    Args:
        cls_weight: Weight for classification loss
        clip_weight: Weight for CLIP loss
        supcon_weight: Weight for supervised contrastive loss
        learn_weights: Whether to learn loss weights
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        clip_weight: float = 0.3,
        supcon_weight: float = 0.3,
        learn_weights: bool = False,
    ):
        super().__init__()

        # Loss functions
        self.cls_loss = AsymmetricFocalLoss()
        self.clip_loss = CLIPLoss()
        self.supcon_loss = SupConLoss()

        # Weights
        self.learn_weights = learn_weights
        if learn_weights:
            # Learnable log-variance for uncertainty weighting
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_clip = nn.Parameter(torch.zeros(1))
            self.log_var_supcon = nn.Parameter(torch.zeros(1))
        else:
            self.cls_weight = cls_weight
            self.clip_weight = clip_weight
            self.supcon_weight = supcon_weight

    def forward(
        self,
        logits: torch.Tensor,
        clip_emb: torch.Tensor,
        supcon_emb: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Classification logits [B, num_classes]
            clip_emb: CLIP projection embeddings [B, D]
            supcon_emb: SupCon projection embeddings [B, D]
            labels: Multi-hot labels [B, num_classes]
            label_mask: Label validity mask [B, num_classes]
            text_emb: Text embeddings for CLIP loss [B, D]

        Returns:
            Dictionary with individual and total losses
        """
        # Individual losses
        loss_cls = self.cls_loss(logits, labels, label_mask)
        loss_clip = self.clip_loss(clip_emb, text_emb)
        loss_supcon = self.supcon_loss(supcon_emb, labels, label_mask)

        # Combine losses
        if self.learn_weights:
            # Uncertainty weighting: L = sum(L_i * exp(-s_i) + s_i)
            total = (
                loss_cls * torch.exp(-self.log_var_cls)
                + self.log_var_cls
                + loss_clip * torch.exp(-self.log_var_clip)
                + self.log_var_clip
                + loss_supcon * torch.exp(-self.log_var_supcon)
                + self.log_var_supcon
            )
        else:
            total = (
                self.cls_weight * loss_cls
                + self.clip_weight * loss_clip
                + self.supcon_weight * loss_supcon
            )

        return {
            "total": total,
            "cls": loss_cls,
            "clip": loss_clip,
            "supcon": loss_supcon,
        }


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross Entropy with label smoothing.

    Smooths hard 0/1 labels to (epsilon, 1-epsilon).

    Args:
        epsilon: Smoothing factor
    """

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute smoothed BCE loss."""
        # Smooth labels
        targets_smooth = targets * (1 - self.epsilon) + 0.5 * self.epsilon

        # BCE loss
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="none"
        )

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()
