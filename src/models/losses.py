"""
Loss functions for multimodal classification with contrastive learning.

Includes:
- CLIPLoss: CLIP-style image-text contrastive loss (InfoNCE)
- SupConLoss: Supervised contrastive loss for multi-label classification
- AsymmetricFocalLoss: Focal loss with asymmetric weighting for imbalanced data
- MultiTaskLoss: Combined loss for multi-task learning
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# FIX #3: Import safe_normalize for NaN-safe embedding normalization
from .multimodal import safe_normalize


class CLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss (symmetric InfoNCE).

    Aligns image and text embeddings in a shared space by:
    1. Computing similarity matrix between all image-text pairs
    2. Using InfoNCE loss in both directions (image->text and text->image)
    3. Learning logit_scale (like OpenAI CLIP) instead of temperature

    Uses FP32 for numerical stability even when model is in FP16.

    Args:
        init_temperature: Initial temperature (converted to logit_scale)
        label_smoothing: Label smoothing factor
        max_logit_scale: Maximum logit scale (minimum effective temperature)
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        label_smoothing: float = 0.0,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        # Learn logit_scale (like OpenAI CLIP) instead of temperature
        # logit_scale = 1/temperature, stored as log for positivity
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))
        self.label_smoothing = label_smoothing
        self.max_logit_scale = max_logit_scale
        # Store max in log-space for clamping the raw parameter
        # math.log(100) = 4.605, which is a safe maximum before exp() issues
        self._max_logit_scale_log = math.log(max_logit_scale)

    def clamp_logit_scale(self) -> None:
        """
        Clamp the raw logit_scale parameter to prevent overflow.

        CRITICAL: Call this after optimizer.step() to prevent the parameter
        from growing unboundedly. Without this, the raw parameter can overflow
        during training even though forward() clamps the exp() result.

        The parameter is clamped in log-space: logit_scale <= log(max_logit_scale)
        which ensures exp(logit_scale) <= max_logit_scale.
        """
        with torch.no_grad():
            self.logit_scale.clamp_(max=self._max_logit_scale_log)

    def forward(
        self,
        img_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss in FP32 for numerical stability.

        Args:
            img_emb: Normalized image embeddings [B, D]
            text_emb: Normalized text embeddings [B, D]

        Returns:
            Scalar loss value
        """
        # Compute in FP32 for numerical stability (critical for mixed precision training)
        with torch.cuda.amp.autocast(enabled=False):
            img_emb = img_emb.float()
            text_emb = text_emb.float()

            # Ensure embeddings are normalized
            # FIX #3: Use safe_normalize instead of F.normalize to handle zero-norm embeddings
            img_emb = safe_normalize(img_emb, dim=-1)
            text_emb = safe_normalize(text_emb, dim=-1)

            # Get logit_scale (always positive via exp, clamped for safety)
            logit_scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)

            # Compute similarity matrix scaled by logit_scale
            # [B, B] where element (i,j) is similarity between image i and text j
            logits = logit_scale * torch.matmul(img_emb, text_emb.T)

            # Clamp logits to safe range to prevent exp() overflow
            logits = logits.clamp(-50, 50)

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

    Uses FP32 for numerical stability and F.log_softmax for proper handling.

    Args:
        temperature: Temperature for scaling similarities
        base_temperature: Base temperature for normalization (fixed)
        min_temperature: Minimum temperature to prevent explosion
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        min_temperature: float = 0.01,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss in FP32 for numerical stability.

        Args:
            embeddings: Normalized embeddings [B, D]
            labels: Multi-hot labels [B, num_classes] (float32)
            mask: Label validity mask [B, num_classes] (1 = valid, 0 = invalid)

        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute in FP32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            embeddings = embeddings.float()
            labels = labels.float()
            if mask is not None:
                mask = mask.float()

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

            # Clamp temperature to safe range
            temp = max(self.temperature, self.min_temperature)

            # Compute embedding similarity
            # FIX #3: Use safe_normalize instead of F.normalize to handle zero-norm embeddings
            embeddings = safe_normalize(embeddings, dim=-1)
            sim = torch.matmul(embeddings, embeddings.T) / temp  # [B, B]

            # Mask out self-similarity with large negative value
            self_mask = torch.eye(batch_size, device=device).bool()
            sim = sim.masked_fill(self_mask, -1e4)

            # Use F.log_softmax for numerical stability (handles overflow internally)
            log_prob = F.log_softmax(sim, dim=1)

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

            # Average over samples with positives, scale by temperature ratio
            loss = -(temp / self.base_temperature) * mean_log_prob[has_positive]

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
        Compute asymmetric focal loss in FP32 for numerical stability.

        Args:
            logits: Raw logits [B, num_classes]
            targets: Multi-hot targets [B, num_classes]
            mask: Label validity mask [B, num_classes] (1 = valid, 0 = invalid)

        Returns:
            Scalar loss value
        """
        # Compute in FP32 for numerical stability (critical for mixed precision training)
        # gamma_neg=4 can cause probs**4 to underflow in FP16 when probs < 0.05
        with torch.cuda.amp.autocast(enabled=False):
            logits = logits.float()
            targets = targets.float()
            if mask is not None:
                mask = mask.float()

            # Compute probabilities
            probs = torch.sigmoid(logits)
            probs_pos = probs
            probs_neg = 1 - probs

            # Asymmetric clipping for negatives
            if self.clip is not None and self.clip > 0:
                probs_neg = (probs_neg + self.clip).clamp(max=1)

            # Compute positive and negative losses with better numerical stability
            # Use clamp with slightly larger epsilon to avoid log(0) issues
            pos_loss = targets * torch.log(probs_pos.clamp(min=1e-6))
            neg_loss = (1 - targets) * torch.log(probs_neg.clamp(min=1e-6))

            # Apply focal weighting
            if not self.disable_focal:
                if self.gamma_neg > 0:
                    # Clamp probs to avoid 0**gamma which can cause issues
                    focal_weight_neg = probs.clamp(min=1e-6) ** self.gamma_neg
                    neg_loss = neg_loss * focal_weight_neg
                if self.gamma_pos > 0:
                    focal_weight_pos = (1 - probs).clamp(min=1e-6) ** self.gamma_pos
                    pos_loss = pos_loss * focal_weight_pos

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
        """Compute focal loss in FP32 for numerical stability."""
        # Compute in FP32 for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            logits = logits.float()
            targets = targets.float()
            if mask is not None:
                mask = mask.float()

            probs = torch.sigmoid(logits)

            # Binary cross entropy
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )

            # Focal weighting
            pt = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - pt).clamp(min=1e-6) ** self.gamma

            # Alpha weighting
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

            loss = alpha_weight * focal_weight * bce

            if mask is not None:
                loss = loss * mask
                return loss.sum() / (mask.sum() + 1e-8)

            return loss.mean()


def _safe_loss(name: str, loss: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Check if loss is NaN/Inf and return flag.

    Args:
        name: Loss name for logging
        loss: Loss tensor to check

    Returns:
        Tuple of (loss, is_valid) where is_valid=False means NaN/Inf detected
    """
    is_nan = torch.isnan(loss) or torch.isinf(loss)
    if is_nan:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"[WARNING] {name} loss was NaN/Inf.")
    return loss, not is_nan


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Combines:
    1. Classification loss (asymmetric focal)
    2. CLIP contrastive loss
    3. Supervised contrastive loss

    Supports learnable loss weights (uncertainty weighting).
    Includes NaN sanitization as a safety net.

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
        Compute combined loss with NaN detection.

        Args:
            logits: Classification logits [B, num_classes]
            clip_emb: CLIP projection embeddings [B, D]
            supcon_emb: SupCon projection embeddings [B, D]
            labels: Multi-hot labels [B, num_classes]
            label_mask: Label validity mask [B, num_classes]
            text_emb: Text embeddings for CLIP loss [B, D]

        Returns:
            Dictionary with individual losses, total loss, and 'valid' flag.
            If valid=False, the training loop should skip this batch.
        """
        device = logits.device

        # ===== EARLY VALIDATION: Check inputs for NaN/Inf before computing losses =====
        # This catches issues upstream (data or model forward) before they propagate
        inputs_valid = True

        if not torch.isfinite(logits).all():
            import logging
            logging.getLogger(__name__).warning("[EARLY CHECK] Non-finite logits detected")
            inputs_valid = False

        if not torch.isfinite(clip_emb).all():
            import logging
            logging.getLogger(__name__).warning("[EARLY CHECK] Non-finite clip_emb detected")
            inputs_valid = False

        if not torch.isfinite(text_emb).all():
            import logging
            logging.getLogger(__name__).warning("[EARLY CHECK] Non-finite text_emb detected")
            inputs_valid = False

        if not torch.isfinite(supcon_emb).all():
            import logging
            logging.getLogger(__name__).warning("[EARLY CHECK] Non-finite supcon_emb detected")
            inputs_valid = False

        # If any inputs are non-finite, return zero losses with valid=False
        if not inputs_valid:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "total": zero_loss,
                "cls": zero_loss,
                "clip": zero_loss,
                "supcon": zero_loss,
                "valid": False,
            }

        # Individual losses
        loss_cls = self.cls_loss(logits, labels, label_mask)
        loss_clip = self.clip_loss(clip_emb, text_emb)
        loss_supcon = self.supcon_loss(supcon_emb, labels, label_mask)

        # Check for NaN/Inf in any loss
        loss_cls, cls_valid = _safe_loss("cls", loss_cls)
        loss_clip, clip_valid = _safe_loss("clip", loss_clip)
        loss_supcon, supcon_valid = _safe_loss("supcon", loss_supcon)

        # All losses must be valid for the batch to be valid
        all_valid = cls_valid and clip_valid and supcon_valid

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
            "valid": all_valid,
        }

    def clamp_logit_scale(self) -> None:
        """
        Clamp the CLIPLoss logit_scale parameter to prevent overflow.

        CRITICAL: Call this after optimizer.step() to prevent cascading
        weight corruption. The CLIP logit_scale parameter can grow unboundedly
        during training, causing exp(logit_scale) to overflow to Inf, which
        then corrupts model parameters during the optimizer update.

        This method delegates to CLIPLoss.clamp_logit_scale().
        """
        self.clip_loss.clamp_logit_scale()


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
