"""
Multimodal classification model with cross-attention fusion.

Components:
- TextEncoder: ClinicalBERT wrapper for clinical text
- StructuredEncoder: normalization + MLP for clinical features (vitals, labs, demographics)
- CrossAttentionFusion: Bidirectional token-level attention between image and text
- MultimodalClassifier: Complete model with contrastive learning heads
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .config import DEFAULT_TEXT_MODEL, LEGACY_IMAGE_MODE
from .mae import MaskedAutoencoder, mae_vit_base_patch16

logger = logging.getLogger(__name__)


# =============================================================================
# FIX #3: Safe Normalization Utility
# =============================================================================

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe L2 normalization that handles zero vectors without producing NaN.

    Standard F.normalize can return NaN when the input has zero L2 norm (division by zero).
    This function explicitly handles that case by returning zeros for zero-norm vectors.

    Args:
        x: Input tensor to normalize
        dim: Dimension along which to normalize (default: -1)
        eps: Epsilon for numerical stability (default: 1e-8)

    Returns:
        Normalized tensor where each vector along `dim` has unit L2 norm.
        Zero-norm vectors remain zero (rather than becoming NaN).

    Example:
        >>> x = torch.tensor([[0., 0., 0.], [3., 4., 0.]])  # First vector is zero
        >>> normalized = safe_normalize(x, dim=-1)
        >>> normalized
        tensor([[0.0000, 0.0000, 0.0000],
                [0.6000, 0.8000, 0.0000]])  # Second vector has unit norm

    Note:
        This function is critical for CLIP and SupCon losses, which require normalized
        embeddings. Without this, zero-norm embeddings (e.g., from projection layers
        with all-zero inputs) cause NaN in cosine similarity computations.

        See: tests/baseline_metrics.md for NaN/Inf root cause analysis
    """
    # Compute L2 norm along specified dimension
    norm = x.norm(p=2, dim=dim, keepdim=True)

    # Normalize (add eps to prevent division by zero)
    normalized = x / (norm + eps)

    # Replace any remaining NaN with zeros (belt-and-suspenders safety)
    # This handles edge cases like Inf inputs or extremely small eps
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized


class TextEncoder(nn.Module):
    """
    ClinicalBERT encoder for clinical text.

    Returns token embeddings (for cross-attention) and the [CLS] embedding.
    Frozen by default to preserve pretrained features; a frozen encoder is
    also kept in eval mode, so dropout does not perturb features that no
    gradient can adapt to.

    The input ids must come from the same model's tokenizer: ids from a
    different vocabulary map to unrelated wordpieces without any error.

    Args:
        model_name: HuggingFace model identifier
        freeze: Whether to freeze BERT weights
        output_dim: Target output dimension (if different from model's hidden size)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        freeze: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        # Load pretrained model
        logger.info(f"Loading text encoder: {model_name}")
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size  # Usually 768
        self.vocab_size = config.vocab_size

        # Freeze if requested
        self.frozen = freeze
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()
            logger.info("Text encoder frozen")

        # Optional projection to different dimension
        self.projection = None
        if output_dim is not None and output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
            self.output_dim = output_dim
        else:
            self.output_dim = self.hidden_size

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.frozen:
            self.bert.eval()
        return self

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text tokens.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len], 1 for real tokens

        Returns:
            Token embeddings [B, seq_len, output_dim]; index 0 is [CLS]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state

        if self.projection is not None:
            tokens = self.projection(tokens)

        return tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the [CLS] token embedding [B, output_dim]."""
        return self.encode_tokens(input_ids, attention_mask)[:, 0]


class StructuredEncoder(nn.Module):
    """
    Normalization + MLP encoder for structured clinical data.

    Raw features arrive with NaN for missing values. Each feature is
    optionally log-transformed (heavy-tailed labs), standardized with
    statistics fitted on the training set, and clipped; missing values become
    0 (the training mean) and a per-feature missingness indicator is appended,
    since whether a test was ordered is itself informative.

    Normalizing matters for more than accuracy: raw values such as NT-proBNP
    (reported up to 70,000 pg/mL) overflow fp16 (max 65,504) in the first
    Linear layer under autocast, and unscaled features let one lab dominate
    the embedding.

    The statistics are buffers, so they are saved in checkpoints and follow
    the model to its device. Call ``fit_normalization`` before training.

    Args:
        input_dim: Number of raw input features
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        dropout: Dropout probability
        log_transform: Per-feature flags for sign(x)*log1p(|x|) before standardizing
        clip_value: Standardized values are clipped to [-clip_value, clip_value]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3,
        log_transform: Optional[Sequence[bool]] = None,
        clip_value: float = 5.0,
    ):
        super().__init__()

        if log_transform is None:
            log_transform = [False] * input_dim
        if len(log_transform) != input_dim:
            raise ValueError(
                f"log_transform has {len(log_transform)} entries, expected {input_dim}"
            )

        self.input_dim = input_dim
        self.clip_value = clip_value
        self.register_buffer("log_transform", torch.tensor(list(log_transform), dtype=torch.bool))
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.register_buffer("normalization_fitted", torch.tensor(False))

        # Input: standardized values + missingness indicators
        self.encoder = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.output_dim = output_dim

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.log_transform, torch.sign(x) * torch.log1p(x.abs()), x)

    @torch.no_grad()
    def fit_normalization(self, raw: torch.Tensor) -> None:
        """
        Fit per-feature mean/std on training data.

        Args:
            raw: Raw features [N, input_dim], NaN/Inf for missing values
        """
        raw = raw.to(dtype=torch.float64, device=self.feature_mean.device)
        observed = torch.isfinite(raw)
        x = self._log_transform(torch.where(observed, raw, torch.zeros_like(raw)))
        count = observed.sum(dim=0)
        mean = (x * observed).sum(dim=0) / count.clamp(min=1)
        var = (((x - mean) ** 2) * observed).sum(dim=0) / (count - 1).clamp(min=1)
        std = var.sqrt()
        # Constant or never-observed features: leave unscaled instead of dividing by ~0
        std = torch.where((count > 1) & (std > 1e-6), std, torch.ones_like(std))

        self.feature_mean.copy_(mean.float())
        self.feature_std.copy_(std.float())
        self.normalization_fitted.fill_(True)

        never_observed = int((count == 0).sum())
        logger.info(
            f"Fitted structured-feature normalization on {raw.shape[0]:,} samples "
            f"({never_observed} feature(s) never observed)"
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Raw features [B, input_dim] -> [B, 2 * input_dim] (standardized, missing)."""
        x = x.float()
        observed = torch.isfinite(x)
        x = self._log_transform(torch.where(observed, x, torch.zeros_like(x)))
        z = ((x - self.feature_mean) / self.feature_std).clamp(-self.clip_value, self.clip_value)
        z = torch.where(observed, z, torch.zeros_like(z))
        return torch.cat([z, (~observed).to(z.dtype)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode structured features.

        Args:
            x: Raw structured features [B, input_dim], NaN for missing

        Returns:
            Embedding [B, output_dim]
        """
        return self.encoder(self.normalize(x))


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional token-level cross-attention between image and text.

    Performs:
    1. Image [CLS] attends over the text tokens (padding masked)
    2. Text [CLS] attends over the image patch tokens
    3. Combines the attended representations via MLP

    Attention needs more than one key: over a single pooled vector the
    softmax weight is always 1, the query/key projections receive no
    gradient, and the module collapses to a linear map.

    No NaN/Inf sanitization happens here on purpose: a non-finite embedding
    must reach the loss so the training loop skips the batch before backward.
    (Masking it here let the loss look valid while backward produced NaN
    gradients.) PyTorch's softmax is already overflow-safe.

    Args:
        embed_dim: Embedding dimension for both modalities
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Image attends to text
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.img_to_text_norm = nn.LayerNorm(embed_dim)

        # Text attends to image
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_img_norm = nn.LayerNorm(embed_dim)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse image and text token sequences via cross-attention.

        Args:
            img_tokens: Image tokens [B, 1 + N_patches, D]; index 0 is [CLS]
            text_tokens: Text tokens [B, L, D]; index 0 is [CLS]
            text_attention_mask: [B, L], 1 for real tokens, 0 for padding

        Returns:
            Fused embedding [B, D]
        """
        img_query = img_tokens[:, :1]
        text_query = text_tokens[:, :1]
        img_patches = img_tokens[:, 1:]

        text_padding = None
        if text_attention_mask is not None:
            text_padding = text_attention_mask == 0
            # Keep position 0 attendable: a study without text has an all-zero
            # mask, and a fully masked row would make the softmax NaN.
            text_padding[:, 0] = False

        # Image attends to text
        img_attended, _ = self.img_to_text_attn(
            query=img_query,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_padding,
        )
        img_attended = self.img_to_text_norm(img_attended + img_query)

        # Text attends to image
        text_attended, _ = self.text_to_img_attn(
            query=text_query,
            key=img_patches,
            value=img_patches,
        )
        text_attended = self.text_to_img_norm(text_attended + text_query)

        # Concatenate and fuse
        combined = torch.cat([img_attended.squeeze(1), text_attended.squeeze(1)], dim=-1)
        return self.fusion_mlp(combined)


class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier with contrastive learning heads.

    Architecture:
        Image -> MAE Encoder -> tokens [B, 1+N, 768]
        Text -> ClinicalBERT -> tokens [B, L, 768]
        Structured -> normalization + MLP -> [B, 256]

        Cross-Attention(image tokens, text tokens) -> [B, 768]
        Concat(Fused, Structured) -> [B, 1024]
        Final MLP -> [B, 512]

        Three output heads:
        1. Classification head -> [B, num_labels] (multi-label)
        2. CLIP projection -> [B, contrastive_dim] (image-text alignment)
        3. SupCon projection -> [B, contrastive_dim] (supervised contrastive)

    Args:
        mae_checkpoint: Path to pretrained MAE weights (or None to init fresh)
        num_labels: Number of classification labels (12 CheXpert pathologies)
        embed_dim: Image/text embedding dimension
        struct_input_dim: Number of raw structured features
        struct_hidden_dim: Hidden dim for structured encoder
        contrastive_dim: Output dimension for contrastive heads
        freeze_mae_epochs: Number of epochs to freeze MAE (for warmup)
        text_model_name: HuggingFace model for text encoder (must match the
            tokenizer that produced the input ids)
        freeze_text: Whether to freeze text encoder
        img_size: Input image size
        struct_log_transform: Per-feature log-transform flags for the structured encoder
    """

    def __init__(
        self,
        mae_checkpoint: Optional[Union[str, Path]] = None,
        num_labels: int = 12,
        embed_dim: int = 768,
        struct_input_dim: int = 43,
        struct_hidden_dim: int = 256,
        contrastive_dim: int = 128,
        freeze_mae_epochs: int = 5,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        freeze_text: bool = True,
        img_size: int = 224,
        struct_log_transform: Optional[Sequence[bool]] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.freeze_mae_epochs = freeze_mae_epochs

        # ---------- Encoders ----------
        # Image encoder (MAE ViT). mae_image_mode records how the pretraining
        # images were preprocessed (None when not loaded from a checkpoint).
        self.mae_image_mode: Optional[str] = None
        self.image_encoder = self._build_mae_encoder(mae_checkpoint, embed_dim, img_size)

        # Text encoder (ClinicalBERT)
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_text,
            output_dim=embed_dim,
        )

        # Structured encoder (normalization + MLP)
        self.struct_encoder = StructuredEncoder(
            input_dim=struct_input_dim,
            hidden_dim=struct_hidden_dim,
            output_dim=struct_hidden_dim,
            log_transform=struct_log_transform,
        )

        # ---------- Fusion ----------
        self.cross_attention = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
        )

        # Final fusion: cross-attention output + structured
        fusion_input_dim = embed_dim + struct_hidden_dim
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
        )

        # ---------- Task Heads ----------
        # Classification head (multi-label)
        self.classifier = nn.Linear(512, num_labels)

        # CLIP-style projection head (for image-text contrastive)
        self.clip_proj = nn.Sequential(
            nn.Linear(512, contrastive_dim),
            nn.LayerNorm(contrastive_dim),
        )

        # Text projection head for CLIP (text_emb -> contrastive_dim)
        self.text_clip_proj = nn.Sequential(
            nn.Linear(embed_dim, contrastive_dim),
            nn.LayerNorm(contrastive_dim),
        )

        # Supervised contrastive projection head
        self.supcon_proj = nn.Sequential(
            nn.Linear(512, contrastive_dim),
            nn.LayerNorm(contrastive_dim),
        )

        # The MAE decoder, mask token and fixed sin-cos position embeddings
        # are never trained here; only the encoder is (see set_epoch).
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self._current_epoch = 0
        self.set_epoch(0)

        logger.info(
            f"Initialized MultimodalClassifier: "
            f"{num_labels} labels, embed_dim={embed_dim}, "
            f"struct_input_dim={struct_input_dim}"
        )

    def _build_mae_encoder(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        embed_dim: int,
        img_size: int,
    ) -> MaskedAutoencoder:
        """Build MAE encoder, optionally loading pretrained weights."""
        # Create MAE model
        mae = mae_vit_base_patch16(img_size=img_size)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"MAE checkpoint not found: {checkpoint_path}")

            logger.info(f"Loading MAE checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Load weights. The decoder is unused here, but a missing encoder
            # weight means the "pretrained" encoder would silently stay random.
            result = mae.load_state_dict(state_dict, strict=False)
            missing_encoder = [
                k for k in result.missing_keys
                if not k.startswith("decoder") and k != "mask_token"
            ]
            if missing_encoder:
                raise RuntimeError(
                    f"MAE checkpoint {checkpoint_path} is missing encoder weights: "
                    f"{missing_encoder[:5]}{' ...' if len(missing_encoder) > 5 else ''}"
                )

            config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
            if isinstance(config, dict):
                config = config.get("mae", config)
                self.mae_image_mode = config.get("image_mode", LEGACY_IMAGE_MODE)
            logger.info("MAE checkpoint loaded successfully")

        return mae

    def image_encoder_parameters(self) -> list[nn.Parameter]:
        """MAE encoder parameters that are fine-tuned (not decoder/mask/pos_embed)."""
        mae = self.image_encoder
        return [
            *mae.patch_embed.parameters(),
            mae.cls_token,
            *mae.encoder_blocks.parameters(),
            *mae.encoder_norm.parameters(),
        ]

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch (for progressive unfreezing)."""
        self._current_epoch = epoch

        # Freeze/unfreeze MAE based on epoch
        if epoch < self.freeze_mae_epochs:
            self._freeze_mae()
        else:
            self._unfreeze_mae()

    def _freeze_mae(self) -> None:
        """Freeze MAE encoder weights."""
        for param in self.image_encoder_parameters():
            param.requires_grad = False
        logger.info("MAE encoder frozen")

    def _unfreeze_mae(self) -> None:
        """Unfreeze the MAE encoder (not the decoder or fixed position embeddings)."""
        for param in self.image_encoder_parameters():
            param.requires_grad = True
        logger.info("MAE encoder unfrozen")

    def get_image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Get image embedding from MAE encoder."""
        return self.image_encoder.encode(images)

    def get_text_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get text embedding from ClinicalBERT."""
        return self.text_encoder(input_ids, attention_mask)

    def get_structured_embedding(self, structured: torch.Tensor) -> torch.Tensor:
        """Get structured data embedding."""
        return self.struct_encoder(structured)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        structured: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass.

        Args:
            images: Input images [B, 3, H, W]
            text_tokens: Text token IDs [B, seq_len]
            structured: Raw structured features [B, num_features], NaN for missing
            attention_mask: Text attention mask [B, seq_len]
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            If return_embeddings=False:
                logits: Classification logits [B, num_labels]
                clip_emb: CLIP projection [B, contrastive_dim]
                supcon_emb: SupCon projection [B, contrastive_dim]

            If return_embeddings=True:
                logits, clip_emb, supcon_emb, plus:
                fused_emb: Fused representation [B, 512]
                img_emb: Image embedding [B, embed_dim]
                text_clip_emb: Text embedding in contrastive space [B, contrastive_dim]
        """
        # Encode each modality
        img_tokens = self.image_encoder.encode_tokens(images)  # [B, 1+N, 768]
        text_token_emb = self.text_encoder.encode_tokens(text_tokens, attention_mask)  # [B, L, 768]
        struct_emb = self.get_structured_embedding(structured)  # [B, 256]
        img_emb = img_tokens[:, 0]
        text_emb = text_token_emb[:, 0]

        # Cross-attention fusion (image + text)
        fused_img_text = self.cross_attention(img_tokens, text_token_emb, attention_mask)  # [B, 768]

        # Concatenate with structured and final fusion
        combined = torch.cat([fused_img_text, struct_emb], dim=-1)  # [B, 1024]
        fused = self.final_fusion(combined)  # [B, 512]

        # Task outputs
        logits = self.classifier(fused)  # [B, num_labels]
        # FIX #3: Use safe_normalize instead of F.normalize to handle zero-norm embeddings
        clip_emb = safe_normalize(self.clip_proj(fused), dim=-1)  # [B, 128]
        supcon_emb = safe_normalize(self.supcon_proj(fused), dim=-1)  # [B, 128]

        # Project text embedding to contrastive space for CLIP loss
        # FIX #3: Use safe_normalize instead of F.normalize to handle zero-norm embeddings
        text_clip_emb = safe_normalize(self.text_clip_proj(text_emb), dim=-1)  # [B, 128]

        if return_embeddings:
            return logits, clip_emb, supcon_emb, fused, img_emb, text_clip_emb

        return logits, clip_emb, supcon_emb

    def get_layer_groups(self) -> list[dict]:
        """
        Parameter groups for layer-wise learning rate decay (LLRD).

        Each group is ``{"name", "params", "decay_exponent", "image_encoder"}``;
        its LR is ``base_lr * lr_decay ** decay_exponent``. Following the
        MAE/BEiT fine-tuning recipe, encoder block ``i`` of ``depth`` gets
        exponent ``depth - i`` and the patch embedding + CLS token get
        ``depth + 1``; the encoder's final norm and everything outside the
        encoders get 0. ``image_encoder`` marks groups that are frozen for the
        first ``freeze_mae_epochs``.

        Every trainable parameter outside the MAE encoder and the pretrained
        BERT lands in the "head" group, so a newly added module cannot end up
        outside the optimizer.
        """
        mae = self.image_encoder
        depth = len(mae.encoder_blocks)

        groups = [{
            "name": "mae.embed",
            "params": [*mae.patch_embed.parameters(), mae.cls_token],
            "decay_exponent": depth + 1,
            "image_encoder": True,
        }]
        for i, block in enumerate(mae.encoder_blocks):
            groups.append({
                "name": f"mae.block{i}",
                "params": list(block.parameters()),
                "decay_exponent": depth - i,
                "image_encoder": True,
            })
        groups.append({
            "name": "mae.norm",
            "params": list(mae.encoder_norm.parameters()),
            "decay_exponent": 0,
            "image_encoder": True,
        })

        bert_params = [p for p in self.text_encoder.bert.parameters() if p.requires_grad]
        if bert_params:
            groups.append({
                "name": "text_encoder",
                "params": bert_params,
                "decay_exponent": depth + 1,
                "image_encoder": False,
            })

        head_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad
            and not name.startswith("image_encoder.")
            and not name.startswith("text_encoder.bert.")
        ]
        groups.append({
            "name": "head",
            "params": head_params,
            "decay_exponent": 0,
            "image_encoder": False,
        })

        return groups


class ImageOnlyClassifier(nn.Module):
    """
    Image-only classifier for comparison (no multimodal fusion).

    Uses MAE encoder + classification head.
    """

    def __init__(
        self,
        mae_checkpoint: Optional[Union[str, Path]] = None,
        num_labels: int = 12,
        embed_dim: int = 768,
        img_size: int = 224,
    ):
        super().__init__()

        # MAE encoder
        self.image_encoder = mae_vit_base_patch16(img_size=img_size)

        if mae_checkpoint is not None:
            checkpoint_path = Path(mae_checkpoint)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                self.image_encoder.load_state_dict(state_dict, strict=False)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        img_emb = self.image_encoder.encode(images)
        return self.classifier(img_emb)
