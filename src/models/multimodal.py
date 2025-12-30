"""
Multimodal classification model with cross-attention fusion.

Components:
- TextEncoder: ClinicalBERT wrapper for radiology text
- StructuredEncoder: MLP for clinical features (vitals, labs, demographics)
- CrossAttentionFusion: Bidirectional attention between image and text
- MultimodalClassifier: Complete model with contrastive learning heads
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

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
    ClinicalBERT encoder for radiology text.

    Extracts [CLS] token embedding from pretrained medical BERT model.
    Model is frozen by default to preserve pretrained features.

    Args:
        model_name: HuggingFace model identifier
        freeze: Whether to freeze BERT weights
        output_dim: Target output dimension (if different from model's hidden size)
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        freeze: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        # Load pretrained model
        logger.info(f"Loading text encoder: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size  # Usually 768

        # Freeze if requested
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("Text encoder frozen")

        # Optional projection to different dimension
        self.projection = None
        if output_dim is not None and output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
            self.output_dim = output_dim
        else:
            self.output_dim = self.hidden_size

    def forward(
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
            CLS token embedding [B, output_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if self.projection is not None:
            cls_embedding = self.projection(cls_embedding)

        return cls_embedding


class StructuredEncoder(nn.Module):
    """
    MLP encoder for structured clinical data.

    Handles demographics, vitals, and lab values with dropout
    for robustness to missing data (encoded as zeros).

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode structured features.

        Args:
            x: Structured features [B, input_dim]
               NaN values should be pre-converted to 0.0

        Returns:
            Embedding [B, output_dim]
        """
        # Handle any remaining NaN values
        x = torch.nan_to_num(x, nan=0.0)
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion between image and text.

    Performs:
    1. Image attends to text (image query, text key/value)
    2. Text attends to image (text query, image key/value)
    3. Combines attended representations via MLP

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
        img_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse image and text embeddings via cross-attention.

        Args:
            img_emb: Image embedding [B, embed_dim]
            text_emb: Text embedding [B, embed_dim]

        Returns:
            Fused embedding [B, embed_dim]
        """
        # FIX #2: Sanitize inputs before attention to prevent NaN propagation
        # Critical for numerical stability - a single NaN in embedding can corrupt entire batch
        img_emb = torch.nan_to_num(img_emb, nan=0.0, posinf=0.0, neginf=0.0)
        text_emb = torch.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp to safe range to prevent overflow in attention softmax
        # Attention computes q @ k.T * scale, which can overflow with large values
        img_emb = img_emb.clamp(-10.0, 10.0)
        text_emb = text_emb.clamp(-10.0, 10.0)

        # Add sequence dimension for attention [B, 1, D]
        img_seq = img_emb.unsqueeze(1)
        text_seq = text_emb.unsqueeze(1)

        # Image attends to text
        img_attended, _ = self.img_to_text_attn(
            query=img_seq,
            key=text_seq,
            value=text_seq,
        )
        img_attended = self.img_to_text_norm(img_attended + img_seq)

        # Text attends to image
        text_attended, _ = self.text_to_img_attn(
            query=text_seq,
            key=img_seq,
            value=img_seq,
        )
        text_attended = self.text_to_img_norm(text_attended + text_seq)

        # Remove sequence dimension [B, D]
        img_attended = img_attended.squeeze(1)
        text_attended = text_attended.squeeze(1)

        # FIX #2: Sanitize attention outputs before fusion
        # Attention can still produce NaN despite input sanitization (e.g., numerical issues in softmax)
        img_attended = torch.nan_to_num(img_attended, nan=0.0)
        text_attended = torch.nan_to_num(text_attended, nan=0.0)

        # Concatenate and fuse
        combined = torch.cat([img_attended, text_attended], dim=-1)
        fused = self.fusion_mlp(combined)

        # FIX #2: Final safety check - ensure output is finite
        # Belt-and-suspenders protection against any remaining NaN/Inf
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        return fused


class MultimodalClassifier(nn.Module):
    """
    Multimodal classifier with contrastive learning heads.

    Architecture:
        Image -> MAE Encoder -> [B, 768]
        Text -> ClinicalBERT -> [B, 768]
        Structured -> MLP -> [B, 256]

        Cross-Attention(Image, Text) -> [B, 768]
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
        struct_input_dim: Number of structured features
        struct_hidden_dim: Hidden dim for structured encoder
        contrastive_dim: Output dimension for contrastive heads
        freeze_mae_epochs: Number of epochs to freeze MAE (for warmup)
        text_model_name: HuggingFace model for text encoder
        freeze_text: Whether to freeze text encoder
    """

    def __init__(
        self,
        mae_checkpoint: Optional[Union[str, Path]] = None,
        num_labels: int = 12,
        embed_dim: int = 768,
        struct_input_dim: int = 44,
        struct_hidden_dim: int = 256,
        contrastive_dim: int = 128,
        freeze_mae_epochs: int = 5,
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        freeze_text: bool = True,
        img_size: int = 224,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.freeze_mae_epochs = freeze_mae_epochs

        # ---------- Encoders ----------
        # Image encoder (MAE ViT)
        self.image_encoder = self._build_mae_encoder(mae_checkpoint, embed_dim, img_size)

        # Text encoder (ClinicalBERT)
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            freeze=freeze_text,
            output_dim=embed_dim,
        )

        # Structured encoder (MLP)
        self.struct_encoder = StructuredEncoder(
            input_dim=struct_input_dim,
            hidden_dim=struct_hidden_dim,
            output_dim=struct_hidden_dim,
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

        # Track current epoch for MAE freezing
        self._current_epoch = 0

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
            if checkpoint_path.exists():
                logger.info(f"Loading MAE checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")

                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Load weights
                mae.load_state_dict(state_dict, strict=False)
                logger.info("MAE checkpoint loaded successfully")
            else:
                logger.warning(f"MAE checkpoint not found: {checkpoint_path}")

        return mae

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
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        logger.info("MAE encoder frozen")

    def _unfreeze_mae(self) -> None:
        """Unfreeze MAE encoder weights."""
        for param in self.image_encoder.parameters():
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
            structured: Structured features [B, num_features]
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
                text_emb: Text embedding [B, embed_dim]
        """
        # Encode each modality
        img_emb = self.get_image_embedding(images)  # [B, 768]
        text_emb = self.get_text_embedding(text_tokens, attention_mask)  # [B, 768]
        struct_emb = self.get_structured_embedding(structured)  # [B, 256]

        # Cross-attention fusion (image + text)
        fused_img_text = self.cross_attention(img_emb, text_emb)  # [B, 768]

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

    def get_layer_groups(self) -> list[list[nn.Parameter]]:
        """
        Get parameter groups for layer-wise learning rate decay.

        Returns groups from deepest (should have lowest LR) to shallowest:
        1. MAE encoder patch embed + early layers
        2. MAE encoder middle layers
        3. MAE encoder late layers
        4. Text encoder (if unfrozen)
        5. Structured encoder + fusion + heads
        """
        groups = []

        # MAE encoder layers (12 total for ViT-Base)
        mae_params = list(self.image_encoder.encoder_blocks.parameters())
        n_layers = len(self.image_encoder.encoder_blocks)

        # Split into thirds
        third = n_layers // 3
        early_idx = third
        mid_idx = 2 * third

        # Group 1: Patch embed + early encoder layers
        group1 = list(self.image_encoder.patch_embed.parameters())
        for i, block in enumerate(self.image_encoder.encoder_blocks):
            if i < early_idx:
                group1.extend(block.parameters())
        groups.append(group1)

        # Group 2: Middle encoder layers
        group2 = []
        for i, block in enumerate(self.image_encoder.encoder_blocks):
            if early_idx <= i < mid_idx:
                group2.extend(block.parameters())
        groups.append(group2)

        # Group 3: Late encoder layers + norm
        group3 = []
        for i, block in enumerate(self.image_encoder.encoder_blocks):
            if i >= mid_idx:
                group3.extend(block.parameters())
        group3.extend(self.image_encoder.encoder_norm.parameters())
        groups.append(group3)

        # Group 4: Structured encoder + cross-attention + fusion + heads
        group4 = []
        group4.extend(self.struct_encoder.parameters())
        group4.extend(self.cross_attention.parameters())
        group4.extend(self.final_fusion.parameters())
        group4.extend(self.classifier.parameters())
        group4.extend(self.clip_proj.parameters())
        group4.extend(self.supcon_proj.parameters())
        groups.append(group4)

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
