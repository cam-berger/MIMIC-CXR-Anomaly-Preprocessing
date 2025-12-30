"""
Masked Autoencoder (MAE) for self-supervised pretraining on chest X-rays.

Implementation based on:
- "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
- "Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification"
  (medical_mae, WACV 2023)

Key adaptations for medical imaging:
- Smaller mask ratio (0.75 vs 0.90 for natural images)
- Moderate augmentations
- Grayscale to 3-channel conversion
"""

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (assumes square grid)
        cls_token: Whether to include position for CLS token

    Returns:
        Positional embeddings [grid_size*grid_size, embed_dim] or
        [1 + grid_size*grid_size, embed_dim] if cls_token
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """Generate positional embeddings from grid coordinates."""
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    Converts [B, C, H, W] to [B, num_patches, embed_dim].
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]

        Returns:
            Patch embeddings [B, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for self-supervised pretraining.

    Architecture:
        Input Image (224x224) -> Patch Embedding (14x14 patches)
            -> Random Masking (keep 25% of patches, mask 75%)
            -> Encoder (ViT, only processes visible patches)
            -> Decoder (smaller ViT, reconstructs all patches)
            -> Reconstructed Image -> MSE Loss with original

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Encoder embedding dimension
        depth: Number of encoder transformer blocks
        num_heads: Number of attention heads in encoder
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of attention heads in decoder
        mlp_ratio: MLP hidden dimension ratio
        mask_ratio: Ratio of patches to mask (0.75 recommended for medical)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_channels = in_channels

        # ---------- Encoder ----------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True
            )
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---------- Decoder ----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Prediction head: predict pixel values for each patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_channels
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize positional embeddings and other weights."""
        # Fixed sin-cos positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.patch_embed.grid_size,
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # Initialize patch_embed like a linear layer
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            imgs: [B, C, H, W]

        Returns:
            patches: [B, num_patches, patch_size**2 * C]
        """
        p = self.patch_size
        c = self.in_channels
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * c)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.

        Args:
            x: [B, num_patches, patch_size**2 * C]

        Returns:
            imgs: [B, C, H, W]
        """
        p = self.patch_size
        c = self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.

        Args:
            x: Input patches [B, N, D]
            mask_ratio: Ratio of patches to mask

        Returns:
            x_masked: Masked patches (visible only) [B, N*(1-mask_ratio), D]
            mask: Binary mask [B, N], 0 = keep, 1 = mask
            ids_restore: Indices to restore original order [B, N]
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask: 0 = keep, 1 = mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible patches.

        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask

        Returns:
            latent: Encoded visible patches [B, N_visible + 1, D]
            mask: Binary mask [B, N]
            ids_restore: Indices to restore order [B, N]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add positional embedding (without CLS)
        x = x + self.pos_embed[:, 1:, :]

        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode and reconstruct all patches.

        Args:
            x: Encoded visible patches [B, N_visible + 1, D]
            ids_restore: Indices to restore order [B, N]

        Returns:
            Predicted patches [B, N, patch_size**2 * C]
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No CLS

        # Unshuffle to original order
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2])
        )

        # Append CLS token back
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict pixels
        x = self.decoder_pred(x)

        # Remove CLS token
        x = x[:, 1:, :]

        return x

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            imgs: Original images [B, C, H, W]
            pred: Predicted patches [B, N, patch_size**2 * C]
            mask: Binary mask [B, N], 1 = masked (to reconstruct)

        Returns:
            Mean reconstruction loss over masked patches
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # FIX #4: Increased epsilon from 1e-6 to 1e-5 for FP16 safety
            # Prevents division by near-zero for low-variance patches (e.g., constant background regions)
            # See: tests/baseline_metrics.md for NaN/Inf analysis
            target = (target - mean) / (var + 1e-5) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch

        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            imgs: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask

        Returns:
            loss: Reconstruction loss
            pred: Predicted patches [B, N, patch_size**2 * C]
            mask: Binary mask [B, N]
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations (for anomaly detection).

        Args:
            imgs: Input images [B, C, H, W]

        Returns:
            CLS token representation [B, D]
        """
        # Patch embedding
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        # Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        # Return CLS token
        return x[:, 0]

    def reconstruct(
        self,
        imgs: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Reconstruct images (for anomaly detection visualization).

        Args:
            imgs: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask

        Returns:
            Reconstructed images [B, C, H, W]
        """
        _, pred, _ = self.forward(imgs, mask_ratio)

        # Denormalize if needed
        if self.norm_pix_loss:
            target = self.patchify(imgs)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6) ** 0.5 + mean

        return self.unpatchify(pred)


def mae_vit_base_patch16(**kwargs) -> MaskedAutoencoder:
    """MAE with ViT-Base encoder (recommended for medical imaging)."""
    return MaskedAutoencoder(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        **kwargs
    )


def mae_vit_large_patch16(**kwargs) -> MaskedAutoencoder:
    """MAE with ViT-Large encoder."""
    return MaskedAutoencoder(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        **kwargs
    )


def mae_vit_small_patch16(**kwargs) -> MaskedAutoencoder:
    """MAE with ViT-Small encoder (for faster training/testing)."""
    return MaskedAutoencoder(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        **kwargs
    )
