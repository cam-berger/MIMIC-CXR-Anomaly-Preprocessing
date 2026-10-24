"""
Image encoders for v2: one registry and one interface.

    enc = load_encoder("rad-dino", size=224)
    feats = enc(pixel_values)                  # [B, enc.dim]
    toks = enc.tokens(pixel_values, grid=12)   # [B, 1 + g*g, enc.token_dim]: CLS + pooled patches (M4b)
    tf = enc.eval_transform()                  # grayscale PIL image -> normalized [3, S, S]
    enc.encode_text(["pleural effusion"])      # SigLIP-type encoders only

Embeddings:
    dinov2 kind   CLS token concatenated with the mean patch token (DINOv2's linear-probe recipe)
    siglip kind   the model's own image embedding (attention-pooled, unnormalized)
    densenet kind global-average-pooled final feature map

The "tiny-*" and "test-pixels" entries are for tests and the wiring check:
they need no downloads. "test-pixels" is not a neural network: its features
are mean intensities over a 3x4 grid, which is exactly where v2.synth plants
its signal, so a correctly wired pipeline must score near-perfect AUROC with it.
"""

from __future__ import annotations

import json
import logging
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

logger = logging.getLogger("v2.encoders")


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    kind: str          # dinov2 | siglip | densenet | pixels
    source: str        # Hugging Face repo id, "torchvision", "random" or "fixed"
    native_size: int
    note: str = ""


REGISTRY: dict[str, EncoderSpec] = {
    spec.name: spec
    for spec in (
        EncoderSpec("rad-dino", "dinov2", "microsoft/rad-dino", 518, "ViT-B/14, self-supervised on chest X-rays (MIT)"),
        EncoderSpec("dinov2-base", "dinov2", "facebook/dinov2-base", 224, "ViT-B/14, natural images (control)"),
        EncoderSpec("medsiglip", "siglip", "google/medsiglip-448", 448, "gated: accept the terms on Hugging Face"),
        EncoderSpec("densenet121", "densenet", "torchvision", 512, "ImageNet init, classic CXR baseline"),
        EncoderSpec("tiny-dinov2", "dinov2", "random", 32, "test only: random tiny ViT"),
        EncoderSpec("tiny-siglip", "siglip", "random", 32, "test only: random tiny SigLIP"),
        EncoderSpec("test-pixels", "pixels", "fixed", 48, "test only: 3x4 grid intensities"),
    )
}
REAL_ENCODERS = ("rad-dino", "dinov2-base", "medsiglip", "densenet121")

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms (module-level classes so DataLoader workers can pickle them)
# ---------------------------------------------------------------------------

class GrayToRGB:
    """[1, H, W] -> [3, H, W] by repeating the channel; RGB tensors pass through."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t.expand(3, -1, -1) if t.shape[0] == 1 else t


def _tail(mean: Sequence[float], std: Sequence[float]) -> list:
    return [T.ToTensor(), GrayToRGB(), T.Normalize(list(mean), list(std))]


def eval_transform(kind: str, size: int, mean, std) -> T.Compose:
    if kind in ("siglip", "pixels"):
        # SigLIP models take a square image without cropping (MedSigLIP model card).
        head = [T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR)]
    else:
        # Shorter side to `size`, then center crop: the RAD-DINO / DINOv2 processors.
        head = [T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size)]
    return T.Compose(head + _tail(mean, std))


def train_transform(size: int, mean, std) -> T.Compose:
    """M3 augmentation. No horizontal flip: it would move the heart."""
    return T.Compose([
        T.RandomRotation(10, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.75, 1.333),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        *_tail(mean, std),
    ])


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def _pooled(out) -> torch.Tensor:
    """get_*_features returns a tensor in transformers 4.x and may return a ModelOutput later."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(out, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"cannot find a pooled embedding in {type(out).__name__}")


def pool_matrix(side: int, grid: int) -> torch.Tensor:
    """
    [grid*grid, side*side] weights that average a side x side token grid into grid x grid cells, with
    the windows of F.adaptive_avg_pool2d. A matmul works on every device; MPS can't adaptive-pool sizes
    that don't divide evenly (37 -> 12).
    """
    w = torch.zeros(grid, side)
    for i in range(grid):
        lo, hi = (i * side) // grid, -((-(i + 1) * side) // grid)
        w[i, lo:hi] = 1.0 / (hi - lo)
    return torch.einsum("ac,bd->abcd", w, w).reshape(grid * grid, side * side)


class Encoder(nn.Module):
    """Backbone that maps normalized [B, 3, S, S] images to [B, dim] features."""

    token_dim: Optional[int] = None        # width of each token from tokens(); None: not supported
    pooled_slice: Optional[tuple] = None   # where tokens()[:, 0] sits inside forward()'s features

    def __init__(self, spec: EncoderSpec, size: int, mean, std, dim: int, revision: Optional[str] = None):
        super().__init__()
        self.spec, self.size, self.dim = spec, size, dim
        self.mean, self.std = tuple(mean), tuple(std)
        self.revision = revision

    def token_grid(self, grid: int) -> int:
        raise NotImplementedError(f"{self.name} has no patch tokens; --tokens supports dinov2-type encoders")

    def tokens(self, pixel_values: torch.Tensor, grid: int) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} has no patch tokens; --tokens supports dinov2-type encoders")

    def tokens_and_pooled(self, pixel_values: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens() and forward() together; encoders that can share one backbone pass override this."""
        return self.tokens(pixel_values, grid), self(pixel_values)

    @property
    def name(self) -> str:
        return self.spec.name

    def eval_transform(self) -> T.Compose:
        return eval_transform(self.spec.kind, self.size, self.mean, self.std)

    def train_transform(self) -> T.Compose:
        return train_transform(self.size, self.mean, self.std)

    def describe(self) -> dict:
        return {"encoder": self.spec.name, "arch": self.spec.kind, "source": self.spec.source,
                "revision": self.revision, "size": self.size, "dim": self.dim, **_versions()}


class DinoEncoder(Encoder):
    def __init__(self, spec, size, model, mean, std, revision=None):
        cfg = model.config
        super().__init__(spec, size, mean, std, 2 * cfg.hidden_size, revision)
        self.backbone = model
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.token_dim, self.pooled_slice = cfg.hidden_size, (0, cfg.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state  # already layer-normed
        return torch.cat([hidden[:, 0], hidden[:, self.n_prefix:].mean(dim=1)], dim=-1)

    def token_grid(self, grid: int) -> int:
        return min(grid, self.size // self.backbone.config.patch_size)

    def tokens(self, pixel_values: torch.Tensor, grid: int) -> torch.Tensor:
        """The CLS token, then the patch tokens averaged into grid x grid cells: [B, 1 + g*g, hidden]."""
        return self.tokens_and_pooled(pixel_values, grid)[0]

    def tokens_and_pooled(self, pixel_values: torch.Tensor, grid: int) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens() and forward()'s embedding from one backbone pass (M5 writes both)."""
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        patches = hidden[:, self.n_prefix:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError(f"{patches.shape[1]} patch tokens don't form a square grid")
        pool = pool_matrix(side, min(grid, side)).to(patches.device, patches.dtype)
        tokens = torch.cat([hidden[:, :1], pool @ patches], dim=1)
        return tokens, torch.cat([hidden[:, 0], patches.mean(dim=1)], dim=-1)


class SiglipEncoder(Encoder):
    def __init__(self, spec, size, model, tokenizer, mean, std, revision=None):
        super().__init__(spec, size, mean, std, model.config.vision_config.hidden_size, revision)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(model.config.text_config.max_position_embeddings)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return _pooled(self.model.get_image_features(pixel_values=pixel_values))

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        if isinstance(self.tokenizer, HashTokenizer):
            ids = self.tokenizer(texts)
        else:
            # SigLIP was trained on text padded to max_length; other padding changes the embeddings.
            ids = self.tokenizer(texts, padding="max_length", max_length=self.max_length,
                                 truncation=True, return_tensors="pt")["input_ids"]
        return _pooled(self.model.get_text_features(input_ids=ids.to(device)))


class DenseNetEncoder(Encoder):
    def __init__(self, spec, size, model, mean, std):
        super().__init__(spec, size, mean, std, model.classifier.in_features, "torchvision IMAGENET1K_V1")
        self.features = model.features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.features(pixel_values))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


class PixelStatsEncoder(Encoder):
    """Test encoder: mean intensity of each 3x4 grid cell, plus global mean and std."""

    token_dim, pooled_slice = 2, (12, 14)  # tokens: (mean, std) per cell; token 0 is the global (mean, std)

    def __init__(self, spec, size):
        super().__init__(spec, size, (0.5,) * 3, (0.5,) * 3, 14, None)
        self.register_buffer("_anchor", torch.zeros(1))  # gives the module a device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        g = pixel_values[:, :1]
        # MPS can't adaptive-pool sizes that 3 or 4 don't divide (e.g. 40 px), and the MPS CPU fallback
        # doesn't cover it, so pool on the CPU. Test-only encoder; the real encoders don't pool like this.
        cells = F.adaptive_avg_pool2d(g.cpu(), (3, 4)).to(g.device).flatten(1)
        return torch.cat([cells, g.mean(dim=(1, 2, 3))[:, None], g.std(dim=(1, 2, 3))[:, None]], dim=1)

    def token_grid(self, grid: int) -> int:
        return min(grid, self.size)

    def tokens(self, pixel_values: torch.Tensor, grid: int) -> torch.Tensor:
        """Mean and std of each grid cell, after a global (mean, std) token: [B, 1 + g*g, 2]."""
        g = pixel_values[:, :1].cpu()
        k = self.token_grid(grid)
        mean = F.adaptive_avg_pool2d(g, k).flatten(1)
        std = (F.adaptive_avg_pool2d(g * g, k).flatten(1) - mean ** 2).clamp(min=0).sqrt()
        glob = torch.stack([g.mean(dim=(1, 2, 3)), g.std(dim=(1, 2, 3))], dim=-1)[:, None]
        return torch.cat([glob, torch.stack([mean, std], dim=-1)], dim=1).to(pixel_values.device)


class HashTokenizer:
    """Test tokenizer: words -> stable ids, padded with the EOS/pad id 1."""

    def __init__(self, vocab_size: int, max_length: int):
        self.vocab_size, self.max_length = vocab_size, max_length

    def __call__(self, texts: list[str]) -> torch.Tensor:
        ids = torch.ones(len(texts), self.max_length, dtype=torch.long)
        for i, text in enumerate(texts):
            words = text.lower().split()[: self.max_length - 1]
            ids[i, : len(words)] = torch.tensor(
                [2 + zlib.crc32(w.encode()) % (self.vocab_size - 2) for w in words], dtype=torch.long
            )
        return ids


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _preprocessor_stats(repo_id: str, default_mean, default_std) -> tuple[tuple, tuple]:
    """image_mean / image_std from the repo's preprocessor_config.json."""
    from huggingface_hub import hf_hub_download

    try:
        cfg = json.loads(open(hf_hub_download(repo_id, "preprocessor_config.json")).read())
        return tuple(cfg["image_mean"]), tuple(cfg["image_std"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s preprocessor config (%s); using defaults", repo_id, exc)
        return tuple(default_mean), tuple(default_std)


def _from_pretrained(cls, repo_id: str):
    """Load weights and refuse to continue if any were left randomly initialized."""
    model, info = cls.from_pretrained(repo_id, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "mask_token" not in k]
    if missing or info.get("mismatched_keys"):
        raise RuntimeError(f"{repo_id}: weights not loaded for {missing[:5] or info['mismatched_keys'][:5]} "
                           "(transformers version mismatch?)")
    return model.eval()


def hub_revision(repo_id: str, model=None, cache_dir: Optional[str] = None) -> Optional[str]:
    """
    The Hub commit the weights came from, so two runs can be checked for the same model. The pinned transformers
    leaves config._commit_hash empty; the local cache names each snapshot folder after its commit.
    """
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if commit:
        return commit
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id, "config.json", cache_dir=cache_dir)
    except Exception:  # no readable cache: the revision stays unrecorded
        return None
    if not isinstance(path, str) or Path(path).parent.parent.name != "snapshots":
        return None
    return Path(path).parent.name


def _versions() -> dict:
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


def _tiny_dinov2():
    from transformers import Dinov2Config, Dinov2Model

    torch.manual_seed(0)
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2,
                       patch_size=8, image_size=32)
    return Dinov2Model(cfg).eval()


def _tiny_siglip():
    from transformers import SiglipConfig, SiglipModel

    torch.manual_seed(0)
    cfg = SiglipConfig(
        text_config=dict(vocab_size=1000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=16,
                         pad_token_id=1, bos_token_id=1, eos_token_id=1),
        vision_config=dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, image_size=32, patch_size=8),
    )
    return SiglipModel(cfg).eval()


def load_encoder(name: str, size: Optional[int] = None, device="cpu", pretrained: bool = True) -> Encoder:
    """Build an encoder from the registry, in eval mode, on `device`."""
    if name not in REGISTRY:
        raise SystemExit(f"Unknown encoder {name!r}. Choose from: {', '.join(REGISTRY)}")
    spec = REGISTRY[name]
    size = int(size or spec.native_size)

    if spec.kind == "dinov2":
        if spec.source == "random":
            enc = DinoEncoder(spec, size, _tiny_dinov2(), (0.5,) * 3, (0.5,) * 3)
        else:
            from transformers import Dinov2Model

            model = _from_pretrained(Dinov2Model, spec.source)
            mean, std = _preprocessor_stats(spec.source, IMAGENET_MEAN, IMAGENET_STD)
            enc = DinoEncoder(spec, size, model, mean, std, hub_revision(spec.source, model))
    elif spec.kind == "siglip":
        if spec.source == "random":
            model = _tiny_siglip()
            tok = HashTokenizer(model.config.text_config.vocab_size,
                                model.config.text_config.max_position_embeddings)
            enc = SiglipEncoder(spec, size, model, tok, (0.5,) * 3, (0.5,) * 3)
        else:
            from transformers import AutoTokenizer, SiglipModel

            model = _from_pretrained(SiglipModel, spec.source)
            tok = AutoTokenizer.from_pretrained(spec.source)
            mean, std = _preprocessor_stats(spec.source, (0.5,) * 3, (0.5,) * 3)
            enc = SiglipEncoder(spec, size, model, tok, mean, std, hub_revision(spec.source, model))
    elif spec.kind == "densenet":
        from torchvision.models import DenseNet121_Weights, densenet121

        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        enc = DenseNetEncoder(spec, size, model, IMAGENET_MEAN, IMAGENET_STD)
    elif spec.kind == "pixels":
        enc = PixelStatsEncoder(spec, size)
    else:  # pragma: no cover - registry is static
        raise ValueError(spec.kind)
    return enc.to(device).eval()
