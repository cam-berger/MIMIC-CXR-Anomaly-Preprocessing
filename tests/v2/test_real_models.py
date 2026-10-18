"""
Real pretrained encoders (downloads ~1.5 GB). Off by default:

    V2_TEST_NETWORK=1 pytest tests/v2/test_real_models.py
    V2_TEST_NETWORK=1 V2_TEST_MEDSIGLIP=1 pytest ...   # needs accepted MedSigLIP terms + huggingface-cli login

Checks that each encoder loads, that v2's own preprocessing matches the model's
Hugging Face image processor, and that embeddings have the expected shape.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from v2 import encoders
from v2.encoders import EncoderSpec, load_encoder

from .conftest import network_enabled

pytestmark = [pytest.mark.network,
              pytest.mark.skipif(not network_enabled(), reason="set V2_TEST_NETWORK=1 to download real models")]


def _cxr_like(h=630, w=518, seed=0):
    """A smooth grayscale image with chest-film-like statistics (dark lungs, bright mediastinum)."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = 90 + 80 * np.exp(-((xx - w / 2) ** 2) / (2 * (w / 8) ** 2)) + 30 * yy / h
    img += np.random.default_rng(seed).normal(0, 8, (h, w))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("name", ["rad-dino", "dinov2-base"])
def test_dino_models_match_the_hf_processor(name):
    from transformers import AutoImageProcessor

    spec = encoders.REGISTRY[name]
    enc = load_encoder(name, size=spec.native_size)
    img = _cxr_like()
    ours = enc.eval_transform()(img)[None]
    processor = AutoImageProcessor.from_pretrained(spec.source)
    if name == "dinov2-base":  # its processor resizes to 256 then crops 224; v2 crops at the input size
        processor.size, processor.crop_size = {"shortest_edge": 224}, {"height": 224, "width": 224}
    ref = processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
    assert ours.shape == ref.shape
    assert (ours - ref).abs().mean() < 0.02, "v2 preprocessing differs from the model's processor"

    with torch.inference_mode():
        feats = enc(ours)
        pooled = enc.backbone(pixel_values=ref).pooler_output
    assert feats.shape == (1, 1536) and torch.isfinite(feats).all()
    cos = torch.nn.functional.cosine_similarity(feats[:, :768], pooled).item()
    assert cos > 0.99, f"CLS embedding differs from the reference pipeline (cos {cos:.4f})"


def test_rad_dino_runs_at_224_for_cpu_pilots():
    enc = load_encoder("rad-dino", size=224)
    with torch.inference_mode():
        feats = enc(enc.eval_transform()(_cxr_like())[None])
    assert feats.shape == (1, 1536) and torch.isfinite(feats).all()


def test_siglip_architecture_with_its_real_tokenizer(monkeypatch):
    """MedSigLIP is gated, so test the same code path with the open SigLIP-base checkpoint."""
    spec = EncoderSpec("siglip-base", "siglip", "google/siglip-base-patch16-224", 224)
    monkeypatch.setitem(encoders.REGISTRY, "siglip-base", spec)
    enc = load_encoder("siglip-base")
    with torch.inference_mode():
        img = enc(enc.eval_transform()(_cxr_like())[None])
        txt = enc.encode_text(["a photo of a chest x-ray", "a photo of a cat", "a photo of a chest x-ray"])
    assert img.shape == (1, enc.dim) and txt.shape == (3, enc.dim)
    assert torch.allclose(txt[0], txt[2], atol=1e-5) and not torch.allclose(txt[0], txt[1])
    sims = torch.nn.functional.cosine_similarity(img, txt[:2])
    assert sims[0] > sims[1], "SigLIP should match a gray chest-like image to 'chest x-ray' over 'cat'"


@pytest.mark.skipif(os.environ.get("V2_TEST_MEDSIGLIP") != "1", reason="set V2_TEST_MEDSIGLIP=1 (gated model)")
def test_medsiglip_loads():
    enc = load_encoder("medsiglip")
    with torch.inference_mode():
        img = enc(enc.eval_transform()(_cxr_like())[None])
        txt = enc.encode_text(["pleural effusion", "no pleural effusion"])
    assert img.shape == (1, enc.dim) and txt.shape == (2, enc.dim)
    assert torch.isfinite(img).all() and torch.isfinite(txt).all()
