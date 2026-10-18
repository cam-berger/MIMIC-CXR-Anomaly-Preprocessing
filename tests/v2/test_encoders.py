"""Encoders: shapes, transforms and the text side, with tiny random models (no downloads)."""

import numpy as np
import pytest
import torch
from PIL import Image

from v2.encoders import REGISTRY, _pooled, load_encoder


def _img(h=70, w=50, seed=0):
    return Image.fromarray((np.random.default_rng(seed).random((h, w)) * 255).astype(np.uint8))


@pytest.mark.parametrize("name", ["tiny-dinov2", "tiny-siglip", "test-pixels"])
def test_embedding_shape_and_finiteness(name):
    enc = load_encoder(name)
    tf = enc.eval_transform()
    x = torch.stack([tf(_img(seed=0)), tf(_img(seed=1))])
    assert x.shape == (2, 3, enc.size, enc.size)
    with torch.no_grad():
        feats = enc(x)
    assert feats.shape == (2, enc.dim)
    assert torch.isfinite(feats).all()
    assert not torch.allclose(feats[0], feats[1])


def test_dino_embedding_is_cls_plus_mean_patch():
    enc = load_encoder("tiny-dinov2")
    x = enc.eval_transform()(_img())[None]
    with torch.no_grad():
        hidden = enc.backbone(pixel_values=x).last_hidden_state
        feats = enc(x)
    assert enc.dim == 2 * enc.backbone.config.hidden_size
    torch.testing.assert_close(feats[0, : hidden.shape[-1]], hidden[0, 0])
    torch.testing.assert_close(feats[0, hidden.shape[-1]:], hidden[0, 1:].mean(0))


def test_dino_works_at_a_size_other_than_native():
    enc = load_encoder("tiny-dinov2", size=48)  # position embeddings are interpolated
    with torch.no_grad():
        feats = enc(enc.eval_transform()(_img())[None])
    assert feats.shape == (1, enc.dim)


def test_eval_transform_keeps_aspect_for_dino_and_squashes_for_siglip():
    # A tall image with a bright top band: center-cropping (DINO) removes it, squashing (SigLIP) keeps it.
    arr = np.zeros((200, 100), dtype=np.uint8)
    arr[:20] = 255
    img = Image.fromarray(arr)
    dino = load_encoder("tiny-dinov2", size=32).eval_transform()(img)
    sig = load_encoder("tiny-siglip", size=32).eval_transform()(img)
    assert dino[0, 0].max() < sig[0, 0].max()


def test_train_transform_shape():
    enc = load_encoder("tiny-dinov2")
    out = enc.train_transform()(_img())
    assert out.shape == (3, enc.size, enc.size)


def test_siglip_text_embeddings():
    enc = load_encoder("tiny-siglip")
    t = enc.encode_text(["pleural effusion", "no pleural effusion", "pleural effusion"])
    assert t.shape == (3, enc.dim)
    torch.testing.assert_close(t[0], t[2])
    assert not torch.allclose(t[0], t[1])


def test_densenet_without_download():
    enc = load_encoder("densenet121", size=64, pretrained=False)
    with torch.no_grad():
        feats = enc(enc.eval_transform()(_img())[None])
    assert feats.shape == (1, 1024)


def test_pooled_accepts_tensors_and_model_outputs():
    t = torch.ones(2, 3)

    class Out:
        pooler_output = t

    assert _pooled(t) is t and _pooled(Out()) is t
    with pytest.raises(TypeError):
        _pooled(object())


def test_registry_and_unknown_name():
    assert {"rad-dino", "dinov2-base", "medsiglip", "densenet121"} <= set(REGISTRY)
    with pytest.raises(SystemExit):
        load_encoder("no-such-model")
