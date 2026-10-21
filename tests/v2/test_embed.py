"""M1 embeddings: file format, skipping finished work, resuming from a partial file."""

import numpy as np
import pytest

from v2 import embed
from v2.common import load_manifest
from v2.embed import load_npz, save_npz


def test_embeddings_cover_the_manifest(prepared, pixel_embeddings):
    ids, emb, meta = load_npz(pixel_embeddings)
    assert set(ids) == set(load_manifest(prepared)["study_id"])
    assert emb.shape == (len(ids), 14) and emb.dtype == np.float32
    assert meta["encoder"] == "test-pixels" and meta["size"] == 48
    assert np.all(np.diff(ids) > 0)  # sorted, no duplicates


def test_finished_embeddings_are_not_recomputed(prepared, pixel_embeddings):
    before = pixel_embeddings.stat().st_mtime_ns
    embed.main(["--model", "test-pixels", "--out", str(prepared), "--workers", "0"])
    assert pixel_embeddings.stat().st_mtime_ns == before


def test_resume_from_partial(prepared, pixel_embeddings, tmp_path):
    ids, emb, meta = load_npz(pixel_embeddings)
    out = prepared / "embeddings" / "test-pixels_40.npz"
    partial = out.with_name("test-pixels_40.partial.npz")
    embed.main(["--model", "test-pixels", "--size", "40", "--out", str(prepared), "--workers", "0"])
    full_ids, full_emb, full_meta = load_npz(out)
    out.unlink()
    # Pretend the first half was done before an interruption, with recognizable values.
    half = len(full_ids) // 2
    save_npz(partial, full_ids[:half], np.full((half, full_emb.shape[1]), 7.0, np.float32), full_meta)
    embed.main(["--model", "test-pixels", "--size", "40", "--out", str(prepared), "--workers", "0"])
    ids2, emb2, _ = load_npz(out)
    np.testing.assert_array_equal(ids2, full_ids)
    assert np.all(emb2[:half] == 7.0)  # kept from the partial file, not recomputed
    np.testing.assert_allclose(emb2[half:], full_emb[half:], rtol=1e-6)
    assert not partial.exists()


def test_tokens_file_resumes_and_matches_the_embeddings(prepared, pixel_embeddings):
    from v2.embed import load_tokens, tokens_path

    path = tokens_path(prepared, "test-pixels", 48, 12)
    embed.main(["--model", "test-pixels", "--tokens", "12", "--out", str(prepared), "--workers", "0"])
    ids, toks, meta = load_tokens(path)
    emb_ids, emb, _ = load_npz(pixel_embeddings)
    np.testing.assert_array_equal(ids, emb_ids)
    assert toks.shape == (len(ids), 145, 2) and toks.dtype == np.float16 and meta["grid"] == 12
    np.testing.assert_allclose(toks[:, 0].astype(np.float32), emb[:, 12:14], rtol=2e-3, atol=2e-3)
    full = np.array(toks)
    # Interrupted after half the studies: the rerun fills in only the rest.
    half = len(ids) // 2
    done = np.zeros(len(ids), bool)
    done[:half] = True
    arr = np.lib.format.open_memmap(path, mode="r+")
    arr[:half] = 7.0
    arr[half:] = 0.0
    arr.flush()
    del arr
    np.savez(path.with_name(path.stem + ".progress.npz"), study_id=ids, done=done,
             key=np.array(embed.json.dumps({"encoder": "test-pixels", "size": 48, "grid": 12, "pooled": False})))
    path.with_suffix(".json").unlink()
    embed.main(["--model", "test-pixels", "--tokens", "12", "--out", str(prepared), "--workers", "0"])
    _, toks2, _ = load_tokens(path)
    assert np.all(toks2[:half] == 7.0)                     # kept, not recomputed
    np.testing.assert_array_equal(toks2[half:], full[half:])
    assert not path.with_name(path.stem + ".progress.npz").exists()


def test_unfinished_tokens_are_refused(tmp_path):
    from v2.embed import load_tokens

    np.save(tmp_path / "x_tokens12.npy", np.zeros((2, 3, 4), np.float16))
    with pytest.raises(SystemExit, match="unfinished"):
        load_tokens(tmp_path / "x_tokens12.npy")


def test_one_pass_writes_tokens_and_the_usual_embeddings(prepared, pixel_embeddings, tmp_path):
    """--with-pooled (M5): the same pass gives the tokens and an .npz matching a separate v2.embed run."""
    import shutil

    from v2.embed import load_tokens, tokens_path

    out = tmp_path / "v2"
    shutil.copytree(prepared, out, ignore=shutil.ignore_patterns("embeddings", "predictions", "reports", "runs"))
    embed.main(["--model", "test-pixels", "--tokens", "12", "--with-pooled", "--out", str(out), "--workers", "0"])
    ids, emb, meta = load_npz(out / "embeddings" / "test-pixels_48.npz")
    ref_ids, ref_emb, _ = load_npz(pixel_embeddings)
    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_allclose(emb, ref_emb, rtol=1e-5, atol=1e-6)
    assert meta["encoder"] == "test-pixels" and "pooled_slice" not in meta
    tok_ids, toks, _ = load_tokens(tokens_path(out, "test-pixels", 48, 12))
    np.testing.assert_array_equal(tok_ids, ids)
    assert not list((out / "embeddings").glob("*.pooled.npy"))  # the temporary file is gone


def test_dino_shares_one_backbone_pass():
    import torch

    from v2.encoders import load_encoder

    enc = load_encoder("tiny-dinov2", size=48)
    x = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        toks, pooled = enc.tokens_and_pooled(x, 12)
        torch.testing.assert_close(toks, enc.tokens(x, 12))
        torch.testing.assert_close(pooled, enc(x))
