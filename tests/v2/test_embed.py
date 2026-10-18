"""M1 embeddings: file format, skipping finished work, resuming from a partial file."""

import numpy as np

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
