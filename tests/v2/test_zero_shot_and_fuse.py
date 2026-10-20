"""M2 zero-shot scoring and M4 fusion: features, red flags and the arms end to end."""

import json

import numpy as np
import pandas as pd
import pytest

from v2 import embed, fuse, zero_shot
from v2.zero_shot import TEMPLATES, zero_shot_scores


def test_zero_shot_score_is_the_cosine_difference():
    img = np.array([[1.0, 0.0], [0.0, 2.0]])
    pos = np.array([[1.0, 0.0]] * 12)
    neg = np.array([[0.0, 1.0]] * 12)
    s = zero_shot_scores(img, pos, neg)
    assert s.shape == (2, 12)
    assert np.allclose(s[0], 1.0) and np.allclose(s[1], -1.0)


def test_zero_shot_end_to_end(prepared):
    embed.main(["--model", "tiny-siglip", "--out", str(prepared), "--workers", "0"])
    zero_shot.main(["--model", "tiny-siglip", "--out", str(prepared)])
    meta = json.loads((prepared / "predictions" / "tiny-siglip_32_zeroshot.json").read_text())
    assert meta["kind"] == "zero_shot" and meta["template"]["name"] in TEMPLATES
    assert set(meta["val_macro_auroc_by_template"]) == set(TEMPLATES)


def test_zero_shot_rejects_embeddings_from_another_model(prepared, pixel_embeddings):
    with pytest.raises(SystemExit):
        zero_shot.main(["--model", "tiny-siglip", "--out", str(prepared), "--embeddings", str(pixel_embeddings)])


def test_imputation_uses_train_medians_and_adds_indicators():
    feats = pd.DataFrame({"study_id": [1, 2, 3], "a": [1.0, np.nan, 100.0]})
    out = fuse.impute_with_indicators(feats, np.array([True, True, False]))
    assert out[:, 0].tolist() == [1.0, 1.0, 100.0]  # median of train rows only
    assert out[:, 1].tolist() == [0.0, 1.0, 0.0]


def test_structured_block_drops_implausible_vitals():
    ctx = pd.DataFrame({"study_id": [1, 2, 3], **{c: [100.0, 100.0, 100.0] for c in fuse.VITALS},
                        "sex_male": [1.0, 0.0, np.nan], "age": [50.0, 60.0, np.nan]})
    ctx["temperature"] = [101.0, 500.0, 99.0]  # 500 F is a data-entry error
    block = fuse.structured_block(ctx, np.array([True, True, True]))
    n = len(fuse.VITALS) + 2
    assert block.shape == (3, 2 * n)
    assert block[1, n + 0] == 1.0 and block[1, 0] == 100.0  # missing indicator set; imputed with the train median


def test_text_features_are_fit_on_train_only():
    texts = pd.Series(["cough fever"] * 6 + ["dyspnea chf"] * 6 + ["zebra zebra"] * 4 + [None] * 2)
    train = np.array([True] * 12 + [False] * 6)
    block, stats = fuse.text_block(texts, train, n_components=4)
    assert stats["train_docs"] == 12 and stats["vocabulary"] == 6  # 2 words + 1 bigram, twice; no "zebra"
    assert np.allclose(block[12:16, :-1], 0.0)                     # words never seen in train carry nothing
    assert block[:16, -1].tolist() == [1.0] * 16 and block[16:, -1].tolist() == [0.0, 0.0]  # presence


def test_red_flags_catch_a_text_only_arm_that_is_too_good():
    def res(macro, classes):
        return {"macro_auroc": macro, "per_class": [{"label": k, "measurable": m, "auroc": a} for k, m, a in classes]}
    clean = {"image": res(0.78, []), "indication": res(0.62, [("Edema", True, 0.95), ("Fracture", False, 0.99)]),
             "text": res(0.80, [])}
    flags = fuse.red_flags(clean)
    assert any("Edema" in f for f in flags) and not any("Fracture" in f for f in flags)  # measurable classes only
    assert any("`text`" in f for f in flags)
    assert fuse.red_flags({"image": res(0.78, []), "text": res(0.60, [("Edema", True, 0.7)])}) == []


def test_fuse_end_to_end(prepared, synth):
    from v2 import context, evaluate, probe

    embed.main(["--model", "tiny-siglip", "--out", str(prepared), "--workers", "0"])
    emb = prepared / "embeddings" / "tiny-siglip_32.npz"
    probe.main(["--embeddings", str(emb), "--out", str(prepared), "--jobs", "1"])
    context.main(["--out", str(prepared), "--reports", str(synth.reports), "--mimic-iv-ed", str(synth.mimic_iv_ed),
                  "--mimic-iv", str(synth.mimic_iv), "--mimic-cxr-jpg", str(synth.mimic_root)])
    fuse.main(["--embeddings", str(emb), "--out", str(prepared), "--n-boot", "50", "--jobs", "1"])
    for arm in fuse.ARMS:
        meta = json.loads((prepared / "predictions" / f"tiny-siglip_32_ctx_{arm}.json").read_text())
        assert meta["kind"] == "fuse" and meta["arm"] == arm
    # The image arm is the M1 probe, exactly.
    a = pd.read_parquet(prepared / "predictions" / "tiny-siglip_32_ctx_image.parquet").sort_values(["study_id", "label"])
    b = pd.read_parquet(prepared / "predictions" / "tiny-siglip_32_probe.parquet").sort_values(["study_id", "label"])
    assert np.array_equal(a["score"].to_numpy(), b["score"].to_numpy())
    # The indication arm finds the planted pre-imaging hints, and only those.
    rep = evaluate.evaluate_run(prepared, "tiny-siglip_32_ctx_indication", n_boot=20)
    per = {c["label"]: c["auroc"] for c in evaluate._get(rep, "val", "v2")["per_class"]}
    per_test = {c["label"]: c["auroc"] for c in evaluate._get(rep, "test_clean", "v2")["per_class"]}
    planted = np.nanmean([per_test[k] for k in ("Pneumonia", "Edema", "Pneumothorax")])
    assert planted > 0.6, f"planted indication signal not found: {planted:.3f} ({per})"
    report = (prepared / "reports" / "m4_tiny-siglip_32.md").read_text()
    assert "Leakage red flags" in report and "**G4**" in report
    assert "G4" in evaluate.summarize(prepared, n_boot=20)
