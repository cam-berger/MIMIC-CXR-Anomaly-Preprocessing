"""M6: a language model for the pre-imaging text: frozen embeddings (v2.textenc) in the probes, and fine-tuned (v2.textft)."""

import json
import shutil

import numpy as np
import pytest
import torch

from v2 import evaluate, fuse, lineup, stack, textenc, textft
from v2.common import LABELS, TEST, TRAIN, VAL, label_matrix, load_manifest, write_predictions
from v2.textenc import HashWordTokenizer, embed_texts, lm_block, load_text, load_text_encoder, mean_pool, text_path

CPU = torch.device("cpu")
P = "test-pixels_48"
M6_FUSE_ARMS = ("indication", "image_indication", "text_prior_labels", "image_text_prior_labels")
M6_TEXT_RUNS = (f"{P}_ctx_indication_tiny-bert", f"{P}_ctx_text_prior_labels_tiny-bert", "tiny-bert_ft_indication_s0")


def test_mean_pool_leaves_padding_out():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]]])
    assert mean_pool(hidden, torch.tensor([[1, 1, 0]])).tolist() == [[2.0, 3.0]]
    assert mean_pool(hidden, torch.zeros(1, 3, dtype=torch.long)).tolist() == [[0.0, 0.0]]  # no tokens: zeros, not NaN


def test_hash_tokenizer_wraps_truncates_and_pads():
    batch = HashWordTokenizer(64)(["a b c d e", "F"], max_length=4)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    assert ids.shape == (2, 4) and ids[:, 0].tolist() == [1, 1] and ids[0, 3] == 2   # [CLS] a b [SEP]
    assert ids[1, 2] == 2 and ids[1, 3] == 0 and mask[1].tolist() == [1, 1, 1, 0]   # [CLS] f [SEP] padding
    assert HashWordTokenizer(64)(["f"], 4)["input_ids"][0, 1] == ids[1, 1]          # stable ids, lower-cased


def test_embeddings_keep_input_order_whatever_the_batching():
    enc = load_text_encoder("tiny-bert")
    texts = ["short", "a much longer text about the chest and its film", "two words", "medium length text here"]
    one = embed_texts(enc, texts, 32, batch_size=1, device=CPU)
    padded = embed_texts(enc, texts, 32, batch_size=3, device=CPU)
    assert one.shape == (4, enc.width) and one.dtype == np.float32
    assert np.allclose(one, padded, atol=1e-5)                                          # padding changes nothing
    assert np.allclose(embed_texts(enc, texts[1:2], 32, 1, CPU)[0], one[1], atol=1e-5)  # row i is text i
    assert not np.allclose(one[0], one[2], atol=1e-3)


def _planted(n: int = 400, shuffle: bool = False, seed: int = 0):
    """Texts of filler words; "edema" appears exactly when the study has Edema (or, shuffled, at random)."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    words = np.array(["cough", "fever", "pain", "fall", "sob", "weak", "line", "tube"])
    texts = [" ".join(rng.permutation([*rng.choice(words, 4), *(["edema"] if yi else [])])) for yi in y]
    raw = np.full((n, len(LABELS)), np.nan)
    raw[:, LABELS.index("Edema")] = rng.permutation(y) if shuffle else y
    return texts, raw, np.array([TRAIN] * (n - 100) + [VAL] * 100)


@pytest.mark.parametrize("shuffled", [False, True])
def test_fine_tuning_learns_a_planted_word(shuffled):
    texts, raw, split = _planted(shuffle=shuffled)
    recipe = textft.TextRecipe(lr=3e-3, epochs=3, batch_size=16, max_tokens=16)
    model, info = textft.fit_text(load_text_encoder("tiny-bert"), texts, raw, split, recipe, seed=0, device=CPU)
    assert len(info["history"]) == 3 and info["bad_steps"] == 0
    if shuffled:  # labels unrelated to the text: nothing to learn
        assert info["best_val_macro_auroc"] < 0.75
    else:
        assert info["best_val_macro_auroc"] > 0.95
    val = np.flatnonzero(split == VAL)
    logits = textft.predict_text(model, texts, val, 16, CPU)
    assert logits.shape == (len(val), len(LABELS)) and np.isfinite(logits).all()


@pytest.fixture(scope="module")
def m6(bench, tmp_path_factory):
    """A copy of M5's benchmark directory with M6's runs for the test encoder, made as run.sh bench_text does."""
    out = tmp_path_factory.mktemp("v2m6") / "bench"
    shutil.copytree(bench, out)
    emb = str(out / "embeddings" / f"{P}.npz")
    common = ["--out", str(out), "--n-boot", "20"]
    textenc.main(["--encoder", "tiny-bert", "--out", str(out), "--device", "cpu"])
    fuse.main(["--embeddings", emb, "--text-encoder", "tiny-bert", "--arms", ",".join(M6_FUSE_ARMS), "--jobs", "1",
               *common])
    fuse.main(["--embeddings", emb, "--text-encoder", "tfidf,tiny-bert", "--arms", "image_indication", "--jobs", "1",
               *common])
    textft.main(["--encoder", "tiny-bert", "--out", str(out), "--epochs", "2", "--lr", "1e-3", "--batch-size", "16",
                 "--device", "cpu"])
    pairs = [arg for img in ("probe", "attn_h0") for text in M6_TEXT_RUNS for arg in ("--pair", f"{P}_{img}", text)]
    stack.main([*pairs, *common])
    return out


def test_m6_runs_are_named_by_their_encoder(m6, bench):
    pred = m6 / "predictions"
    for src in ("indication", "chief_complaint", "prior_report"):
        ids, emb, present, meta = load_text(text_path(m6, "tiny-bert", src))
        assert emb.shape == (len(ids), 32) and present.any() and not emb[~present].any()  # no text: zeros
        assert (meta["source"], meta["max_tokens"], meta["revision"]) == (src, textenc.MAX_TOKENS[src], "-")
    want = [f"{P}_ctx_{arm}_tiny-bert" for arm in M6_FUSE_ARMS] + [f"{P}_ctx_image_indication_tfidf-tiny-bert"]
    want += ["tiny-bert_ft_indication_s0"]
    want += [f"{P}_{img}_stack_{tag}" for img in ("probe", "attn_h0")
             for tag in ("indication_tiny-bert", "text_prior_labels_tiny-bert", "tiny-bert_ft_indication")]
    missing = [run for run in want if not (pred / f"{run}.parquet").exists()]
    assert not missing, missing
    assert not list(pred.glob(f"{P}_ctx_image_tiny-bert*"))  # no text in the image arm: not refit per encoder

    meta = json.loads((pred / f"{P}_ctx_indication_tiny-bert.json").read_text())
    assert (meta["milestone"], meta["text_encoder"], meta["stack_tag"]) == ("M6", "tiny-bert", "indication_tiny-bert")
    assert meta["text_features"]["indication"]["tiny-bert"]["width"] == 32
    both = json.loads((pred / f"{P}_ctx_image_indication_tfidf-tiny-bert.json").read_text())
    assert set(both["text_features"]["indication"]) == {"tfidf", "tiny-bert"}
    ft = json.loads((pred / "tiny-bert_ft_indication_s0.json").read_text())
    assert (ft["kind"], ft["milestone"], ft["arm"], ft["stack_tag"]) == \
        ("text_finetune", "M6", "indication", "tiny-bert_ft_indication")
    assert ft["best_epoch"] in (0, 1) and ft["recipe"]["max_tokens"] == 64
    st = json.loads((pred / f"{P}_attn_h0_stack_tiny-bert_ft_indication.json").read_text())
    assert (st["milestone"], st["text_arm"], st["text_tag"]) == ("M6", "indication", "tiny-bert_ft_indication")

    reports = m6 / "reports"
    assert (reports / f"m4b_{P}_tiny-bert.md").read_text().startswith(f"# M5: text arms on the benchmark ({P}_tiny-bert)")
    assert (reports / f"m4_{P}_tfidf-tiny-bert.md").exists()
    assert (reports / f"m4_{P}.md").read_text() == (bench / "reports" / f"m4_{P}.md").read_text()  # M4's own untouched
    stacked = (reports / "m5_stack.md").read_text()
    for tag in ("indication_tiny-bert", "text_prior_labels_tiny-bert"):  # the check, against the same encoder's concat
        assert f"`{P}_probe_stack_{tag}` minus `{P}_ctx_image_{tag}`" in stacked


def test_frozen_text_block_is_standardized_on_train_rows_with_text(m6):
    m = load_manifest(m6).drop_duplicates("study_id").sort_values("study_id")
    ids, train = m["study_id"].to_numpy(), (m["split"] == TRAIN).to_numpy()
    X, info = lm_block(m6, "tiny-bert", "indication", ids, train)
    has = X[:, -1].astype(bool)
    assert X.shape == (len(ids), 33) and info == {"encoder": "tiny-bert", "train_docs": int((train & has).sum()),
                                                  "width": 32, "revision": "-"}
    assert not X[~has, :-1].any()
    fit = X[train & has, :-1]
    assert np.allclose(fit.mean(0), 0, atol=1e-6) and np.allclose(fit.std(0), 1, atol=1e-3)
    flipped, _ = lm_block(m6, "tiny-bert", "indication", ids[::-1], train[::-1])
    assert np.allclose(flipped, X[::-1])  # rows follow the ids asked for
    with pytest.raises(SystemExit, match="re-run v2.textenc"):
        lm_block(m6, "tiny-bert", "indication", np.append(ids, ids.max() + 1), np.append(train, True))


def test_textenc_keeps_a_finished_source(m6):
    path = text_path(m6, "tiny-bert", "indication")
    before = path.stat().st_mtime_ns
    textenc.main(["--encoder", "tiny-bert", "--out", str(m6), "--device", "cpu", "--sources", "indication"])
    assert path.stat().st_mtime_ns == before
    with pytest.raises(SystemExit, match="Unknown sources"):
        textenc.main(["--encoder", "tiny-bert", "--out", str(m6), "--sources", "findings"])


def test_fuse_refuses_an_unknown_text_encoder_or_textless_arms(m6):
    emb = str(m6 / "embeddings" / f"{P}.npz")
    with pytest.raises(SystemExit, match="Unknown text encoder"):
        fuse.main(["--embeddings", emb, "--out", str(m6), "--text-encoder", "gpt"])
    with pytest.raises(SystemExit, match="only arms with text"):
        fuse.main(["--embeddings", emb, "--out", str(m6), "--text-encoder", "tiny-bert", "--arms", "image"])


def test_m6_runs_line_up_by_their_inputs(m6):
    pairs = {(c["run"], c["published"]) for c in lineup.lineup(m6, n_boot=20)["compared"]}
    assert ("tiny-bert_ft_indication_s0", "BERT") in pairs                         # MMBT's fine-tuned text baseline
    assert (f"{P}_ctx_indication_tiny-bert", "BERT") in pairs
    assert (f"{P}_attn_h0_stack_tiny-bert_ft_indication", "MMBT") in pairs
    assert (f"{P}_ctx_image_indication_tfidf-tiny-bert", "MMBT") in pairs
    assert not any("text_prior_labels" in run for run, _ in pairs)                 # all text: not like for like
    summary = evaluate.summarize(m6, n_boot=20).splitlines()
    for run in ("tiny-bert_ft_indication_s0", f"{P}_ctx_indication_tiny-bert", f"{P}_probe_stack_indication_tiny-bert"):
        assert any(f"| {run} |" in line and "| M6 |" in line for line in summary), run
    assert any(f"| {P}_probe |" in line and "| M5 |" in line for line in summary)


def test_m6_report_before_the_real_encoders_ran(m6):
    res = textft.write_report(m6, prefix=P, image_run=f"{P}_attn_h0", n_boot=20)
    text = res["path"].read_text()
    assert res["path"] == m6 / "reports" / "m6_text.md"
    for heading in ("Red flags", "Text alone: the indication", "Gate G6", "Reported comparisons",
                    "Image + indication", "All earlier text", "Caveat"):
        assert heading in text, heading
    assert res["gate"]["cmp"] is None and res["gate"]["verdict"].startswith("not run")
    assert f"`{P}_attn_h0_stack_tiny-bert_ft_indication`" in text and f"`{P}_ctx_indication_tiny-bert`" in text
    assert res["best"] in {f"{P}_attn_h0_stack_{tag}" for tag in ("indication", "indication_tiny-bert",
                                                                   "tiny-bert_ft_indication")}
    assert "## 4. Per class" not in text  # needs the fine-tuned bert


def test_gate_g6_takes_the_medical_encoder_with_the_better_val_score(m6, tmp_path):
    out = tmp_path / "bench"
    shutil.copytree(m6, out)
    m = load_manifest(out).drop_duplicates("study_id")
    m = m[m["split"].isin([VAL, TEST])].reset_index(drop=True)
    rng = np.random.default_rng(0)
    good = 3.0 * (label_matrix(m) == 1.0) + rng.normal(size=(len(m), len(LABELS)))

    def fake(encoder, scores):
        write_predictions(out, textft.ft_run(encoder), m["study_id"], m["split"], scores, {
            "kind": "text_finetune", "milestone": "M6", "arm": "indication", "text_encoder": encoder,
            "stack_tag": f"{encoder}_ft_indication"})

    fake("bert", rng.normal(size=good.shape))
    fake("biomedbert", good)
    fake("clinicalbert", rng.normal(size=good.shape))
    res = textft.write_report(out, prefix=P, image_run=f"{P}_attn_h0", n_boot=50)
    assert res["gate"]["medical"] == "biomedbert" and res["gate"]["verdict"].startswith("PASS")
    text = res["path"].read_text()
    assert "**G6** (`biomedbert_ft_indication_s0`" in text
    assert any("`biomedbert_ft_indication_s0`" in flag for flag in res["flags"])  # text this good is a red flag
    assert any("`biomedbert_ft_indication_s0`" in flag for flag in lineup.lineup(out, n_boot=20)["flags"])
    table = text.split("## 4. Per class")[1].splitlines()
    first = next(line for line in table if line.startswith("| ") and not line.startswith("| Class"))
    assert first.startswith("| Cardiomegaly |")

    fake("bert", good)
    fake("biomedbert", rng.normal(size=good.shape))
    res = textft.write_report(out, prefix=P, image_run=f"{P}_attn_h0", n_boot=50)
    assert res["gate"]["verdict"].startswith("FAIL")
