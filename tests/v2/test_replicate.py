"""M8: a second seed of the benchmark fine-tunes, and a longer fine-tune (docs/ANALYSIS.md, M8)."""

import shutil

import pytest

from v2 import evaluate, external, finetune, replicate, stack
from v2.common import LABELS, read_predictions, write_predictions
from v2.synth import make_chexpert_test

P = "test-pixels_48"  # the synthetic benchmark's text runs
T = "tiny-dinov2_32"  # the tiny fine-tunes, standing in for rad-dino_518's
NEW = [f"{T}_ft_attn_s1", f"{T}_ft_s1", f"{T}_ft_attn_e6_s0"]
IMAGES = [f"{T}_ft_attn_s0", f"{T}_ft_s0", *NEW]


@pytest.fixture(scope="module")
def m8(bench, synth, tmp_path_factory):
    """A copy of M5's benchmark with M5b's and M8's tiny fine-tunes, their stacks, and the CheXpert test set."""
    out = tmp_path_factory.mktemp("v2m8") / "bench"
    shutil.copytree(bench, out)
    args = ["--model", "tiny-dinov2", "--batch-size", "16", "--accum", "1", "--encoder-lr", "1e-3", "--head-lr", "1e-2",
            "--llrd", "1.0", "--steps-per-epoch", "4", "--workers", "0", "--device", "cpu", "--out", str(out)]
    for seed in (0, 1):
        for head in ("attention", "mean"):
            finetune.main([*args, "--epochs", "2", "--head", head, "--seed", str(seed),
                           *(["--milestone", "M8"] if seed else [])])
    finetune.main([*args, "--epochs", "4", "--head", "attention", "--run", f"{T}_ft_attn_e6_s0", "--milestone", "M8"])
    # M6's adopted text (frozen BERT on the indication), stood in for by the TF-IDF run under BERT's tag.
    pred, meta = read_predictions(out / "predictions" / f"{P}_ctx_indication.parquet")
    write_predictions(out, f"{P}_ctx_indication_bert", pred["study_id"], pred["split"], pred[list(LABELS)].to_numpy(),
                      {**meta, "stack_tag": "indication_bert", "text_encoder": "bert", "milestone": "M6"})
    pairs = [arg for img in IMAGES for text in ("indication_bert", "indication", "text_prior_labels")
             if not (text == "indication_bert" and "_ft_s" in img)
             for arg in ("--pair", img, f"{P}_ctx_{text}")]
    stack.main([*pairs, "--out", str(out), "--n-boot", "20", "--report", "m8_stack.md"])
    ext = out.parent / "ext"
    external.prepare_chexpert(make_chexpert_test(tmp_path_factory.mktemp("chexpert"), n_patients=150, seed=7), ext,
                              out, str(synth.rad_dino_list), workers=0)
    external.apply_models(out, ext, IMAGES, workers=0, device="cpu", n_jobs=1)
    return out, ext


def test_replicate_report(m8, synth):
    bench, ext = m8
    out = replicate.main(["--bench", str(bench), "--ext", str(ext), "--labels",
                          str(synth.mimic_root / external.RADIOLOGIST_FILE), "--prefix", T, "--n-boot", "20",
                          "--min-count", "5"])  # a few dozen synthetic studies carry the radiologist's labels
    text = out["path"].read_text()
    assert out["path"] == bench / "reports" / "m8_replication.md"
    assert out["flags"] == [], out["flags"]  # each twin has its recipe; nothing stopped early
    assert [c["run"] for c in out["checks"]] == NEW and not any(c["differs"] for c in out["checks"])
    for heading in ("## Red flags", "## Checks", "## The headline verdicts", "## Seed noise", "## Training length",
                    "## Per class"):
        assert heading in text, heading
    verdicts = [c["verdict"] for c in out["claims"]]
    assert len(verdicts) == 4 and all(v.startswith(("stands", "seed-dependent")) for v in verdicts), verdicts
    assert " -> G7 PASS" in verdicts[2] or " -> G7 FAIL" in verdicts[2]
    assert all(r for c in out["claims"] for r in c["results"])  # both seeds scored on every claim
    noise = {n["a"]: n for n in out["noise"]}
    assert set(noise) == {f"{T}_ft_attn_s1", f"{T}_ft_s1", f"{T}_ft_attn_s1_stack_indication_bert",
                          *(f"{T}_{h}_s1_stack_{tag}" for h in ("ft_attn", "ft") for tag in ("indication", "text_prior_labels"))}
    assert noise[f"{T}_ft_s1"]["chexpert"] and noise[f"{T}_ft_s1"]["val"]  # fine-tunes: on CheXpert too
    assert noise[f"{T}_ft_s1_stack_indication"]["chexpert"] is None  # stacks need text, which CheXpert lacks
    assert set(out["reloads"]) == set(NEW) and all(r["correlation"] >= 0.99 for r in out["reloads"].values())
    length = out["length"]
    assert length["adopted"] == (length["val"]["diff_ci"][0] > 0) and length["test"] and length["chexpert"]
    assert "adopted" in length["decision"] and "6 epochs against 3" in text


def test_replicate_flags_a_recipe_that_differs(m8, tmp_path):
    bench = tmp_path / "bench"
    shutil.copytree(m8[0], bench, ignore=shutil.ignore_patterns("images", "runs", "embeddings", "*_tokens*"))
    run = f"{T}_ft_s1"
    pred, meta = read_predictions(bench / "predictions" / f"{run}.parquet")
    write_predictions(bench, run, pred["study_id"], pred["split"], pred[list(LABELS)].to_numpy(),
                      {**meta, "config": {**meta["config"], "head_lr": 0.005}, "stopped_early": True})
    out = replicate.replicate(bench, None, None, prefix=T, n_boot=20)
    assert any(f"`{run}`'s recipe differs" in f and "head_lr 0.005 against 0.01" in f for f in out["flags"])
    assert any(f"`{run}` stopped on its time limit" in f for f in out["flags"])
    text = out["path"].read_text()
    assert "Not recomputed: G7" in text and "Not recomputed: M7a" in text and "DIFFERS" in text
    assert [c["verdict"][:11] for c in out["claims"][2:]] == ["not decided", "not decided"]
    assert "Unchecked: revision isn't recorded" in text  # the tiny models have no Hub commit
    for r, rev in ((run, "abd"), (f"{T}_ft_s0", "abc")):  # both recorded and different: a red flag
        pred, meta = read_predictions(bench / "predictions" / f"{r}.parquet")
        write_predictions(bench, r, pred["study_id"], pred["split"], pred[list(LABELS)].to_numpy(),
                          {**meta, "config": {**meta["config"], "revision": rev}})
    out = replicate.replicate(bench, None, None, prefix=T, n_boot=20)
    assert any("revision abd against abc" in f for f in out["flags"])


def test_combine_states_the_verdict_both_seeds_support():
    order = replicate.PUBLISHED_ORDER
    assert replicate.combine("below ours", "below ours", order) == "stands: below ours"
    assert replicate.combine("below ours", "level", order) == "seed-dependent (below ours / level); both support: level"
    assert replicate.combine("**above ours**", "level", order).endswith("both support: **above ours**")
    assert replicate.combine("above 0", None, replicate.LABEL_ORDER) == "not decided: no result for seed 1"


def test_replicate_needs_the_new_runs(bench):
    with pytest.raises(SystemExit, match="run m8_gpu"):
        replicate.replicate(bench, None, None, prefix=P, n_boot=20)


def test_m8_stacks_keep_m5bs_report_and_carry_the_milestone(m8, bench):
    out, ext = m8
    assert (out / "reports" / "m8_stack.md").exists()
    assert (out / "reports" / "m5_stack.md").read_text() == (bench / "reports" / "m5_stack.md").read_text()
    milestone = [read_predictions(out / "predictions" / f"{run}_stack_{tag}.parquet")[1].get("milestone")
                 for run, tag in ((f"{T}_ft_attn_s1", "indication_bert"), (f"{T}_ft_attn_s0", "indication_bert"),
                                  (f"{T}_ft_attn_s0", "indication"))]
    assert milestone == ["M8", "M6", None]  # the later of the image's and the text's
    with pytest.raises(SystemExit, match="takes a file name"):
        stack.main(["--pair", f"{T}_ft_s1", f"{P}_ctx_indication", "--out", str(out), "--report", "../m8.md"])
    rows = evaluate.summarize(out, n_boot=10).splitlines()
    assert any("| M8 | " in r and f"| {T}_ft_attn_s1 |" in r for r in rows)
    assert any("| M5 | " in r and f"| {T}_ft_attn_s0 |" in r for r in rows)
    ext_rows = evaluate.summarize(ext, n_boot=10).splitlines()
    assert any("| M8 | " in r and f"| {T}_ft_attn_e6_s0 |" in r for r in ext_rows)
    assert any("| M7 | " in r and f"| {T}_ft_attn_s0 |" in r for r in ext_rows)
