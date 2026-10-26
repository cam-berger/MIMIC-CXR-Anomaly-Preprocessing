"""
v3's wiring check on synthetic data (docs/V3_PLAN.md §7 steps 1-2).

    python -m v3.smoke                            # offline, a fake Jev, ~10 s
    python -m v3.smoke --live --model MODEL       # 12 synthetic studies per format through the real API (no MIMIC)
    python -m v3.smoke --live-image --model MODEL # synthetic images: does Jev see them, do image requests work (§8)

It builds a small synthetic benchmark as v2's tests do (v2.synth, official split, pixel embeddings, the probe, the
context table, TF-IDF text arms), runs v3.setup, then the order of §7: jev_est_text on val in both question formats,
val's choice, the other arms on val and every arm on test in the chosen format, and v3.report. The fake Jev reads
the estimates and v2.synth's planted text hints back out of `state`, so a study given another study's estimates or
text, or answers stored under the wrong study, shows up as a score that differs from the one expected from that
study's own inputs. Checks: every score as expected; a re-run asks nothing; the losing format can't run on test;
jev_text finds the planted text signal; the report is written.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v2 import context, embed, fuse, prepare_data, probe
from v2.common import LABELS, load_manifest, read_predictions, setup_logging
from v2.evaluate import auroc
from v2.synth import COMPLAINTS, HINTS, make_dataset

from . import prompt, report
from . import run as v3run
from . import setup as v3setup
from .common import load_setup
from .jev import JevClient, Response

logger = logging.getLogger("v3.smoke")

FAKE_MODEL = "jev-fake-0"
BUMP = 2.0          # the fake's log-odds lift for a planted hint
TEXT_ONLY_BASE = -1.0
TEXT_SIGNAL_MIN = 0.6
ANY_SIZE = (1, 10**6)  # the fake Jev takes any image; the synthetic benchmark's are 48 px
_ESTIMATE = re.compile(r"^- (.+?): (<0\.1|>99\.9|\d+\.\d)%$", re.M)
# v2.synth's planted phrases (indications and chief complaints), per finding.
PHRASES = {label: [t[label][0].lower() for t in (HINTS, COMPLAINTS) if label in t] for label in {*HINTS, *COMPLAINTS}}


def _percent(token: str) -> float:
    return {"<0.1": 0.0005, ">99.9": 0.9995}.get(token, float(token) / 100.0 if token[0].isdigit() else 0.0)


def fake_probabilities(estimates: Optional[dict], text: str) -> dict:
    """The fake Jev's rule: the estimate's log-odds (or a fixed base), plus BUMP per planted hint in the text."""
    text = text.lower()
    out = {}
    for label in LABELS:
        p = estimates.get(label) if estimates else None
        z = math.log(p / (1 - p)) if p is not None else TEXT_ONLY_BASE
        z += BUMP * sum(phrase in text for phrase in PHRASES.get(label, []))
        out[label] = 1.0 / (1.0 + math.exp(-z))
    return out


def fake_value(question: dict, p: float) -> float:
    """The fake's answer in a question's own type: the probability, or the band whose midpoint it is (clipped)."""
    if question["type"] == "noul":
        return p
    n = len(question["criteria"])
    return min(max(n * p - 0.5, 0.0), n - 1.0)


class FakeJev:
    """A transport for JevClient that answers from `state` alone, and counts requests."""

    def __init__(self, model: str = FAKE_MODEL):
        self.model, self.calls = model, 0

    def __call__(self, method, url, headers, json_body=None, files=None, timeout=None) -> Response:
        self.calls += 1
        body = json_body if json_body is not None else json.loads(files["request"][1])
        state = body["state"]
        found = {label: _percent(tok) for label, tok in _ESTIMATE.findall(state)}
        probs = fake_probabilities(found or None, state)
        answers = {}
        for label in LABELS:
            q = body["questions"][prompt.question_name(label)]
            answers[prompt.question_name(label)] = {"type": q["type"], q["type"]: fake_value(q, probs[label])}
        return Response(200, {"model": self.model, "answers": answers,
                              "usage": {"input_tokens": len(state) // 4, "output_tokens": len(answers)}}, {})


def expected_scores(out_dir: Path, arm: str, fmt: str) -> pd.DataFrame:
    """What the fake Jev should answer for each study, from that study's own inputs (not from its state)."""
    setup = load_setup(out_dir)
    ctx = pd.read_parquet(out_dir / "context.parquet").drop_duplicates("study_id").set_index("study_id")
    probs = v3run.stage1_probabilities(out_dir, setup["image_run"])
    blocks = [b for b in prompt.ARMS[arm] if b in prompt.TEXT_BLOCKS]
    questions = prompt.questions(fmt)
    rows = {}
    for sid in ctx.index:
        text = " ".join(prompt._text(ctx.at[sid, b]) for b in blocks)
        est = None
        if prompt.needs_estimates(arm) and sid in probs.index:
            est = {label: _percent(prompt.fmt_probability(p)[:-1]) for label, p in probs.loc[sid].items()}
        fake = fake_probabilities(est, text)
        rows[sid] = {label: prompt.likelihood(q, fake_value(q, fake[label]))
                     for label, q in zip(LABELS, questions.values())}
    return pd.DataFrame.from_dict(rows, orient="index")[list(LABELS)]


def build_synthetic_bench(root: Path, n_patients: int = 400, seed: int = 3, image_size: int = 48) -> Path:
    """A small M5-style benchmark on v2.synth: official split, pixel probe, context, TF-IDF text-only arms."""
    synth = make_dataset(root / "synth", n_patients=n_patients, seed=seed)
    bench = root / "bench"
    prepare_data.main(["--mimic-cxr-jpg", str(synth.mimic_root), "--out", str(bench), "--population", "all",
                       "--split-scheme", "official", "--image-format", "jpg", "--image-size", str(image_size), "--workers", "0",
                       "--min-count", "3", "--rad-dino-list", str(synth.rad_dino_list)])
    embed.main(["--model", "test-pixels", "--out", str(bench), "--workers", "0"])
    emb = bench / "embeddings" / "test-pixels_48.npz"
    probe.main(["--embeddings", str(emb), "--out", str(bench), "--jobs", "1"])
    context.main(["--out", str(bench), "--reports", str(synth.reports), "--mimic-iv-ed", str(synth.mimic_iv_ed),
                  "--mimic-iv", str(synth.mimic_iv), "--mimic-cxr-jpg", str(synth.mimic_root)])
    fuse.main(["--embeddings", str(emb), "--out", str(bench), "--arms", "indication,text", "--n-boot", "20",
               "--jobs", "1"])
    return bench


def setup_synthetic(root: Path, n_patients: int = 400, seed: int = 3, image_size: int = 48) -> Path:
    bench = build_synthetic_bench(root, n_patients, seed, image_size)
    out = root / "v3"
    v3setup.setup(bench, out, "test-pixels_48_probe", "test-pixels_48_ctx_indication", "test-pixels_48_ctx_text",
                  n_boot=20)
    return out


def check(out: Path, fake: FakeJev, chosen: str, with_image: bool = False) -> list[str]:
    problems = []
    n_val, n_test = len(v3run.split_rows(out, "val")), len(v3run.split_rows(out, "test"))
    for f in prompt.FORMATS:
        for arm in prompt.V3_0_ARMS + (prompt.V3_1_ARMS if f == chosen else ()):
            run = prompt.run_name(arm, f)
            if not (out / "predictions" / f"{run}.parquet").exists():
                if (f == chosen or arm == prompt.PRIMARY) and (with_image or arm not in prompt.V3_1_ARMS):
                    problems.append(f"{run}: no predictions")
                continue
            wide, meta = read_predictions(out / "predictions" / f"{run}.parquet")
            exp = expected_scores(out, arm, f).loc[wide["study_id"]]
            worst = float(np.abs(wide[list(LABELS)].to_numpy() - exp.to_numpy()).max())
            if worst > 1e-9:
                problems.append(f"{run}: scores differ from the study's own inputs by up to {worst:.3g} (misaligned?)")
            n = n_val + n_test if f == chosen else n_val
            if len(wide) != n:
                problems.append(f"{run}: {len(wide)} studies in predictions, expected {n}")
            if meta.get("model") != fake.model or meta.get("prompt_hash") != prompt.prompt_hash(arm, f):
                problems.append(f"{run}: sidecar model or prompt hash wrong")
    before = fake.calls
    client = JevClient(fake.model, api_key="smoke", transport=fake)
    v3run.run_arm(out, prompt.PRIMARY, chosen, "val", client)
    if fake.calls != before:
        problems.append(f"re-run asked {fake.calls - before} questions again: the cache is not used")
    other = next(f for f in prompt.FORMATS if f != chosen)
    try:
        v3run.run_arm(out, prompt.PRIMARY, other, "test", client)
        problems.append(f"the {other} format ran on test although val chose {chosen}")
    except SystemExit:
        pass
    # The planted text signal, read by jev_text alone (the synthetic images' signal is near perfect, so on top of the
    # estimates the hints can only blur the ranking). Hint rates of 0.4-0.6 against 0.1 give AUROCs of about 0.65-0.75.
    manifest = load_manifest(out).drop_duplicates("study_id").set_index("study_id")
    wide, _ = read_predictions(out / "predictions" / f"{prompt.run_name('jev_text', chosen)}.parquet")
    aucs = []
    for label in PHRASES:
        y = manifest.loc[wide["study_id"], label].to_numpy()
        keep = y != -1  # the v2 policy: uncertain ignored, blank negative
        aucs.append(auroc(y[keep] == 1, wide[label].to_numpy()[keep]))
    if not np.nanmean(aucs) > TEXT_SIGNAL_MIN:
        problems.append(f"planted text signal not found: jev_text's mean AUROC on {', '.join(sorted(PHRASES))} "
                        f"{np.nanmean(aucs):.3f} (expected above {TEXT_SIGNAL_MIN})")
    rep = report.main(["--out", str(out), "--n-boot", "20"])
    if not Path(rep["report"]).exists() or "Gate J1" not in Path(rep["report"]).read_text():
        problems.append("v3_report.md missing or without J1")
    return problems


def run_order(out: Path, client: JevClient, concurrency: int = 2) -> str:
    """§7's order: the primary arm on val in both formats, val's choice, then the rest. Returns the chosen format."""
    for f in prompt.FORMATS:
        v3run.run_arm(out, prompt.PRIMARY, f, "val", client, concurrency=concurrency)
    chosen = report.choose_format(out)["chosen"]
    for arm in prompt.V3_0_ARMS:
        if arm != prompt.PRIMARY:
            v3run.run_arm(out, arm, chosen, "val", client, concurrency=concurrency)
    for arm in prompt.V3_0_ARMS:
        v3run.run_arm(out, arm, chosen, "test", client, concurrency=concurrency)
    return chosen


def run_v3_1(out: Path, client: JevClient, chosen: str, concurrency: int = 2) -> None:
    """§8: the image arms, val then test, in the format val chose. The fake Jev ignores the image itself."""
    for split in ("val", "test"):
        for arm in prompt.V3_1_ARMS:
            v3run.run_arm(out, arm, chosen, split, client, concurrency=concurrency, allow_image=True,
                          image_sides=ANY_SIZE)


def offline(root: Path) -> list[str]:
    out = setup_synthetic(root)
    fake = FakeJev()
    client = JevClient(fake.model, api_key="smoke", transport=fake)
    chosen = run_order(out, client)
    run_v3_1(out, client, chosen)
    problems = check(out, fake, chosen, with_image=True)
    if "Gate J2" not in (out / "reports" / "v3_report.md").read_text():
        problems.append("v3_report.md without J2")
    return problems


def live(root: Path, model: str, n: int = 12) -> list[str]:
    """The real API on synthetic text, in both formats: checks the requests and the answers' shape. No MIMIC data."""
    out = setup_synthetic(root, n_patients=150)
    problems = []
    for f in prompt.FORMATS:
        stats = v3run.run_arm(out, prompt.PRIMARY, f, "val", JevClient(model), limit=n, concurrency=2)
        logger.info("live %s: %d of %d answered by %s; mean input tokens %s; latency median %s s; distinct scores "
                    "per finding (median) %s", f, stats["n_answered"], stats["n_studies"], ", ".join(stats["models"]),
                    stats["mean_input_tokens"], stats["latency_median"], stats["distinct_scores_median"])
        if stats["n_answered"] != stats["n_studies"]:
            problems.append(f"{f}: {stats['n_failed']} live requests failed")
    return problems


WHITE = {"white": {"type": "noul", "instructions": "Is this image almost entirely white?",
                   "criteria": {"true": "Nearly every pixel is white or very light", "false": "The image is mostly dark"}}}


def _flat_jpeg(value: int, size: tuple = (518, 630)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full(size[::-1], value, np.uint8), mode="L").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def live_image(root: Path, model: str, n: int = 6) -> list[str]:
    """
    §8 on synthetic images: Jev must see the image (a white one judged white, a black one not), and the image arm's
    requests (the prepared JPEG as a multipart asset) must come back with 12 answers. No MIMIC data.
    """
    client, problems = JevClient(model), []
    seen = {name: client.ask(f"Here is an image: {prompt.IMAGE_PLACEHOLDER}", WHITE, _flat_jpeg(v)).scores["white"]
            for name, v in (("white", 250), ("black", 5))}
    logger.info("live image: P(white) %.2f for the white image, %.2f for the black one", seen["white"], seen["black"])
    if not (seen["white"] > 0.8 and seen["black"] < 0.2):
        problems.append(f"Jev does not seem to see the image: P(white) {seen['white']:.2f} and {seen['black']:.2f}")
    out = setup_synthetic(root, n_patients=150, image_size=518)  # Jev rejects images under 384 px
    rows = v3run.split_rows(out, "val").head(n)
    states = v3run.study_states(out, prompt.PRIMARY_V3_1, rows, load_setup(out)["image_run"])
    tokens = []
    for _, state, image in states:
        answer = client.ask(state, prompt.questions("noul"), image.read_bytes())
        tokens.append(answer.usage.get("input_tokens", 0))
        if len(answer.scores) != len(LABELS):
            problems.append(f"an image-arm answer has {len(answer.scores)} of 12 findings")
    logger.info("live image: %d of %d image-arm requests answered; mean input tokens %.0f", len(tokens), n,
                np.mean(tokens))
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Send 12 synthetic studies per format to the real API")
    parser.add_argument("--live-image", action="store_true",
                        help="Send synthetic images to the real API: does Jev see them, do image-arm requests work")
    parser.add_argument("--model", help="The Jev version for --live and --live-image")
    parser.add_argument("--keep", help="Build in this directory and keep it (default: a temporary one)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if (args.live or args.live_image) and not args.model:
        raise SystemExit("--live and --live-image need --model")
    mode = "live image" if args.live_image else "live" if args.live else "offline"
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="v3smoke_"))
    try:
        problems = (live_image(root, args.model) if args.live_image else live(root, args.model) if args.live
                    else offline(root))
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    for p in problems:
        logger.error("SMOKE: %s", p)
    logger.info("v3 smoke (%s): %s", mode, "FAILED" if problems else "passed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
