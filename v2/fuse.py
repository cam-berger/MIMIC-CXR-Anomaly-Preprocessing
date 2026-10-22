"""
M4: does text written before the X-ray add to the image? (docs/V2_PLAN.md §6 M4)

    python -m v2.context                 # first: context.parquet
    python -m v2.fuse --embeddings output/v2/embeddings/rad-dino_518.npz

Fits the M1 probe once per arm, on the same studies, and writes
predictions/{embeddings stem}_ctx_{arm}.parquet:

    image                                   the image embedding only (reproduces the M1 probe)
    image_text                              + indication, chief complaint and prior report: gate G4
    image_indication, image_complaint, image_prior
                                            + one text source
    text, indication, complaint, prior      text only: the controls
    image_vitals, image_text_vitals         + triage vitals, acuity, sex and age

M4b (§6), only with --arms: the prior report encoded as the prior study's 14
CheXpert labels (one-hot, 57 features), alone and with the image:

    prior_labels, image_prior_labels, image_text_prior_labels
    text_prior_labels                       all text with them, text only (§6 M5b amendment)

Text features per source: TF-IDF over word 1-2-grams and a truncated SVD, both
fit on train only, plus a presence indicator. With --text-encoder ENCODER (M6) a
language model's frozen embeddings from v2.textenc take TF-IDF's place, and
--text-encoder tfidf,ENCODER uses both; run names then end in _ENCODER or
_tfidf-ENCODER. Then it checks the leakage red
flags, runs gate G4 and writes reports/m4_{prefix}.md (m4b_{prefix}.md when M4b arms
are among the arms). In M5's benchmark directory the report has no gate: the
comparisons with published results are in reports/lineup.md. Only aggregates are
logged.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import DEFAULT_SEED, TEST, TRAIN, VAL, default_out_dir, label_matrix, load_manifest, read_predictions
from .common import setup_logging, write_predictions
from .context import PRIOR_LABEL_COLUMNS, TEXT_SOURCES
from .embed import load_npz
from .evaluate import _get, compare_runs, evaluate_run, fmt, fmt_ci, split_scheme
from .probe import DEFAULT_CS, fit_probe, split_xy
from .textenc import TEXT_ENCODERS, lm_block

logger = logging.getLogger("v2.fuse")

SHORT = {"indication": "indication", "chief_complaint": "complaint", "prior_report": "prior"}


class Arm(NamedTuple):
    image: bool                     # the image embedding
    sources: tuple[str, ...]        # text sources
    structured: bool = False        # triage vitals, acuity, sex and age
    prior_labels: bool = False      # M4b: the prior study's CheXpert labels


ARMS: dict[str, Arm] = {
    "image": Arm(True, ()),
    "image_text": Arm(True, TEXT_SOURCES),
    **{f"image_{SHORT[s]}": Arm(True, (s,)) for s in TEXT_SOURCES},
    "text": Arm(False, TEXT_SOURCES),
    **{SHORT[s]: Arm(False, (s,)) for s in TEXT_SOURCES},
    "image_vitals": Arm(True, (), structured=True),
    "image_text_vitals": Arm(True, TEXT_SOURCES, structured=True),
    "prior_labels": Arm(False, (), prior_labels=True),
    "image_prior_labels": Arm(True, (), prior_labels=True),
    "image_text_prior_labels": Arm(True, TEXT_SOURCES, prior_labels=True),
    "text_prior_labels": Arm(False, TEXT_SOURCES, prior_labels=True),
}
M4B_ARMS = ("prior_labels", "image_prior_labels", "image_text_prior_labels", "text_prior_labels")
DEFAULT_ARMS = tuple(arm for arm in ARMS if arm not in M4B_ARMS)
# M4b's pre-registered comparisons: the labels against TF-IDF of the same prior report, and beyond all text.
PRIOR_LABEL_CMPS = (("image_prior_labels", "image_prior"), ("image_text_prior_labels", "image_text"))
PRIMARY, REFERENCE = "image_text", "image"
RED_FLAG_CLASS_AUROC = 0.90
TEXT_COMPONENTS = 128
MIN_DF = 5

# Plausible ranges; values outside are treated as missing.
VITALS = {
    "temperature": (90.0, 110.0),   # Fahrenheit
    "heartrate": (20.0, 250.0),
    "resprate": (4.0, 60.0),
    "o2sat": (50.0, 100.0),
    "sbp": (50.0, 260.0),
    "dbp": (20.0, 180.0),
    "acuity": (1.0, 5.0),
}


def text_block(texts: pd.Series, train: np.ndarray, n_components: int = TEXT_COMPONENTS,
               min_df: int = MIN_DF, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, dict]:
    """TF-IDF and truncated SVD fit on the train rows that have text, plus a presence indicator."""
    texts = texts.fillna("").astype(str).reset_index(drop=True)
    present = texts.str.strip().ne("").to_numpy()
    fit = np.asarray(train, dtype=bool) & present
    if fit.sum() < 3:
        return present[:, None].astype(np.float64), {"train_docs": int(fit.sum()), "vocabulary": 0, "components": 0}

    def vectorizer(df: int) -> TfidfVectorizer:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=df, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                               sublinear_tf=True, max_features=50_000, dtype=np.float32)

    try:
        vec = vectorizer(min_df).fit(texts[fit])
    except ValueError:  # fewer documents than min_df (tiny data): keep every term
        vec = vectorizer(1).fit(texts[fit])
    X = vec.transform(texts)
    k = max(1, min(n_components, X.shape[1], int(fit.sum()) - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(X[fit])
    Z = svd.transform(X)
    Z[~present] = 0.0
    stats = {"train_docs": int(fit.sum()), "vocabulary": len(vec.vocabulary_), "components": k,
             "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 3)}
    return np.concatenate([Z, present[:, None]], axis=1).astype(np.float64), stats


def impute_with_indicators(feats: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    """Median-impute each column (medians from train) and append a missing indicator per column."""
    values = feats.drop(columns="study_id").to_numpy(np.float64)
    missing = np.isnan(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-missing column (age without MIMIC-IV) has no median
        medians = np.nanmedian(np.where(train_mask[:, None], values, np.nan), axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    filled = np.where(missing, medians, values)
    return np.concatenate([filled, missing.astype(np.float64)], axis=1)


def structured_block(ctx: pd.DataFrame, train: np.ndarray) -> np.ndarray:
    """Triage vitals and acuity (implausible values -> missing), sex and age, imputed with indicators."""
    feats = pd.DataFrame({"study_id": ctx["study_id"].to_numpy()})
    for col, (lo, hi) in VITALS.items():
        v = pd.to_numeric(ctx[col], errors="coerce").to_numpy()
        feats[col] = np.where((v >= lo) & (v <= hi), v, np.nan)
    for col in ("sex_male", "age"):
        feats[col] = pd.to_numeric(ctx[col], errors="coerce").to_numpy()
    return impute_with_indicators(feats, train)


def prior_label_block(ctx: pd.DataFrame) -> np.ndarray:
    """
    The prior study's 14 CheXpert labels, each one-hot over positive / negative / uncertain / not mentioned,
    plus an indicator that a labeled prior exists (all zeros without one): 57 features.
    """
    found = ctx["prior_labels_found"].fillna(False).to_numpy(bool)
    cols = []
    for col in PRIOR_LABEL_COLUMNS:
        v = pd.to_numeric(ctx[col], errors="coerce").to_numpy(np.float64)
        cols += [found & (v == 1.0), found & (v == 0.0), found & (v == -1.0), found & np.isnan(v)]
    return np.stack(cols + [found], axis=1).astype(np.float64)


def text_flags(name: str, res: Optional[dict], ref: Optional[dict]) -> list[str]:
    """M4's red flags for one text-only run: at or above the image run `ref`, or a measurable class above 0.90."""
    if res is None:
        return []
    flags = []
    if ref is not None and np.isfinite(res["macro_auroc"]) and res["macro_auroc"] >= ref["macro_auroc"]:
        flags.append(f"text-only `{name}` ({fmt(res['macro_auroc'])}) is not below `image` ({fmt(ref['macro_auroc'])})")
    for c in res["per_class"]:
        if c["measurable"] and np.isfinite(c["auroc"]) and c["auroc"] > RED_FLAG_CLASS_AUROC:
            flags.append(f"text-only `{name}`: {c['label']} AUROC {c['auroc']:.3f} > {RED_FLAG_CLASS_AUROC}")
    return flags


def red_flags(clean: dict) -> list[str]:
    """Leakage red flags (§6 M4) on every text-only arm, against the `image` arm."""
    ref = clean.get(REFERENCE)
    return [flag for arm, res in clean.items() if not ARMS[arm].image for flag in text_flags(arm, res, ref)]


def reproduces_probe(out_dir: Path, image_run: str, probe_run: str) -> Optional[float]:
    """Largest score difference between the `image` arm and the M1 probe on the same embeddings, if it exists."""
    path = Path(out_dir) / "predictions" / f"{probe_run}.parquet"
    if not path.exists():
        return None
    a, _ = read_predictions(Path(out_dir) / "predictions" / f"{image_run}.parquet")
    b, _ = read_predictions(path)
    both = a.merge(b, on=["study_id", "split"], suffixes=("", "_m1"))
    if len(both) != len(a):
        return float("nan")
    cols = [c for c in a.columns if c not in ("study_id", "split")]
    return float(np.max(np.abs(both[cols].to_numpy() - both[[f"{c}_m1" for c in cols]].to_numpy())))


def gate_report(out_dir: Path, prefix: str, runs: dict, n_boot: int, features: str = "frozen",
                suffix: str = "") -> dict:
    """
    Red flags, paired comparisons against `image` on test_clean and gate G4; writes reports/m4_{prefix}.md.
    G4 is defined on frozen image features. On fine-tuned ones the same comparison is exploratory.
    With M4b's prior-label arms it writes reports/m4b_{prefix}.md instead: no G4, and the M4 arms its
    comparisons need are read from M4's predictions when this call didn't fit them. `suffix` names the text
    encoder of the runs (M6): their report gets it too, and the `image` arm they're compared with is the TF-IDF one.
    """
    m4b = any(arm in M4B_ARMS for arm in runs)
    bench = split_scheme(out_dir) == "official"  # M5: test_clean is the official test split, test_v1 is empty
    borrowed = {}
    if m4b or suffix:
        for arm in (REFERENCE, *(b for _, b in PRIOR_LABEL_CMPS)):
            run = f"{prefix}_ctx_{arm}" + ("" if arm == REFERENCE else suffix)
            if arm not in runs and (Path(out_dir) / "predictions" / f"{run}.parquet").exists():
                borrowed[arm] = run
        runs = {**borrowed, **runs}
    reps = {arm: evaluate_run(out_dir, run, n_boot) for arm, run in runs.items()}
    clean = {arm: _get(rep, "test_clean", "v2") for arm, rep in reps.items()}
    flags = red_flags(clean)
    cmps = {}
    if REFERENCE in runs:
        for arm, run in runs.items():
            if arm != REFERENCE and ARMS[arm].image:
                cmps[arm] = compare_runs(out_dir, run, runs[REFERENCE], "test_clean", "v2", n_boot)
    prior_cmps = []
    for a, b in PRIOR_LABEL_CMPS:
        if a in runs and b in runs:
            sets = (("test_clean", "v2"), ("test_clean", "u0")) if bench else \
                (("test_clean", "v2"), ("test_v1", "v2"), ("test_v1", "v1"))
            for test_set, metric in sets:
                try:
                    prior_cmps.append(compare_runs(out_dir, runs[a], runs[b], test_set, metric, n_boot))
                except SystemExit as exc:  # no shared studies or no frozen classes for that set
                    logger.warning("%s - %s on %s skipped: %s", runs[a], runs[b], test_set, exc)
    g4 = cmps.get(PRIMARY)
    if g4 is None:
        verdict = "not run: needs the `image` and `image_text` arms (all three text sources)"
    elif not np.isfinite(g4["diff"]):
        verdict = "n/a: no measurable classes (rehearsal data?)"
    elif g4["a_better"]:
        verdict = "PASS: pre-imaging text adds to the image"
    else:
        verdict = "FAIL: no detectable gain from pre-imaging text"
    drift = reproduces_probe(out_dir, runs[REFERENCE], f"{prefix}_probe") if REFERENCE in runs else None

    gate = "G4" if features == "frozen" else "Text gain on fine-tuned features (exploratory, not G4)"
    title = f"# M4b: prior-study labels ({prefix}{suffix})" if m4b else f"# M4: pre-imaging text + image ({prefix}{suffix})"
    if bench:
        title = f"# M5: text arms on the benchmark ({prefix}{suffix})"
    lines = [title, "", f"Macro AUROC on {'the official test split' if bench else 'test_clean'}; paired differences "
             "against `image` (docs/V2_PLAN.md §6 " + ("M4b part 2)." if m4b else "M4)."), ""]
    if features != "frozen":
        lines += ["**Exploratory:** these image features come from a fine-tune that was fit on the training "
                  "studies, which likely understates what text adds. G4 is defined on frozen features only.", ""]
    lines += ["## Leakage red flags (read these first)", ""]
    lines += [f"- **RED FLAG:** {f}" for f in flags] or ["- none"]
    lines += ["", "## Arms", "", f"| Arm | {'official test' if bench else 'test_clean'} | minus `image` |",
              "|---|---|---|"]
    for arm, run in runs.items():
        res = clean[arm]
        auc = fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"
        diff = fmt_ci(cmps[arm]["diff"], cmps[arm]["diff_ci"]) if arm in cmps else ""
        lines.append(f"| `{run}`{(' (from M5a)' if bench else ' (from M4)') if arm in borrowed else ''} | {auc} | "
                     f"{diff} |")
    if not m4b:
        if bench:
            lines += ["", "No gate here: G4 is defined on the v2 cohort. The comparisons with published results are "
                      "in reports/lineup.md.", ""]
        else:
            g4_text = fmt_ci(g4["diff"], g4["diff_ci"]) if g4 else "-"
            lines += ["", f"**{gate}** (`{PRIMARY}` minus `{REFERENCE}`): {g4_text} -> **{verdict}**", ""]
        if drift is not None:
            lines.append(f"`image` arm vs `{prefix}_probe`: largest score difference {drift:.2e} (should be 0).")
    if prior_cmps:
        lines += ["", "## Prior-study labels (M4b, exploratory)", "", "| Comparison | Set | Difference [95% CI] |",
                  "|---|---|---|"]
        for c in prior_cmps:
            where = ("official test" if bench else f"{c['test_set']}") + \
                {"v1": ", v1 metric", "u0": ", positive vs rest", "v2": ""}[c["metric"]]
            lines.append(f"| `{c['a']}` minus `{c['b']}` | {where} | {fmt_ci(c['diff'], c['diff_ci'])} |")
        if not bench:
            lines += ["", "Read test_v1 under the v2 policy first. Under the v1 policy only findings the report "
                      "mentions count, and text the radiologist reads makes a mention more likely, which biases text "
                      "gains toward zero (§6 M4b amendment)."]
        lines += ["", "Caveat (§6 M4b): radiologists read the prior report and often carry its wording forward, "
                  "and the same labeler reads both reports, so part of any gain is predicting the report rather "
                  "than the patient."]
    rep_dir = Path(out_dir) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / f"{'m4b' if m4b else 'm4'}_{prefix}{suffix}.md"
    path.write_text("\n".join(lines) + "\n")
    if bench:
        verdict = "not a gate here: G4 is defined on the v2 cohort"
    return {"flags": flags, "g4": None if m4b else g4, "verdict": verdict, "probe_drift": drift, "gate": gate,
            "prior_label_cmps": prior_cmps, "path": path, "m4b": m4b}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", required=True,
                        help="Image embeddings: frozen ones from v2.embed (G4), or fine-tuned ones from "
                             "v2.finetune --save-embeddings (exploratory)")
    parser.add_argument("--context", help="context.parquet from v2.context (default: OUT/context.parquet)")
    parser.add_argument("--out", default=None, help="v2 output dir (default: $OUTPUT_PATH/v2 or output/v2)")
    parser.add_argument("--run", help="Run-name prefix (default: {embeddings stem})")
    parser.add_argument("--arms", help=f"Comma-separated subset of: {', '.join(ARMS)} "
                                       f"(default: all but the M4b arms {', '.join(M4B_ARMS)})")
    parser.add_argument("--text-encoder", default="tfidf",
                        help="tfidf (M4), a v2.textenc encoder whose frozen embeddings replace TF-IDF (M6), or "
                             "tfidf,ENCODER for both")
    parser.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()  # also loads .env
    ctx_path = Path(args.context) if args.context else out_dir / "context.parquet"
    if not ctx_path.exists():
        raise SystemExit(f"{ctx_path} not found: run `python -m v2.context` first")
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(DEFAULT_ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"Unknown arms {unknown}; choose from {list(ARMS)}")
    encoders = [e.strip() for e in args.text_encoder.split(",")]
    if any(e != "tfidf" and e not in TEXT_ENCODERS for e in encoders):
        raise SystemExit(f"Unknown text encoder in {args.text_encoder!r}: tfidf or one of {sorted(TEXT_ENCODERS)}")
    suffix = "" if encoders == ["tfidf"] else "_" + "-".join(encoders)
    if suffix:  # arms without text are the same with any encoder: they're the TF-IDF runs
        textless = [a for a in arms if not ARMS[a].sources]
        if textless:
            logger.info("Arms %s use no text, so --text-encoder doesn't change them: skipped", ", ".join(textless))
        arms = [a for a in arms if ARMS[a].sources]
        if not arms:
            raise SystemExit("--text-encoder changes only arms with text, and none was asked for")

    manifest = load_manifest(out_dir)
    ids, emb, meta = load_npz(Path(args.embeddings))
    ctx = pd.read_parquet(ctx_path).drop_duplicates("study_id").set_index("study_id")
    no_ctx = int((~pd.Index(ids).isin(ctx.index)).sum())
    if no_ctx:
        logger.warning("%d embedded studies have no context row: treated as having no text or vitals", no_ctx)
    ctx = ctx.reindex(ids).rename_axis("study_id").reset_index()
    train = (manifest.set_index("study_id")["split"].reindex(ids) == TRAIN).to_numpy()

    needed = {src for arm in arms for src in ARMS[arm].sources}  # features only for the sources the arms use
    blocks, available = {}, {}
    for src in TEXT_SOURCES:
        available[src] = bool(ctx[src].fillna("").str.strip().ne("").any())
        if not (available[src] and src in needed):
            continue
        parts = {e: text_block(ctx[src], train) if e == "tfidf" else lm_block(out_dir, e, src, ids, train)
                 for e in encoders}
        blocks[src] = parts["tfidf"] if not suffix else \
            (np.concatenate([b for b, _ in parts.values()], axis=1), {e: s for e, (_, s) in parts.items()})
        logger.info("Text features for %s: %s", src, blocks[src][1])
    structured = structured_block(ctx, train)
    prior_block = None
    if any(ARMS[arm].prior_labels for arm in arms):
        if "prior_labels_found" not in ctx.columns:
            raise SystemExit(f"{ctx_path} has no prior-study labels: re-run `python -m v2.context` (M4b adds them)")
        prior_block = prior_label_block(ctx)
        logger.info("Prior-study labels for %.1f%% of studies", 100 * prior_block[:, -1].mean())

    prefix = args.run or Path(args.embeddings).stem
    features = "fine-tuned" if meta.get("finetune_run") else "frozen"
    image_meta = {**{k: meta.get(k) for k in ("encoder", "size", "revision", "torch", "transformers")},
                  "image_features": features}
    runs = {}
    for arm in arms:
        use_image, sources, use_struct, use_prior = ARMS[arm]
        absent = [s for s in sources if not available[s]]
        if use_prior and not prior_block[:, -1].any():
            absent.append("prior-label")
        if absent:
            logger.warning("Arm %s skipped: no %s input in the context", arm, ", ".join(absent))
            continue
        extra = [blocks[s][0] for s in sources] + ([structured] if use_struct else []) + \
            ([prior_block] if use_prior else [])
        # float32 throughout, as in M1: the `image` arm is then the M1 probe exactly.
        X = emb if not extra else np.concatenate(([emb] if use_image else []) + extra, axis=1).astype(np.float32)
        parts = split_xy(manifest, ids, X)
        (tr_rows, X_tr), (va_rows, X_va), (te_rows, X_te) = parts[TRAIN], parts[VAL], parts[TEST]
        probe = fit_probe(X_tr, label_matrix(tr_rows), X_va, label_matrix(va_rows), DEFAULT_CS, args.jobs)
        rows = pd.concat([va_rows, te_rows], ignore_index=True)
        run = f"{prefix}_ctx_{arm}{suffix}"
        milestone = "M6" if suffix else "M4b" if arm in M4B_ARMS else "M4"
        write_predictions(out_dir, run, rows["study_id"], rows["split"], probe.predict(np.concatenate([X_va, X_te])), {
            "kind": "fuse", "milestone": milestone, "arm": arm, "image": use_image, "text_encoder": args.text_encoder,
            "stack_tag": f"{arm}{suffix}", "sources": list(sources), "structured": use_struct, "prior_labels": use_prior,
            "embeddings": str(args.embeddings), "context": str(ctx_path),
            **(image_meta if use_image else {"encoder": None}),
            "text_features": {s: blocks[s][1] for s in sources}, "per_label": probe.summary(),
        })
        runs[arm] = run

    report = gate_report(out_dir, prefix, runs, args.n_boot, features, suffix)
    for flag in report["flags"]:
        logger.warning("RED FLAG: %s", flag)
    g4 = report["g4"]
    if not report["m4b"]:
        logger.info("%s (%s - %s, test_clean): %s -> %s", report["gate"], PRIMARY, REFERENCE,
                    fmt_ci(g4["diff"], g4["diff_ci"]) if g4 else "-", report["verdict"])
    if report["probe_drift"] is not None and not report["m4b"]:
        logger.info("image arm vs %s_probe: largest score difference %.2e", prefix, report["probe_drift"])
    for c in report["prior_label_cmps"]:
        logger.info("M4b %s - %s (%s, %s metric): %s", c["a"], c["b"], c["test_set"], c["metric"],
                    fmt_ci(c["diff"], c["diff_ci"]))
    logger.info("Report: %s", report["path"])


if __name__ == "__main__":
    main()
