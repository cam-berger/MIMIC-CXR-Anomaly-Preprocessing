"""
v3.3 (docs/V3_PLAN.md §8c): Jev as a report labeler, scored against a radiologist's labels.

    python -m v3.label run --split val --limit 40 --model MODEL    # the pilot: val reports, mechanics only
    python -m v3.label run --model MODEL                           # the radiologist's official-test studies
    python -m v3.label report                                      # -> reports/v3_label.md (gate L1)

Jev reads each study's own report (FINDINGS and IMPRESSION, v2.context.prior_text) and answers 12 three-way
questions. Its answers and the CheXpert labeler's labels on the same studies become v2 prediction files
(labeler_jev, labeler_jev_prob, labeler_chexpert), which v2.evaluate scores against the radiologist's labels with
M7a's manifest. The report is what gets labeled; it is never an input to an image model. Answers are cached in
jev_cache/label__{hash}__{model}.jsonl (record-level: keep local). Paths: --reports or MIMIC_CXR_REPORTS_PATH,
--labels or MIMIC_CXR_JPG_PATH/mimic-cxr-2.1.0-test-set-labeled.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v2.common import LABELS, TEST, label_matrix, load_dotenv, load_manifest, read_predictions, setup_logging
from v2.common import write_predictions
from v2.context import Reports, prior_text
from v2.evaluate import compare_runs, eval_rows, evaluate_set, fmt, fmt_ci
from v2.external import RADIOLOGIST_FILE, joint_classes, radiologist_manifests, read_radiologist

from . import prompt
from .common import default_out_dir
from .jev import JevClient, JevError
from .run import ALIASES, MAX_FAILED_SHARE, PILOT_SEED, load_cache, summarize

logger = logging.getLogger("v3.label")

LABEL_VERSION = 1
OPTIONS = ("present", "uncertain", "absent")
LEVEL = {"present": 1.0, "uncertain": 0.5, "absent": 0.0}  # the CheXpert labeler's 1 / -1 / 0-or-blank, as scores
RUNS = {"labeler_chexpert": "the CheXpert labeler (comparator)", "labeler_jev": "Jev's chosen option (L1)",
        "labeler_jev_prob": "Jev's probabilities (secondary)"}
RED_FLAG = 0.70  # §8c: the CheXpert labeler below this against the radiologist means misaligned studies or labels
PREAMBLE = "Findings and impression from the radiologist's report of a chest X-ray:"


def questions() -> dict:
    """12 three-way questions in the labeler's terms, keyed like the combiner's (prompt.question_name)."""
    return {
        prompt.question_name(label): {
            "type": "choice",
            "instructions": f"What does this report say about {prompt.NAMES[label]}?",
            "criteria": {
                "present": f"The report says it is present. {prompt.DEFINITIONS[label]}",
                "uncertain": "The report says it may be present: possible, probable, suggestive of, or cannot be "
                             "excluded.",
                "absent": "The report says it is absent, or does not mention it.",
            },
        }
        for label in LABELS
    }


def render_state(text: str) -> str:
    return f"{PREAMBLE}\n\n{text.strip() or prompt.MISSING}"


def label_hash() -> str:
    blob = json.dumps({"version": LABEL_VERSION, "questions": questions(), "state": render_state("REPORT")},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cache_path(out_dir: Path, model: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    return Path(out_dir) / "jev_cache" / f"label__{label_hash()}__{safe}.jsonl"


def resolve_paths(reports: Optional[str], labels: Optional[str], need_reports: bool = True,
                  need_labels: bool = True) -> tuple[Optional[Path], Optional[Path]]:
    """The reports and the radiologist's labels; each is required only where it is used (a val pilot needs no labels)."""
    load_dotenv()
    rep = reports or os.environ.get("MIMIC_CXR_REPORTS_PATH")
    lab = labels or (str(Path(os.environ["MIMIC_CXR_JPG_PATH"]) / RADIOLOGIST_FILE)
                     if os.environ.get("MIMIC_CXR_JPG_PATH") else None)
    if need_reports and (not rep or not Path(rep).exists()):
        raise SystemExit("the reports are not found: pass --reports or set MIMIC_CXR_REPORTS_PATH")
    if need_labels and (not lab or not Path(lab).exists()):
        raise SystemExit(f"the radiologist's labels are not found at {lab or '(no path)'}: connect the drive that "
                         f"holds MIMIC_CXR_JPG_PATH, or pass --labels with {RADIOLOGIST_FILE}'s location")
    return (Path(rep) if rep else None), (Path(lab) if lab else None)


def radiologist_set(out_dir: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """M7a's two manifests (the labeler's labels, the radiologist's), the jointly measurable classes, and counts."""
    rad, _ = read_radiologist(labels_path)
    labeler, reader, info = radiologist_manifests(out_dir, rad)
    if not info["in_official_test"]:
        raise SystemExit("none of the radiologist's studies is in the official test split")
    return labeler, reader, joint_classes(labeler, reader), info


def target_rows(out_dir: Path, split: str, labels_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """test: the radiologist's official-test studies; val: val studies (pilots only). One row per study."""
    if split == "test":
        _, reader, _, _ = radiologist_set(out_dir, labels_path)
        rows = eval_rows(reader, "test_clean")
    else:
        rows = eval_rows(load_manifest(out_dir), "val")
    rows = rows.drop_duplicates("study_id").sort_values("study_id")
    if limit is not None:
        rows = rows.sample(n=min(limit, len(rows)), random_state=PILOT_SEED).sort_values("study_id")
    return rows.reset_index(drop=True)


def report_texts(rows: pd.DataFrame, reports_path: Path) -> dict[int, str]:
    """Each study's own FINDINGS and IMPRESSION ("" where the report is missing)."""
    reports = Reports(reports_path)
    try:
        out = {}
        for sid, subject in zip(rows["study_id"], rows["subject_id"]):
            text = reports.get(int(subject), int(sid))
            out[int(sid)] = prior_text(text) if text else ""
        return out
    finally:
        reports.close()


def run(out_dir: Path, split: str, client: JevClient, reports_path: Path, labels_path: Path,
        limit: Optional[int] = None, concurrency: int = 4) -> dict:
    if split == "val" and limit is None:
        raise SystemExit("val is for pilots only (--limit): the radiologist's labels are on the official test")
    rows = target_rows(out_dir, split, labels_path, limit)
    path = cache_path(out_dir, client.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(path)
    todo = rows[~rows["study_id"].isin(list(cache))]
    logger.info("labeler on %s: %d studies, %d already answered, %d to ask (model %s, prompt %s)", split, len(rows),
                len(rows) - len(todo), len(todo), client.model, label_hash())
    qs, names = questions(), [prompt.question_name(label) for label in LABELS]
    texts = report_texts(todo, reports_path) if len(todo) else {}
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool, path.open("a") as f:
        futures = {pool.submit(client.ask, render_state(text), qs): sid for sid, text in texts.items()}
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                ans = fut.result()
            except JevError as exc:
                failures[sid] = str(exc.status or "network")
                continue
            rec = {"study_id": sid, "model": ans.model, "choices": [ans.scores[n]["choice"] for n in names],
                   "probs": [[ans.scores[n]["probabilities"][o] for o in OPTIONS] for n in names],
                   "scores": [LEVEL[ans.scores[n]["choice"]] for n in names],  # for run.summarize
                   "usage": ans.usage, "latency": round(ans.latency, 3), "empty_report": not texts[sid].strip()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cache[sid] = rec
    stats = summarize(rows, cache, failures)
    answered = [cache[s] for s in rows["study_id"] if s in cache]
    stats.update({"split": split, "requested_model": client.model, "prompt_hash": label_hash(),
                  "empty_reports": int(sum(r.get("empty_report", False) for r in answered)),
                  "choice_shares": {o: round(float(np.mean([c == o for r in answered for c in r["choices"]])), 3)
                                    for o in OPTIONS} if answered else {}})
    logger.info("labeler on %s: %d answered, %d failed %s; %d empty reports; input tokens %s (mean %s); latency "
                "median %s s; choices %s", split, stats["n_answered"], stats["n_failed"], stats["failed_by_status"] or "",
                stats["empty_reports"], stats["input_tokens"], stats["mean_input_tokens"], stats["latency_median"],
                stats["choice_shares"])
    if len({rec["model"] for rec in cache.values()}) > 1:
        raise SystemExit("labeler answers come from more than one model: §3 pins one")
    if stats["n_failed"] > MAX_FAILED_SHARE * len(rows):
        raise SystemExit(f"{stats['n_failed']} of {len(rows)} studies have no answer, more than "
                         f"{MAX_FAILED_SHARE:.0%}: stop (docs/V3_PLAN.md §6)")
    if limit is not None:
        stats["report"] = str(write_pilot(out_dir, stats))
    else:
        stats["predictions"] = write_label_predictions(out_dir, rows, cache, client.model)
    return stats


def write_label_predictions(out_dir: Path, rows: pd.DataFrame, cache: dict, model: str) -> list[str]:
    """
    The three §8c runs on the answered studies: Jev's choices, Jev's probabilities, and the CheXpert labeler's. The
    labeler's labels come from the benchmark's own manifest: `rows` may carry the radiologist's (M7a's manifest).
    """
    rows = rows[rows["study_id"].isin(list(cache))]
    ids, splits = rows["study_id"].to_numpy(), rows["split"].to_numpy()
    choice = np.array([cache[int(s)]["scores"] for s in ids], dtype=np.float64)
    probs = np.array([cache[int(s)]["probs"] for s in ids], dtype=np.float64)  # (n, 12, present/uncertain/absent)
    raw = label_matrix(load_manifest(out_dir).drop_duplicates("study_id").set_index("study_id").loc[ids].reset_index())
    chexpert = np.where(raw == 1.0, 1.0, np.where(raw == -1.0, 0.5, 0.0))
    meta = {"kind": "labeler", "milestone": "v3.3", "prompt_hash": label_hash(), "label_version": LABEL_VERSION,
            "requested_model": model, "model": next(iter({r["model"] for r in cache.values()}), None)}
    for run, scores in (("labeler_jev", choice), ("labeler_jev_prob", probs[:, :, 0] + 0.5 * probs[:, :, 1])):
        write_predictions(out_dir, run, ids, splits, scores, {**meta, "score": RUNS[run]})
    write_predictions(out_dir, "labeler_chexpert", ids, splits, chexpert,
                      {"kind": "labeler", "milestone": "v3.3", "score": RUNS["labeler_chexpert"]})
    return list(RUNS)


def write_pilot(out_dir: Path, stats: dict) -> Path:
    path = Path(out_dir) / "reports" / "v3_label_pilot.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    shares = ", ".join(f"{o} {v:.1%}" for o, v in stats["choice_shares"].items())
    rows = [("Studies", stats["n_studies"]), ("Answered", stats["n_answered"]),
            ("Failed", f"{stats['n_failed']} {stats['failed_by_status'] or ''}".strip()),
            ("Empty reports (no findings or impression)", stats["empty_reports"]),
            ("Model asked / answered", f"`{stats['requested_model']}` / {', '.join(f'`{m}`' for m in stats['models'])}"),
            ("Prompt hash", f"`{stats['prompt_hash']}`"),
            ("Input tokens, total / mean per study", f"{stats['input_tokens']:,} / {stats['mean_input_tokens']}"),
            ("Latency per request, median / p90 (s)", f"{stats['latency_median']} / {stats['latency_p90']}"),
            ("Jev's choices over all 12 findings", shares)]
    lines = [f"# v3.3 pilot: Jev as a report labeler, on {stats['split']}", "",
             "Aggregates only (docs/V3_PLAN.md §8c); no metric on a pilot.", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def build_report(out_dir: Path, labels_path: Path, n_boot: int = 1000) -> dict:
    labeler, reader, frozen, info = radiologist_set(out_dir, labels_path)
    for run in RUNS:
        if not (Path(out_dir) / "predictions" / f"{run}.parquet").exists():
            raise SystemExit(f"predictions/{run}.parquet not found: run `python -m v3.label run` first")
    preds = {run: read_predictions(Path(out_dir) / "predictions" / f"{run}.parquet")[0] for run in RUNS}
    scored = {run: {m: evaluate_set(p, reader, frozen, "test_clean", m, n_boot) for m in ("v2", "u0")}
              for run, p in preds.items()}
    agree = evaluate_set(preds["labeler_jev"], labeler, frozen, "test_clean", "v2", n_boot)  # Jev against the labeler
    cmp = {f"{a}:{m}": compare_runs(out_dir, a, "labeler_chexpert", "test_clean", m, n_boot, write=False,
                                manifest=reader, frozen=frozen)
           for a in ("labeler_jev", "labeler_jev_prob") for m in ("v2", "u0")}
    l1 = cmp["labeler_jev:v2"]
    verdict = ("n/a: no measurable classes" if not np.isfinite(l1["diff"]) else
               "PASS: Jev labels closer to the radiologist than the CheXpert labeler" if l1["a_better"] else
               "FAIL: the CheXpert labeler is closer to the radiologist" if l1["b_better"] else
               "no detectable difference between the two labelers (the CI includes 0)")
    flags = []
    base = scored["labeler_chexpert"]["v2"]
    if base and base["macro_auroc"] < RED_FLAG:
        flags.append(f"MISALIGNED: the CheXpert labeler scores {fmt(base['macro_auroc'])} against the radiologist, "
                     f"below {RED_FLAG}: stop and inspect locally (§8c)")
    answered = {run: int(p["split"].eq(TEST).sum()) for run, p in preds.items()}
    return {"info": info, "frozen": frozen, "scored": scored, "agree": agree, "cmp": cmp, "l1": l1,
            "verdict": verdict, "flags": flags, "answered": answered}


def write_report(out_dir: Path, r: dict) -> Path:
    def auc(res):
        return fmt_ci(res["macro_auroc"], res["macro_auroc_ci"]) if res else "-"

    info, l1 = r["info"], r["l1"]
    lines = ["# v3.3: Jev as a report labeler, against a radiologist", "",
             f"The radiologist's studies in the official test: {info['in_official_test']:,} studies, "
             f"{info['patients']:,} patients; {r['answered']['labeler_jev']:,} answered by Jev. Each labeler's "
             "three-level score (present 1, uncertain 0.5, absent 0) is scored by v2.evaluate against the "
             "radiologist's labels, over the classes measurable under both sets of labels (docs/V3_PLAN.md §8c).", "",
             f"**Gate L1:** `labeler_jev` minus `labeler_chexpert` {fmt_ci(l1['diff'], l1['diff_ci'])} "
             f"({len(l1['classes'])} classes, v2 policy) -> **{r['verdict']}**", "",
             "## Red flags", ""] + ([f"- {f}" for f in r["flags"]] or ["- none"]) + [""]
    lines += ["## Against the radiologist", "", "| Labeler | Macro AUROC (v2 policy) | Positive vs rest | Minus the "
              "CheXpert labeler (v2) | Minus the CheXpert labeler (u0) |", "|---|---|---|---|---|"]
    for run, role in RUNS.items():
        s = r["scored"][run]
        d = [r["cmp"].get(f"{run}:{m}") for m in ("v2", "u0")]
        lines.append(f"| `{run}`: {role} | {auc(s['v2'])} | {auc(s['u0'])} | "
                     + " | ".join(fmt_ci(c["diff"], c["diff_ci"]) if c else "-" for c in d) + " |")
    lines += ["", f"Jev against the CheXpert labeler's own labels (how often the two agree): {auc(r['agree'])}.", "",
              "## Per finding (v2 policy)", "",
              "| Finding | Radiologist positives | CheXpert labeler | Jev | Jev, probabilities |", "|---|---|---|---|---|"]
    per = {run: {c["label"]: c for c in r["scored"][run]["v2"]["per_class"]} for run in RUNS}
    for label in LABELS:
        if label not in r["frozen"]["test_clean:v2"]:
            continue
        c = per["labeler_chexpert"][label]
        lines.append(f"| {label} | {c['n_pos']:,} | " + " | ".join(fmt(per[run][label]["auroc"]) for run in RUNS) + " |")
    lines += ["", "The radiologist read the same reports as the labeler (M7a), so this measures how each labeler reads "
              "a report, not report-image disagreement."]
    path = Path(out_dir) / "reports" / "v3_label.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(r, indent=2, default=str))
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Ask Jev to label reports")
    p_run.add_argument("--split", choices=["test", "val"], default="test")
    p_run.add_argument("--limit", type=int, help="A pilot: this many studies, sampled with a fixed seed")
    p_run.add_argument("--model", required=True, help="A pinned Jev version")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_rep = sub.add_parser("report", help="Score the labelers against the radiologist (gate L1)")
    p_rep.add_argument("--n-boot", type=int, default=1000)
    for p in (p_run, p_rep):
        p.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
        p.add_argument("--reports", help="mimic-cxr-reports.zip or its folder (default: MIMIC_CXR_REPORTS_PATH)")
        p.add_argument("--labels", help=f"the radiologist's labels (default: MIMIC_CXR_JPG_PATH/{RADIOLOGIST_FILE})")
        p.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    out_dir = Path(args.out) if args.out else default_out_dir()
    if args.command == "run":
        if args.model in ALIASES:
            raise SystemExit(f"{args.model} is an alias that can change under a run: pin a version (§3)")
        reports, labels = resolve_paths(args.reports, args.labels, need_labels=args.split == "test")
        return run(out_dir, args.split, JevClient(args.model), reports, labels, args.limit, args.concurrency)
    labels = resolve_paths(args.reports, args.labels, need_reports=False)[1]
    result = build_report(out_dir, labels, args.n_boot)
    path = write_report(out_dir, result)
    logger.info("L1: %s -> %s", fmt_ci(result["l1"]["diff"], result["l1"]["diff_ci"]), result["verdict"])
    for flag in result["flags"]:
        logger.warning("%s", flag)
    logger.info("Report: %s", path)
    return {**result, "report": path}


if __name__ == "__main__":
    main()
