"""
Ask Jev about every study of a split, for one arm (docs/V3_PLAN.md §3-4, §7).

    python -m v3.run --arm jev_est_text --split val --limit 40 --model MODEL   # the pilot
    python -m v3.run --arm jev_est_text --split val --model MODEL
    python -m v3.run --arm jev_est_text --split test --model MODEL             # after val, same prompt and model

Answers are cached in V3_OUT/jev_cache/{arm}__{prompt hash}__{model}.jsonl, which is record-level: keep it local and
never print it. A re-run asks only about studies without an answer, so nothing is paid for twice. Without --limit,
the run then writes predictions/{arm}.parquet from every cached answer (val, and test once it is done). With
--limit (the pilot) it writes reports/v3_pilot_{arm}.md instead. Console output is counts, tokens and timings only;
the studies that failed go to logs/, locally.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from v2.common import LABELS, TEST, VAL, load_manifest, read_predictions, setup_logging, write_predictions
from v2.evaluate import eval_rows

from . import prompt
from .common import default_out_dir, load_setup
from .jev import JevClient, JevError

logger = logging.getLogger("v3.run")

SPLITS = {"val": "val", "test": "test_clean"}  # v2.evaluate's names; on the benchmark test_clean is the official test
MAX_FAILED_SHARE = 0.01                       # §6: more than this share of a split unanswered stops the run
ALIASES = ("jev-latest", "jev-preview")        # §3: a run pins a version
PILOT_SEED = 0


def cache_path(out_dir: Path, arm: str, model: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
    return Path(out_dir) / "jev_cache" / f"{arm}__{prompt.prompt_hash(arm)}__{safe}.jsonl"


def load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            records[int(rec["study_id"])] = rec
    return records


def split_rows(out_dir: Path, split: str) -> pd.DataFrame:
    manifest = load_manifest(out_dir).drop_duplicates("study_id")
    return eval_rows(manifest, SPLITS[split]).sort_values("study_id").reset_index(drop=True)


def stage1_probabilities(out_dir: Path, image_run: str) -> pd.DataFrame:
    """The image run's 12 scores as probabilities (v2 runs write logits), indexed by study_id."""
    wide, _ = read_predictions(Path(out_dir) / "predictions" / f"{image_run}.parquet")
    probs = 1.0 / (1.0 + np.exp(-wide[list(LABELS)].to_numpy(np.float64)))
    return pd.DataFrame(probs, index=wide["study_id"].to_numpy(), columns=list(LABELS))


def study_states(out_dir: Path, arm: str, rows: pd.DataFrame, image_run: str) -> list[tuple[int, str, Optional[Path]]]:
    """(study_id, state, image path or None) for each row."""
    ctx = pd.read_parquet(Path(out_dir) / "context.parquet", columns=["study_id", *prompt.TEXT_BLOCKS])
    ctx = ctx.drop_duplicates("study_id").set_index("study_id")
    absent = np.setdiff1d(rows["study_id"], ctx.index)
    if len(absent):
        raise SystemExit(f"context.parquet lacks {len(absent)} of the split's studies: re-run v2.context")
    probs = stage1_probabilities(out_dir, image_run) if prompt.needs_estimates(arm) else None
    if probs is not None:
        absent = np.setdiff1d(rows["study_id"], probs.index)
        if len(absent):
            raise SystemExit(f"{image_run} has no scores for {len(absent)} of the split's studies")
    out = []
    for _, row in rows.iterrows():
        sid = int(row["study_id"])
        est = probs.loc[sid].to_numpy() if probs is not None else None
        image = Path(out_dir) / row["image"] if prompt.needs_image(arm) else None
        out.append((sid, prompt.render_state(arm, ctx.loc[sid].to_dict(), est), image))
    return out


def val_done(out_dir: Path, arm: str, model: str) -> bool:
    """A full val run of this arm wrote predictions with the current prompt and model."""
    sidecar = Path(out_dir) / "predictions" / f"{arm}.json"
    if not sidecar.exists():
        return False
    meta = json.loads(sidecar.read_text())
    return meta.get("prompt_hash") == prompt.prompt_hash(arm) and meta.get("requested_model") == model


def _ask(client: JevClient, state: str, questions: dict, image: Optional[Path]):
    return client.ask(state, questions, image.read_bytes() if image is not None else None)


def run_arm(out_dir: Path, arm: str, split: str, client: JevClient, limit: Optional[int] = None,
            concurrency: int = 4, allow_image: bool = False) -> dict:
    if arm not in prompt.ARMS:
        raise SystemExit(f"unknown arm {arm!r}: one of {', '.join(prompt.ARMS)}")
    if prompt.needs_image(arm) and not allow_image:
        raise SystemExit(f"{arm} sends the X-ray itself (v3.1, docs/V3_PLAN.md §8): pass --allow-image once its "
                         "amendment is written")
    setup = load_setup(out_dir)
    path = cache_path(out_dir, arm, client.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(path)
    if split == "test" and not val_done(out_dir, arm, client.model):
        raise SystemExit(f"{arm}: run val first with the same prompt and model (§7): test runs once, after val")
    rows = split_rows(out_dir, split)
    if limit is not None:
        rows = rows.sample(n=min(limit, len(rows)), random_state=PILOT_SEED).sort_values("study_id")
    todo = rows[~rows["study_id"].isin(list(cache))]
    logger.info("%s on %s: %d studies, %d already answered, %d to ask (model %s, prompt %s)", arm, split, len(rows),
                len(rows) - len(todo), len(todo), client.model, prompt.prompt_hash(arm))
    questions = prompt.questions()
    names = [prompt.question_name(label) for label in LABELS]
    failures: dict[int, str] = {}
    if len(todo):
        states = study_states(out_dir, arm, todo, setup["image_run"])
        with ThreadPoolExecutor(max_workers=concurrency) as pool, path.open("a") as f:
            futures = {pool.submit(_ask, client, state, questions, image): sid for sid, state, image in states}
            for done, fut in enumerate(as_completed(futures), 1):
                sid = futures[fut]
                try:
                    ans = fut.result()
                except JevError as exc:
                    failures[sid] = str(exc.status or "network")
                    continue
                rec = {"study_id": sid, "model": ans.model, "scores": [ans.scores[n] for n in names],
                       "usage": ans.usage, "latency": round(ans.latency, 3)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                cache[sid] = rec
                if done % 200 == 0:
                    logger.info("%s: %d of %d asked (%d failed)", arm, done, len(todo), len(failures))
    stats = summarize(rows, cache, failures)
    stats.update({"arm": arm, "split": split, "requested_model": client.model, "prompt_hash": prompt.prompt_hash(arm),
                  "cache": path.name})
    log_failures(out_dir, arm, split, failures)
    logger.info("%s on %s: %d answered, %d failed %s; input tokens %s (mean %s); latency median %s s, p90 %s s",
                arm, split, stats["n_answered"], stats["n_failed"], stats["failed_by_status"] or "",
                stats["input_tokens"], stats["mean_input_tokens"], stats["latency_median"], stats["latency_p90"])
    models = {rec["model"] for rec in cache.values()}
    if len(models) > 1:
        raise SystemExit(f"{arm}: answers come from {len(models)} models ({', '.join(sorted(models))}): §3 pins one")
    if stats["n_failed"] > MAX_FAILED_SHARE * len(rows):
        raise SystemExit(f"{arm} on {split}: {stats['n_failed']} of {len(rows)} studies have no answer, more than "
                         f"{MAX_FAILED_SHARE:.0%}: stop (docs/V3_PLAN.md §6)")
    if limit is not None:
        stats["report"] = str(write_pilot_report(out_dir, stats))
    else:
        stats["predictions"] = str(write_arm_predictions(out_dir, arm, cache, setup, client.model, path))
    return stats


def summarize(rows: pd.DataFrame, cache: dict, failures: dict) -> dict:
    answered = [cache[sid] for sid in rows["study_id"] if sid in cache]
    tokens = [rec["usage"].get("input_tokens", 0) for rec in answered]
    latency = np.array([rec["latency"] for rec in answered]) if answered else np.array([np.nan])
    scores = np.array([rec["scores"] for rec in answered]) if answered else np.zeros((0, len(LABELS)))
    return {
        "n_studies": int(len(rows)), "n_answered": len(answered), "n_failed": len(failures),
        "failed_by_status": {k: int(v) for k, v in pd.Series(list(failures.values()), dtype=object).value_counts().items()},
        "input_tokens": int(sum(tokens)), "mean_input_tokens": round(float(np.mean(tokens)), 1) if tokens else None,
        "latency_median": round(float(np.nanmedian(latency)), 2), "latency_p90": round(float(np.nanpercentile(latency, 90)), 2),
        "models": sorted({rec["model"] for rec in answered}),
        "distinct_scores_median": float(np.median([len(np.unique(scores[:, k])) for k in range(len(LABELS))]))
        if len(scores) else None,
    }


def log_failures(out_dir: Path, arm: str, split: str, failures: dict) -> None:
    """Record-level: which studies failed, for a local re-run. Never printed."""
    path = Path(out_dir) / "logs" / f"{arm}_{split}_failures.csv"
    if failures:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"study_id": list(failures), "status": list(failures.values())}).to_csv(path, index=False)
    elif path.exists():
        path.unlink()


def write_arm_predictions(out_dir: Path, arm: str, cache: dict, setup: dict, model: str, path: Path) -> Path:
    manifest = load_manifest(out_dir).drop_duplicates("study_id")
    rows = pd.concat([eval_rows(manifest, name) for name in SPLITS.values()]).drop_duplicates("study_id")
    rows = rows[rows["study_id"].isin(list(cache))].sort_values("study_id")
    scores = np.array([cache[int(sid)]["scores"] for sid in rows["study_id"]], dtype=np.float64)
    expected = {split: len(eval_rows(manifest, name)) for split, name in SPLITS.items()}
    answered = {VAL: int((rows["split"] == VAL).sum()), TEST: int((rows["split"] == TEST).sum())}
    return write_predictions(out_dir, arm, rows["study_id"], rows["split"], scores, {
        "kind": "jev", "milestone": "v3", "arm": arm, "blocks": list(prompt.ARMS[arm]),
        "prompt_version": prompt.PROMPT_VERSION, "prompt_hash": prompt.prompt_hash(arm),
        "requested_model": model, "model": next(iter({rec["model"] for rec in cache.values()}), None),
        "image_run": setup["image_run"] if prompt.needs_estimates(arm) else None, "cache": path.name,
        "studies_expected": expected, "studies_answered": answered,
        "input_tokens": int(sum(rec["usage"].get("input_tokens", 0) for rec in cache.values())),
    })


def write_pilot_report(out_dir: Path, stats: dict) -> Path:
    path = Path(out_dir) / "reports" / f"v3_pilot_{stats['arm']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [("Studies", stats["n_studies"]), ("Answered", stats["n_answered"]),
            ("Failed", f"{stats['n_failed']} {stats['failed_by_status'] or ''}".strip()),
            ("Model asked / answered", f"`{stats['requested_model']}` / {', '.join(f'`{m}`' for m in stats['models'])}"),
            ("Prompt hash", f"`{stats['prompt_hash']}`"),
            ("Input tokens, total / mean per study", f"{stats['input_tokens']:,} / {stats['mean_input_tokens']}"),
            ("Latency per request, median / p90 (s)", f"{stats['latency_median']} / {stats['latency_p90']}"),
            ("Distinct scores per finding (median of 12)", stats["distinct_scores_median"])]
    lines = [f"# v3 pilot: `{stats['arm']}` on {stats['split']}", "",
             "Aggregates only (docs/V3_PLAN.md §7 step 3).", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=list(prompt.ARMS))
    parser.add_argument("--split", required=True, choices=list(SPLITS))
    parser.add_argument("--model", required=True, help="A pinned Jev version (see `python -m v3.jev models`)")
    parser.add_argument("--out", default=None, help="v3 output dir (default: V3_OUT or $OUTPUT_PATH/v3)")
    parser.add_argument("--limit", type=int, help="The pilot: this many studies, sampled with a fixed seed")
    parser.add_argument("--concurrency", type=int, default=4, help="Requests in flight at once")
    parser.add_argument("--long-context", action="store_true", help="Send TypeSafe's long-context header")
    parser.add_argument("--allow-image", action="store_true", help="v3.1 only: allow an arm that sends the X-ray")
    parser.add_argument("--allow-alias", action="store_true", help="Allow jev-latest / jev-preview (not for results)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.model in ALIASES and not args.allow_alias:
        raise SystemExit(f"{args.model} is an alias that can change under a run: pin a version (§3)")
    client = JevClient(args.model, long_context=args.long_context)
    return run_arm(Path(args.out) if args.out else default_out_dir(), args.arm, args.split, client, args.limit,
                   args.concurrency, args.allow_image)


if __name__ == "__main__":
    main()
