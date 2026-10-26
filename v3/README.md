# v3: Jev as the combiner

v3 gives TypeSafe's Jev the 12 estimates of v2's image model plus the text written before the X-ray, asks 12
yes/no questions per study, and scores the answers with v2's evaluation. The question, the arms, gate J1 and the
stop rules are fixed in [`docs/V3_PLAN.md`](../docs/V3_PLAN.md). v3 reuses v2's package and its virtual
environment (`.venv-v2`); it adds no dependencies.

## 1. Set up

v3 needs M5's benchmark (`BENCH_OUT`, built by v2's `run.sh bench`, `bench_ft` and `bench_stack`) and a TypeSafe
key. Add the key to `.env`, which is gitignored:

```bash
TYPESAFE_API_KEY=...
```

Then link the benchmark into v3's own output dir, `V3_OUT` (default `$OUTPUT_PATH/v3`). This also fits the val
combiner on all three text sources, the baseline for gate J1:

```bash
python -m v3.setup
```

Nothing in `BENCH_OUT` is written.

## 2. Run (docs/V3_PLAN.md §7)

```bash
python -m pytest tests/v3 -q                        # offline, no data, no network
python -m v3.smoke                                  # offline: synthetic benchmark and a fake Jev
python -m v3.jev models                             # pick a pinned version (no MIMIC data is sent)
python -m v3.smoke --live --model MODEL             # the real API on 12 synthetic studies
python -m v3.run --arm jev_est_text --split val --limit 40 --model MODEL   # the pilot -> reports/v3_pilot_*.md
```

Then each v3.0 arm on val, the report (the val prompt rule, §7 step 4), and only then the official test:

```bash
for arm in jev_est jev_est_indication jev_est_text jev_text; do
  python -m v3.run --arm $arm --split val --model MODEL
done
python -m v3.report
for arm in jev_est jev_est_indication jev_est_text jev_text; do
  python -m v3.run --arm $arm --split test --model MODEL
done
python -m v3.report                                 # -> V3_OUT/reports/v3_report.md
```

`--concurrency` sets the number of requests in flight (default 4). A run that stops (a crash, a rate limit,
Ctrl-C) loses nothing: answers are cached per (arm, prompt, model), and a re-run asks only about the studies
without one. If more than 1% of a split's studies have no answer after retries, the run stops without writing
predictions (§6). The failed study IDs go to `V3_OUT/logs/`, and a re-run retries only those.

## 3. Outputs (`V3_OUT`)

| Path | What | Share? |
|---|---|---|
| `reports/v3_report.md` | Gate J1, the context ladder, red flags, every run's macro AUROC | yes (aggregate) |
| `reports/v3_pilot_*.md`, `reports/v3_stack.md`, `reports/{run}.md` | Pilot, val combiners, per run | yes (aggregate) |
| `predictions/jev_*.parquet` | v2's prediction format, one row per study and finding | **no** (record-level) |
| `jev_cache/*.jsonl` | Jev's answers per study | **no** (record-level) |
| `logs/*_failures.csv` | Study IDs without an answer | **no** (record-level) |
| `manifest.parquet`, `context.parquet`, `images`, v2 runs | Links into `BENCH_OUT` | **no** |

## 4. MIMIC data

v3 sends each study's pre-imaging text and its image-model estimates to TypeSafe's API. The user verified on
2026-10-26 that TypeSafe keeps no data (`docs/V3_PLAN.md` §3). Everything in `v2/README.md` §9 still holds: an
AI assistant runs the commands and reads only the aggregate reports, never the cache, the predictions or the
context table. v3's console output is counts, tokens and timings only. The client's errors carry the HTTP status
and the API's validation messages, without the request's text or any input the API echoes back.

## 5. Adding the image (v3.1)

`jev_img_est_text` puts the prepared frontal JPEG (`BENCH_OUT/images/{study_id}.jpg`) in the request as an image
asset, behind TypeSafe's `TypeSafe-Danger-Zone-Images` header. It refuses to run without `--allow-image`, which
waits for the v3.1 amendment to `docs/V3_PLAN.md` (§8). Further context sources become new blocks and arms in
`v3/prompt.py`.
