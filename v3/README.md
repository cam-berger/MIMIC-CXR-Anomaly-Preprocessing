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
python -m v3.jev models                             # lists aliases only; jev-1.14.0 is pinned (§9)
python -m v3.smoke --live --model MODEL             # the real API on 12 synthetic studies per format
python -m v3.run --arm jev_est_text --format score --split val --limit 40 --model MODEL   # a pilot
```

Each question format is a run of its own: `--format noul` (yes/no) or `--format score` (10 likelihood bands),
named `{arm}_{format}`. Val picks the format (§7 step 4): `jev_est_text` on all of val in both, then the report.

```bash
for fmt in noul score; do python -m v3.run --arm jev_est_text --format $fmt --split val --model MODEL; done
python -m v3.report                                 # "Question format (§7, val): ... -> FORMAT"
```

Then the other arms on val in that format, the report (the val prompt rule, §7 step 5), and only then the
official test. `v3.run` refuses test in the format val didn't choose.

```bash
for arm in jev_est jev_est_indication jev_text; do
  python -m v3.run --arm $arm --format FORMAT --split val --model MODEL
done
python -m v3.report
for arm in jev_est jev_est_indication jev_est_text jev_text; do
  python -m v3.run --arm $arm --format FORMAT --split test --model MODEL
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

## 5. The X-ray itself (v3.1, docs/V3_PLAN.md §8)

The three image arms (`jev_img`, `jev_img_est`, `jev_img_est_text`) put the prepared frontal JPEG
(`BENCH_OUT/images/{study_id}.jpg`, 518 px on the shorter side) in each request as a multipart image asset, behind
TypeSafe's `TypeSafe-Danger-Zone-Images` header. They run only with `--allow-image`, only in the format val chose in
v3.0, and only after every image passes a local size check (Jev rejects images under 384 px or over about 1100 px).

```bash
python -m v3.smoke --live-image --model MODEL        # synthetic images: does Jev see them (no MIMIC data)
python -m v3.run --arm jev_img_est_text --format noul --split val --limit 40 --model MODEL --allow-image   # pilot
for arm in jev_img jev_img_est jev_img_est_text; do
  python -m v3.run --arm $arm --format noul --split val --model MODEL --allow-image
done
for arm in jev_img jev_img_est jev_img_est_text; do
  python -m v3.run --arm $arm --format noul --split test --model MODEL --allow-image
done
python -m v3.report                                  # gate J2 and the v3.1 ladder
```

Further context sources become new blocks and arms in `v3/prompt.py`.

## 6. Jev as a report labeler (v3.3, docs/V3_PLAN.md §8c)

Jev reads each study's own report (findings and impression) and labels the 12 findings as present, uncertain or
absent. v2's evaluation scores its labels and the CheXpert labeler's against a radiologist's, on the 596 official-test
studies the radiologist labeled (gate L1). The report is what gets labeled; it never becomes a model input. The
official-test run needs `mimic-cxr-2.1.0-test-set-labeled.csv` in `MIMIC_CXR_JPG_PATH` (or `--labels`); the val pilot
needs only the reports.

```bash
python -m v3.smoke --live-label --model MODEL                  # synthetic reports (no MIMIC data)
python -m v3.label run --split val --limit 40 --model MODEL    # pilot -> reports/v3_label_pilot.md
python -m v3.label run --model MODEL                           # the radiologist's 596 studies
python -m v3.label report                                      # -> reports/v3_label.md
```

## 7. Jev's labels as training targets (v3.5, docs/V3_PLAN.md §8e)

Jev labels 20,000 training reports and all of val; v2's probe is then trained three times on the same studies and
frozen RAD-DINO features (CheXpert labels, Jev's answers, Jev's probabilities as soft targets) and judged by the
radiologist (gate L2). The first two commands send report text to TypeSafe and are cached and resumable; the third
runs locally.

```bash
python -m v3.label run --split train --limit 20000 --model MODEL
python -m v3.label run --split val --model MODEL
python -m v3.downstream --model MODEL --labels output/mimic-cxr-2.1.0-test-set-labeled.csv   # -> reports/v3_downstream.md
```
