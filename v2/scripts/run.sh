#!/usr/bin/env bash
# Run v2 stages with logs and time limits (docs/V2_PLAN.md §6, §8).
#
#   bash v2/scripts/run.sh m0             # data prep + audit (your machine, data drive attached)
#   bash v2/scripts/run.sh m1 eval        # frozen embeddings + probes, then reports
#   bash v2/scripts/run.sh gpu            # the one GPU session: smoke m1 m2 m3 eval
#
# Stages:
#   m0     prepare_data + audit                          needs MIMIC_CXR_JPG_PATH (and COHORT_DIR)
#   smoke  real-data smoke test of embed + fine-tune     minutes; stops the run if anything fails
#   m1     rad-dino and dinov2-base embeddings + probes
#   m2     medsiglip embeddings + zero-shot + probe      skipped without Hugging Face access
#   m3     fine-tune rad-dino (seeds 0 and 1) + densenet121
#   m4     text written before the X-ray + frozen rad-dino embeddings (needs MIMIC_CXR_REPORTS_PATH, MIMIC_IV_ED_PATH)
#   m4b    prior-study labels, a combiner fit on val, then rad-dino patch tokens + attention heads (gate G4b);
#          needs m4's sources plus MIMIC_CXR_JPG_PATH, ~7 GB of disk, a few hours on a Mac (TOTAL_HOURS=8)
#   bench  M5a: all frontal studies, official split, in BENCH_OUT; RAD-DINO 518 px embeddings and tokens (one pass),
#          probe, M4b's h0/h1 heads, M4 fusion (if the M4 sources are set), reports/lineup.md and summary.md;
#          ~65 GB, ~12 h on a Mac (TOTAL_HOURS=20)
#   bench_ft  M5b on a GPU: fine-tune RAD-DINO 518 px on BENCH_OUT's train split for 3 epochs (BENCH_SEEDS, default 0)
#   eval   reports for every run + reports/summary.md with gates G1, G2, G4 and G4b
#   gpu    smoke m1 m2 m3 eval
#
# Settings (environment or .env):
#   MIMIC_CXR_JPG_PATH  MIMIC-CXR-JPG 2.1.0 root (m0 only)
#   COHORT_DIR          v1 cohort files (default: $OUTPUT_PATH/cohorts if it has anomalous_train.parquet)
#   MIMIC_CXR_REPORTS_PATH, MIMIC_IV_ED_PATH, MIMIC_IV_PATH (optional)   M4 sources (m4 only)
#   V2_OUT              output dir (default: $OUTPUT_PATH/v2, or output/v2)
#   BENCH_OUT           M5's benchmark output dir (default: $OUTPUT_PATH/v2_bench); never the same as V2_OUT
#   SIZE                encoder input size (default: 518 on CUDA, 224 on Apple MPS or CPU)
#   MAX_TRAIN           embed and probe only N training studies (CPU pilots); default: all
#   FT_BATCH, FT_ACCUM  fine-tune batch and accumulation (default 16 x 2; use 8 x 4 under 40 GB)
#   GRAD_CKPT=1         gradient checkpointing for fine-tuning (less memory, slower)
#   FT_HOURS            training-time budget per fine-tune; it then stops, keeps its best epoch and predicts (default 1.5)
#   STAGE_HOURS         hard time limit for every other stage (default 2); TOTAL_HOURS: skip stages after this (default 4)
#   WORKERS             data-loading workers (default: min(8, CPU cores))
set -euo pipefail
cd "$(dirname "$0")/../.."

# Check the arguments before doing anything slow.
[ $# -gt 0 ] || { sed -n '2,36p' "$0"; exit 2; }
stages=()
for arg in "$@"; do
  if [ "$arg" = gpu ]; then stages+=(smoke m1 m2 m3 eval); else stages+=("$arg"); fi
done
for stage in "${stages[@]}"; do
  case "$stage" in
    m0|smoke|m1|m2|m3|m4|m4b|bench|bench_ft|eval) ;;
    *) echo "Unknown stage: $stage (see the header of $0)" >&2; exit 2 ;;
  esac
done

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env || echo "warning: could not read .env" >&2
  set +a
fi
# Prefer the project venv, so an unactivated shell doesn't silently use another Python.
if [ -z "${PYTHON:-}" ] && [ -x .venv-v2/bin/python ]; then PY=.venv-v2/bin/python
else PY=${PYTHON:-$(command -v python || command -v python3)}; fi
OUTPUT_PATH=${OUTPUT_PATH:-output}
OUT=${V2_OUT:-$OUTPUT_PATH/v2}
BENCH_OUT=${BENCH_OUT:-$OUTPUT_PATH/v2_bench}
if [ -z "${COHORT_DIR:-}" ] && [ -f "$OUTPUT_PATH/cohorts/anomalous_train.parquet" ]; then
  COHORT_DIR=$OUTPUT_PATH/cohorts
fi
DEVICE=$("$PY" -c 'import torch; print("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))')
if [ "$DEVICE" = cuda ]; then DEFAULT_SIZE=518; DN_SIZE=512; else DEFAULT_SIZE=224; DN_SIZE=224; fi
SIZE=${SIZE:-$DEFAULT_SIZE}
WORKERS=${WORKERS:-$("$PY" -c 'import os; print(min(8, os.cpu_count() or 1))')}
FT_BATCH=${FT_BATCH:-16}
FT_ACCUM=${FT_ACCUM:-2}
FT_HOURS=${FT_HOURS:-1.5}
STAGE_HOURS=${STAGE_HOURS:-2}
TOTAL_HOURS=${TOTAL_HOURS:-4}
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG_DIR=$OUT/logs
mkdir -p "$LOG_DIR"
START=$(date +%s)
TIMEOUT=$(command -v timeout || command -v gtimeout || true)
echo "v2 run: device $DEVICE, size $SIZE, output $OUT, cohort ${COHORT_DIR:-none}, logs $LOG_DIR"
PINNED=$(sed -n 's/^transformers==\([0-9.]*\).*/\1/p' v2/requirements.txt)
FOUND=$("$PY" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo none)
echo "python $PY, transformers $FOUND"
if [ "$FOUND" != "$PINNED" ]; then
  echo "!!! transformers $FOUND, but v2 pins $PINNED: embeddings at non-native sizes depend on it." >&2
  echo "!!! Run bash v2/scripts/setup.sh, or set PYTHON to the venv's python." >&2
  [ "${ALLOW_UNPINNED:-0}" = 1 ] || exit 1
fi

run() {  # run [--hours H] NAME CMD...: logged and time-limited; any failure stops the script
  local hours=$STAGE_HOURS
  if [ "$1" = --hours ]; then hours=$2; shift 2; fi
  local name=$1
  shift
  if [ $(( $(date +%s) - START )) -gt "$(awk "BEGIN{print int($TOTAL_HOURS*3600)}")" ]; then
    echo "!!! TOTAL_HOURS ($TOTAL_HOURS h) reached: skipping $name"
    return 0
  fi
  local log
  log="$LOG_DIR/$(date +%Y%m%d-%H%M%S)_$name.log"
  echo ">>> $name: $*"
  local status=0
  if [ -n "$TIMEOUT" ]; then
    "$TIMEOUT" "${hours}h" "$@" 2>&1 | tee "$log" || status=$?
  else
    "$@" 2>&1 | tee "$log" || status=$?
  fi
  if [ "$status" -ne 0 ]; then
    if [ "$status" -eq 124 ]; then echo "!!! $name hit its ${hours} h time limit; log: $log"
    else echo "!!! $name failed (exit $status); log: $log"; fi
    exit "$status"
  fi
}
# Fine-tunes stop themselves after FT_HOURS (--max-hours) and then predict; the hard limit leaves room for that.
FT_LIMIT=$(awk "BEGIN{print $FT_HOURS + 0.5}")

ft_args=(--out "$OUT" --workers "$WORKERS" --batch-size "$FT_BATCH" --accum "$FT_ACCUM" --max-hours "$FT_HOURS" --resume)
[ "${GRAD_CKPT:-0}" = 1 ] && ft_args+=(--grad-checkpointing)
emb_args=(--out "$OUT" --workers "$WORKERS")
[ -n "${MAX_TRAIN:-}" ] && emb_args+=(--max-train "$MAX_TRAIN")

stage_m0() {
  : "${MIMIC_CXR_JPG_PATH:?set MIMIC_CXR_JPG_PATH to the MIMIC-CXR-JPG 2.1.0 directory}"
  run m0 "$PY" -m v2.prepare_data --mimic-cxr-jpg "$MIMIC_CXR_JPG_PATH" ${COHORT_DIR:+--cohort-dir "$COHORT_DIR"} \
    --out "$OUT" --workers "$WORKERS"
  echo "Audit: $OUT/audit.md"
}

stage_smoke() {  # every step on a few real studies; runs named *_smoke are left out of the summary
  run smoke_embed "$PY" -m v2.embed --model rad-dino --size "$SIZE" --smoke "${emb_args[@]}"
  run smoke_probe "$PY" -m v2.probe --embeddings "$OUT/embeddings/rad-dino_${SIZE}_smoke.npz" \
    --run "rad-dino_${SIZE}_probe_smoke" --out "$OUT"
  run smoke_finetune "$PY" -m v2.finetune --model rad-dino --size "$SIZE" --smoke --out "$OUT" --workers "$WORKERS"
  run smoke_evaluate "$PY" -m v2.evaluate --run "rad-dino_${SIZE}_probe_smoke" "rad-dino_${SIZE}_ft_s0_smoke" \
    --out "$OUT" --n-boot 50
}

stage_m1() {
  for model in rad-dino dinov2-base; do
    run "m1_embed_$model" "$PY" -m v2.embed --model "$model" --size "$SIZE" "${emb_args[@]}"
    run "m1_probe_$model" "$PY" -m v2.probe --embeddings "$OUT/embeddings/${model}_${SIZE}.npz" --out "$OUT"
  done
}

stage_m2() {
  if ! "$PY" -c 'from huggingface_hub import hf_hub_download as d; d("google/medsiglip-448", "config.json")' >/dev/null 2>&1; then
    echo "!!! m2 skipped: no access to google/medsiglip-448 (accept its terms on Hugging Face, then: hf auth login)"
    return 0
  fi
  run m2_embed "$PY" -m v2.embed --model medsiglip --size 448 "${emb_args[@]}"
  run m2_zero_shot "$PY" -m v2.zero_shot --model medsiglip --out "$OUT"
  run m2_probe "$PY" -m v2.probe --embeddings "$OUT/embeddings/medsiglip_448.npz" --out "$OUT"
}

stage_m3() {
  for seed in 0 1; do
    run --hours "$FT_LIMIT" "m3_rad-dino_s$seed" "$PY" -m v2.finetune --model rad-dino --size "$SIZE" --seed "$seed" \
      --save-embeddings "${ft_args[@]}"
  done
  run --hours "$FT_LIMIT" m3_densenet121 "$PY" -m v2.finetune --model densenet121 --size "$DN_SIZE" --seed 0 "${ft_args[@]}"
}

stage_m4() {  # the plan's M4 uses the 518 px embeddings: on a Mac, run it as SIZE=518 bash v2/scripts/run.sh m4 eval
  : "${MIMIC_CXR_REPORTS_PATH:?m4 needs MIMIC_CXR_REPORTS_PATH (mimic-cxr-reports.zip)}"
  : "${MIMIC_IV_ED_PATH:?m4 needs MIMIC_IV_ED_PATH (MIMIC-IV-ED 2.2)}"
  local emb="$OUT/embeddings/rad-dino_${SIZE}.npz"
  [ -f "$emb" ] || { echo "!!! m4 needs $emb (run m1 at SIZE=$SIZE first)" >&2; exit 1; }
  run m4_context "$PY" -m v2.context --out "$OUT"
  run m4_fuse "$PY" -m v2.fuse --embeddings "$emb" --out "$OUT"
}

stage_m4b() {  # docs/V2_PLAN.md §6 M4b, on the 518 px embeddings: on a Mac, SIZE=518 TOTAL_HOURS=8 bash v2/scripts/run.sh m4b eval
  : "${MIMIC_CXR_REPORTS_PATH:?m4b needs MIMIC_CXR_REPORTS_PATH (it may rebuild context.parquet)}"
  : "${MIMIC_IV_ED_PATH:?m4b needs MIMIC_IV_ED_PATH (it may rebuild context.parquet)}"
  : "${MIMIC_CXR_JPG_PATH:?m4b needs MIMIC_CXR_JPG_PATH (prior-study labels)}"
  local emb="$OUT/embeddings/rad-dino_${SIZE}.npz" pre="rad-dino_${SIZE}"
  [ -f "$emb" ] || { echo "!!! m4b needs $emb (run m1 at SIZE=$SIZE first)" >&2; exit 1; }
  for run in "${pre}_probe" "${pre}_ctx_image" "${pre}_ctx_image_prior" "${pre}_ctx_image_text" "${pre}_ctx_text"; do
    [ -f "$OUT/predictions/$run.parquet" ] || { echo "!!! m4b needs predictions/$run.parquet (run m4 first)" >&2; exit 1; }
  done
  # Rebuild context.parquet only if it predates M4b; the schema check reads no rows.
  if ! "$PY" -c 'import sys, pyarrow.parquet as pq; sys.exit("prior_labels_found" not in pq.read_schema(sys.argv[1]).names)' \
      "$OUT/context.parquet" 2>/dev/null; then
    run m4b_context "$PY" -m v2.context --out "$OUT"
  fi
  run m4b_prior_labels "$PY" -m v2.fuse --embeddings "$emb" --out "$OUT" \
    --arms prior_labels,image_prior_labels,image_text_prior_labels
  local pairs=(--pair "${pre}_probe" "${pre}_ctx_text")
  [ -f "$OUT/predictions/${pre}_ft_s1.parquet" ] && pairs=(--pair "${pre}_ft_s1" "${pre}_ctx_text" "${pairs[@]}")
  run m4b_stack "$PY" -m v2.stack "${pairs[@]}" --out "$OUT"
  run --hours 4 m4b_tokens "$PY" -m v2.embed --model rad-dino --size "$SIZE" --tokens 12 "${emb_args[@]}"
  run --hours 6 m4b_attend "$PY" -m v2.attend --tokens "$OUT/embeddings/${pre}_tokens12.npy" --out "$OUT"
}

bench_dir_ok() {  # M5 writes its own manifest: never into v2.0's output dir
  if [ "$(cd "$(dirname "$BENCH_OUT")" 2>/dev/null && pwd)/$(basename "$BENCH_OUT")" = \
       "$(cd "$(dirname "$OUT")" 2>/dev/null && pwd)/$(basename "$OUT")" ]; then
    echo "!!! BENCH_OUT ($BENCH_OUT) must differ from the v2 output dir ($OUT)" >&2; exit 1
  fi
}

stage_bench() {  # docs/V2_PLAN.md §6 M5a. On a Mac: TOTAL_HOURS=20 bash v2/scripts/run.sh bench
  : "${MIMIC_CXR_JPG_PATH:?bench needs MIMIC_CXR_JPG_PATH}"
  bench_dir_ok
  local emb="$BENCH_OUT/embeddings/rad-dino_518.npz"
  run --hours 6 bench_prepare "$PY" -m v2.prepare_data --mimic-cxr-jpg "$MIMIC_CXR_JPG_PATH" --population all \
    --split-scheme official --image-format jpg --out "$BENCH_OUT" --workers "$WORKERS"
  # One pass writes the pooled embeddings and the 12 x 12 token cache (§6 M5 amendment).
  run --hours 12 bench_embed "$PY" -m v2.embed --model rad-dino --size 518 --tokens 12 --with-pooled \
    --out "$BENCH_OUT" --workers "$WORKERS"
  run --hours 3 bench_probe "$PY" -m v2.probe --embeddings "$emb" --out "$BENCH_OUT"
  local heads=h0
  if [ -n "${MIMIC_CXR_REPORTS_PATH:-}" ] && [ -n "${MIMIC_IV_ED_PATH:-}" ]; then
    run bench_context "$PY" -m v2.context --out "$BENCH_OUT"
    run --hours 6 bench_fuse "$PY" -m v2.fuse --embeddings "$emb" --out "$BENCH_OUT" \
      --arms image,image_indication,image_text,indication,text
    heads=h0,h1
  else
    echo "!!! bench: MIMIC_CXR_REPORTS_PATH or MIMIC_IV_ED_PATH not set: image-only lineup"
  fi
  run --hours 6 bench_attend "$PY" -m v2.attend --tokens "$BENCH_OUT/embeddings/rad-dino_518_tokens12.npy" \
    --out "$BENCH_OUT" --arms "$heads"
  run bench_lineup "$PY" -m v2.lineup --out "$BENCH_OUT"
  run bench_eval "$PY" -m v2.evaluate --all --out "$BENCH_OUT"
  echo "Lineup: $BENCH_OUT/reports/lineup.md; results-log rows: $BENCH_OUT/reports/summary.md"
}

stage_bench_ft() {  # docs/V2_PLAN.md §6 M5b, on a GPU, with BENCH_OUT's manifest and images copied over
  bench_dir_ok
  [ -f "$BENCH_OUT/manifest.parquet" ] || { echo "!!! bench_ft needs $BENCH_OUT from the bench stage" >&2; exit 1; }
  local hours=${BENCH_FT_HOURS:-3}
  for seed in ${BENCH_SEEDS:-0}; do
    run --hours "$(awk "BEGIN{print $hours + 0.5}")" "bench_ft_s$seed" "$PY" -m v2.finetune --model rad-dino --size 518 \
      --seed "$seed" --epochs 3 --save-embeddings --out "$BENCH_OUT" --workers "$WORKERS" --batch-size "$FT_BATCH" \
      --accum "$FT_ACCUM" --max-hours "$hours" --resume
  done
}

stage_eval() {
  run eval "$PY" -m v2.evaluate --all --out "$OUT"
  echo "Summary: $OUT/reports/summary.md"
}

for stage in "${stages[@]}"; do
  "stage_$stage"
done
echo "Done in $(( ($(date +%s) - START) / 60 )) min."
