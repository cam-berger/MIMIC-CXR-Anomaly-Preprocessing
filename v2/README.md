# v2: how to run it

v2 tests pretrained chest X-ray encoders (RAD-DINO, MedSigLIP, DINOv2) on 12
CheXpert findings in MIMIC-CXR, adds text written before the X-ray, lines the
results up with published models and checks them at another hospital. The
findings are in [`docs/ANALYSIS.md`](../docs/ANALYSIS.md). This file covers how
to run each step. A laptop with the data drive does most of it; fine-tuning
needs a rented GPU.

## 1. Set up

```bash
git clone <repo> && cd MIMIC-CXR-Anomaly-Preprocessing && git checkout v2
bash v2/scripts/setup.sh          # venv, install, unit tests, synthetic wiring check (~5 min)
source .venv-v2/bin/activate
```

You need Python 3.10 or newer. On macOS: `brew install python@3.11`, then
`PYTHON=python3.11 bash v2/scripts/setup.sh`. On a Linux machine without a GPU,
add `--cpu` for a smaller PyTorch download. On Windows, use WSL2.

Put your paths in `.env` at the repo root; every command reads it.

```bash
MIMIC_CXR_JPG_PATH=/Volumes/<drive>/mimic-cxr-jpg/2.1.0   # the folder with files/ and the CSVs
OUTPUT_PATH=./output
COHORT_DIR=./output/cohorts       # v1's anomalous_{train,val}.parquet (optional)
MIMIC_CXR_REPORTS_PATH=/Volumes/<drive>/mimic-cxr/2.1.0/mimic-cxr-reports.zip   # text steps
MIMIC_IV_ED_PATH=/Volumes/<drive>/mimic-iv-ed/2.2   # text steps: ed/edstays and ed/triage
MIMIC_IV_PATH=/Volumes/<drive>/mimiciv/3.1          # optional: hosp/patients, for age
CHEXPERT_TEST_PATH=/path/to/chexpert-test           # M7b: the CheXpert test images and groundtruth.csv
```

Then check the machine and the data: `python -m v2.doctor` (add
`--load-models` to download and test each encoder once).

Without v1's cohort files, leave `COHORT_DIR` unset: the population is then
every abnormal study and `test_v1` is empty. v1's `python build_cohort.py
--anomalous-only` rebuilds them.

## 2. Run the steps

`v2/scripts/run.sh` runs each step with a log and a time limit. Its settings
(`SIZE`, `MAX_TRAIN`, `FT_BATCH`/`FT_ACCUM`, `GRAD_CKPT`, `FT_HOURS`,
`STAGE_HOURS`, `TOTAL_HOURS`, `WORKERS`) are listed at its top. The default image
size is 518 px on CUDA and 224 px on Apple silicon or CPU.

**The v2 cohort** (`output/v2/`):

| Step | Command | Where | Time (measured) |
|---|---|---|---|
| M0 data and audit | `bash v2/scripts/run.sh m0` | laptop with the drive | 10-30 min |
| M1 frozen encoders | `SIZE=518 bash v2/scripts/run.sh m1 eval` | laptop with an Apple GPU | about 1 h per model |
| M2 zero-shot (optional) | `bash v2/scripts/run.sh m2 eval` | laptop or GPU | like M1 |
| M3 fine-tunes | `bash v2/scripts/run.sh smoke m3 eval` (§3) | rented GPU | 2.15 h on an A100 |
| M4 text before the X-ray | `SIZE=518 bash v2/scripts/run.sh m4 eval` | laptop | 15 min |
| M4b attention, prior labels, combiner | `SIZE=518 TOTAL_HOURS=8 bash v2/scripts/run.sh m4b eval` | laptop with an Apple GPU, 7 GB free | 71 min |

**The benchmark** (`BENCH_OUT`, default `output/v2_bench`), and the external sets:

| Step | Command | Where | Time (measured) |
|---|---|---|---|
| M5 the benchmark | `TOTAL_HOURS=20 bash v2/scripts/run.sh bench` | laptop with an Apple GPU and the drive, 65 GB free | 9 h to prepare from a USB drive, 6 h of embeddings |
| M5 fine-tunes | `TOTAL_HOURS=8 bash v2/scripts/run.sh bench_ft` (§3) | rented GPU | 102 min on a GH200 |
| M5 text on the fine-tunes | `bash v2/scripts/run.sh bench_stack` | laptop | 30 min |
| M6 language models for the text | `TOTAL_HOURS=16 bash v2/scripts/run.sh bench_text` | laptop with an Apple GPU, 8 GB free | 9.5 h |
| M7a a radiologist's labels | `bash v2/scripts/run.sh m7a` | laptop with the drive | 2 min |
| M7b the CheXpert test set | `bash v2/scripts/run.sh m7b` (writes to `EXT_OUT`, default `output/v2_chexpert`) | laptop with an Apple GPU | 6 min |
| M8 second seed, 6 epochs | `TOTAL_HOURS=8 bash v2/scripts/run.sh m8_gpu` (§3) | rented GPU | 201 min on a GH200 |
| M8 the comparison | `bash v2/scripts/run.sh m8` | laptop with the drive | 17 min |

`run.sh` stops at the first failure and skips the remaining steps after
`TOTAL_HOURS`. Run the same command again to resume: fine-tunes continue from
their last epoch and finished steps are kept.

**Fine-tuning on a laptop instead.** On Apple silicon or a local NVIDIA card,
at 224 px: `FT_HOURS=6 TOTAL_HOURS=24 bash v2/scripts/run.sh m3 eval`. Run
`smoke` first and read the images/s in its log to estimate the time. Compare
the results only with other 224 px runs.

## 3. The GPU session

1. Rent one GPU with 40 GB or more, CUDA 12 and 50 GB of disk (an A100, H100 or
   GH200).
2. On the GPU box:
   ```bash
   git clone <repo> && cd MIMIC-CXR-Anomaly-Preprocessing && git checkout v2
   bash v2/scripts/setup.sh && source .venv-v2/bin/activate
   python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'   # must print True
   ```
   Don't copy your `.env` there.
3. From your machine, upload the prepared images, not the raw JPGs. Plain
   `rsync -a` prints nothing per file, so no study IDs reach the console:
   ```bash
   ssh <gpu> 'mkdir -p MIMIC-CXR-Anomaly-Preprocessing/output/v2_bench'
   rsync -a output/v2_bench/manifest.parquet output/v2_bench/measurable_classes.json output/v2_bench/audit.json \
       output/v2_bench/images <gpu>:MIMIC-CXR-Anomaly-Preprocessing/output/v2_bench/
   ```
   For M3, upload `output/v2/` the same way.
4. On the GPU box, inside `tmux`: `run.sh smoke m3 eval` (M3), `run.sh
   bench_ft` (M5) or `run.sh m8_gpu` (M8). Each starts with a smoke run on a
   few real studies. Leave `FT_BATCH` and `FT_ACCUM` at their defaults unless
   memory forces it (`FT_BATCH=8 FT_ACCUM=4` keeps the same effective batch);
   M8 compares each run's settings with seed 0's.
5. Bring home the new runs only, then delete the data and the instance. For M8:
   ```bash
   ssh <gpu> 'cd MIMIC-CXR-Anomaly-Preprocessing/output && tar -czf m8.tgz v2_bench/predictions/*_ft_*s1.* \
       v2_bench/predictions/*_ft_attn_e6_s0.* v2_bench/runs/*_ft_*s1/best.pt v2_bench/runs/*_ft_*s1/*.json \
       v2_bench/runs/*_ft_attn_e6_s0/best.pt v2_bench/runs/*_ft_attn_e6_s0/*.json v2_bench/embeddings/*_ft_s1.npz v2/logs'
   scp <gpu>:MIMIC-CXR-Anomaly-Preprocessing/output/m8.tgz output/
   tar -xzf output/m8.tgz -C output && rm output/m8.tgz
   ssh <gpu> 'rm -rf MIMIC-CXR-Anomaly-Preprocessing/output'
   bash v2/scripts/run.sh m8
   ```
   For M5, take `predictions/*_ft_*s[0-9].*`, `runs/*_ft_*s[0-9]/{best.pt,*.json}`
   and `embeddings/*_ft_s[0-9].npz`, then run `bench_stack`. For M3, the same
   from `output/v2/`, then `python -m v2.evaluate --all`.

The MIMIC data stays under your PhysioNet credential: use a provider you trust,
and keep the data on the machine only for the run.

## 4. Outputs

Each output directory (`output/v2`, `BENCH_OUT`, `EXT_OUT`) has the same layout:

```
manifest.parquet           one row per study: ids, view, split, test-set flags, labels
images/                    one resized frontal image per study (input to every later step)
audit.md, audit.json       split sizes, label counts, drops, the RAD-DINO overlap check
measurable_classes.json    the findings scored on each test set, fixed at M0
embeddings/                frozen and fine-tuned image embeddings, patch tokens, text embeddings
predictions/{run}.parquet  one format for every model: val and test scores, plus {run}.json
runs/{run}/                fine-tunes: last.pt (resume), best.pt, history.json, config.json
context.parquet            the text written before each X-ray (v2 cohort and benchmark)
reports/                   one report per run, the step reports and summary.md
logs/                      one log per run.sh step (in output/v2)
```

The record-level files (manifest, images, embeddings, predictions,
`context.parquet`) stay local; see §8.

Runs are named `{model}_{size}_probe`, `_zeroshot`, `_ft_s{seed}` (mean pooling)
or `_ft_attn_s{seed}` (attention head), `{embeddings}_ctx_{arm}` (image and text
features in one classifier), `{model}_{size}_attn_{h0,h1,h2}` (attention heads
on frozen tokens), `{encoder}_ft_{source}_s{seed}` (a fine-tuned language model)
and `{image run}_stack_{text}` (image and text scores combined on val). Runs
ending in `_smoke` only check the setup.

## 5. Reading the results

`reports/summary.md` lists every run with its macro AUROC and 95% interval. The
step reports compare runs, paired:

| Report | What it answers |
|---|---|
| `audit.md` | How many studies and findings each split holds; whether RAD-DINO saw any test image |
| `reports/m4_rad-dino_518.md` | What each text source adds to the image |
| `reports/m4b_attention_rad-dino_518.md`, `m4b_stack.md` | Text-steered attention; the val combiner |
| `reports/lineup.md` (benchmark) | Our runs next to the published results, scored like them (positive vs rest) |
| `reports/m5_stack.md`, `m6_text.md` (benchmark) | Text on the fine-tuned image; the language models compared |
| `reports/m7a_radiologist.md` (benchmark) | Every run against a radiologist's labels |
| `reports/m7b_chexpert.md` (`EXT_OUT`) | The CheXpert test set, against CheXzero |
| `reports/m8_replication.md` (benchmark) | Each headline result with a second seed; 6 epochs against 3 |

Every report opens with its red flags: checks that must pass before the numbers
mean anything, such as text that gives the answer away or a model that isn't the
one scored before. Read them first.

To compare two runs: `python -m v2.evaluate --compare RUN_A RUN_B --test-set test_clean`.

## 6. Commands

Every step is a module with `--help`; all take `--out` (default `$OUTPUT_PATH/v2`).

```bash
python -m v2.prepare_data --mimic-cxr-jpg PATH [--cohort-dir DIR] [--limit N]
python -m v2.prepare_data --population all --split-scheme official --image-format jpg --out output/v2_bench
python -m v2.embed --model {rad-dino,dinov2-base,medsiglip,densenet121} --size 518 [--tokens 12]
python -m v2.probe --embeddings output/v2/embeddings/rad-dino_518.npz
python -m v2.zero_shot --model medsiglip
python -m v2.finetune --model rad-dino --size 518 --seed 0 [--head attention] [--resume] [--max-hours H]
python -m v2.context [--reports ZIP_OR_DIR] [--mimic-iv-ed DIR] [--mimic-iv DIR]
python -m v2.fuse --embeddings output/v2/embeddings/rad-dino_518.npz [--arms image,image_text] [--text-encoder bert]
python -m v2.attend --tokens output/v2/embeddings/rad-dino_518_tokens12.npy [--arms h0,h1,h2]
python -m v2.stack --pair IMAGE_RUN TEXT_RUN [--pair ...] [--report NAME]
python -m v2.lineup --out output/v2_bench
python -m v2.textenc --encoder bert --out output/v2_bench
python -m v2.textft --encoder bert --out output/v2_bench    # --report writes m6_text.md
python -m v2.external radiologist --bench output/v2_bench
python -m v2.external chexpert --chexpert DIR --bench output/v2_bench --out output/v2_chexpert
python -m v2.replicate --bench output/v2_bench --ext output/v2_chexpert
python -m v2.evaluate --all | --run RUN... | --compare RUN_A RUN_B
python -m v2.doctor [--load-models]
python -m v2.smoke
```

MedSigLIP is gated: accept its terms at <https://huggingface.co/google/medsiglip-448>,
then run `hf auth login`. `run.sh m2` skips itself without access.

## 7. Tests

```bash
pytest tests/v2 -q                                     # synthetic data and tiny random models, a few minutes
python -m v2.smoke                                     # the whole pipeline on synthetic data, ~2 min
V2_TEST_NETWORK=1 pytest tests/v2/test_real_models.py  # downloads the real encoders (~3 GB)
```

The wiring check (`v2.smoke`) builds a fake MIMIC-CXR tree with a planted signal
and runs every step through the same entry points as the real data. Each step
must recover the signal, and each built-in check (misaligned embeddings, leaked
text, a contaminated encoder, a mismatched recipe) must catch its planted fault.

Tested on two library stacks (torch 2.14.1 with the pinned transformers 5.20.1,
and torch 2.2.0 with transformers 4.38.0), on CPU Linux, Apple silicon (MPS),
a Lambda A100 and a Lambda GH200 (aarch64, the provider's torch 2.7.0, with
`numpy<2`, which `setup.sh` installs when needed).

## 8. Working on MIMIC data with an AI assistant

PhysioNet's credentialed data use agreement forbids sharing the data with third
parties, and its [LLM guidance](https://physionet.org/news/post/llm-responsible-use/)
(September 2025) counts sending it through an API or online service. A coding
assistant sends what it reads to its servers. So:

- Let the assistant run the v2 commands, but never open, print or list the
  dataset, the cohort files, `manifest.parquet`, `context.parquet`, embeddings
  (token caches and their `.json` sidecars included) or predictions.
- Share `audit.md` and `reports/*.md`: aggregate counts and metrics only.
  Console output is aggregate too; record-level diagnostics go to local files.
- Never commit `output/` (it's gitignored) or anything derived from a single
  study: images, crops, reconstructions, tensors, per-study features.
- File names hold study IDs. Check disk use with totals only:
  `du -sh DIR 2>/dev/null`, `find DIR -type f | wc -l`.

The CheXpert test set falls under the same rules.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor` reports slow JPG reads | Expected on USB drives. M0 reads each JPG once; later steps use the resized images. On a spinning disk, `WORKERS=2` can be faster |
| `prepare_data` stops with "is the drive still mounted?" | Over 0.5% of JPGs failed to read: the drive dropped off or the mount went stale. Remount it, keep the Mac awake (`caffeinate -ims`) and rerun; written images are kept |
| `prepare_data` says a directory "holds a manifest for population ..." | Point the benchmark at its own directory (`BENCH_OUT`) |
| CUDA out of memory | `FT_BATCH=8 FT_ACCUM=4` (same effective batch), or `GRAD_CKPT=1` |
| `RuntimeError: Numpy is not available` | The machine's torch was built for NumPy 1.x: `.venv-v2/bin/python -m pip install 'numpy<2'` (`setup.sh` does this when needed) |
| An op isn't implemented on MPS | v2 sets `PYTORCH_ENABLE_MPS_FALLBACK=1`; export it yourself if you import torch before v2 |
| DataLoader worker errors on macOS | `WORKERS=0` |
| "bad steps ... aborting" in a fine-tune | Non-finite loss or gradients in over 1% of steps; check the data and learning rate |
| `run.sh` stops with "transformers X, but v2 pins Y" | It found a Python other than the venv: run `setup.sh`, or set `PYTHON=` to the venv's python |
| `m4` or `bench_stack` needs the text sources | Set `MIMIC_CXR_REPORTS_PATH` (from the MIMIC-CXR project, not MIMIC-CXR-JPG) and `MIMIC_IV_ED_PATH`; `doctor` checks them |
| `bench_text` stopped by `TOTAL_HOURS` | Run it again: finished fine-tunes are kept |
| `m7b` finds no CheXpert test images | The images are in the CheXlocalize release on Stanford AIMI's Azure portal (download with azcopy; treat the link as a password). Put them, in any layout with `patient#/study#/view#_frontal.jpg`, and `groundtruth.csv` under `CHEXPERT_TEST_PATH` |
| `m8` asks to mount the MIMIC drive | M7a's radiologist labels (`mimic-cxr-2.1.0-test-set-labeled.csv`) are on it |
| A `RED FLAG` in any report | Stop and look before reading the numbers: the report says what failed. Inspect record-level files locally, without sharing them |
| A smoke step fails on the GPU | Don't start the long runs; read the error line |
| `token 0 differs from ... npz` in attend | The tokens and the embeddings came from different images or library versions: rerun `v2.embed --tokens` in the venv |
