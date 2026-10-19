# v2: pretrained chest X-ray encoders on the 12-class task

v2 measures what modern pretrained models (RAD-DINO, MedSigLIP, DINOv2) do on
v1's 12 CheXpert classes. It uses images only, under a fixed evaluation
protocol, and runs from a laptop with the data drive or a rented GPU. The plan,
task definition and decision gates are in [`docs/V2_PLAN.md`](../docs/V2_PLAN.md).
This file covers how to run it.

```
M0 prepare_data -> M1 embed + probe -> (M2 zero-shot) -> M3 fine-tune -> (M4 + vitals) -> evaluate
     laptop           laptop or GPU          optional         GPU (or Apple GPU)    laptop       anywhere
```

## 1. Set up (any machine)

```bash
git clone <repo> && cd MIMIC-CXR-Anomaly-Preprocessing && git checkout v2
bash v2/scripts/setup.sh          # venv, install, unit tests, synthetic wiring check (~5 min)
source .venv-v2/bin/activate
```

You need Python 3.10 or newer. On macOS: `brew install python@3.11`, then
`PYTHON=python3.11 bash v2/scripts/setup.sh`. On a Linux machine with no GPU,
add `--cpu` for a much smaller PyTorch download. On Windows, use WSL2 (the
scripts are bash; the `python -m v2.*` commands also work natively).

Put your paths in `.env` at the repo root. Every v2 command and script reads it.

```bash
MIMIC_CXR_JPG_PATH=/Volumes/<drive>/mimic-cxr-jpg/2.1.0   # the folder with files/ and the CSVs
OUTPUT_PATH=./output
COHORT_DIR=./output/cohorts       # v1's anomalous_{train,val}.parquet
```

Then check the machine and the data:

```bash
python -m v2.doctor               # add --load-models to download and test each encoder once
```

If you don't have v1's cohort files, leave `COHORT_DIR` unset. The population
is then every abnormal study, and `test_v1` is empty. Alternatively, rebuild
them with v1's `python build_cohort.py --anomalous-only`.

## 2. Run the milestones

| Step | Command | Where | Rough time |
|---|---|---|---|
| M0: data + audit | `bash v2/scripts/run.sh m0` | laptop with the drive | 10-30 min: ~30k JPGs read once (~45 GB; the drive's speed decides), ~70 ms of CPU each |
| Check the audit | open `output/v2/audit.md`; **G0** needs ≥6 measurable classes in `test_clean` | | |
| M1 pilot | `MAX_TRAIN=10000 bash v2/scripts/run.sh m1 eval` | laptop, 224 px | embeddings ~6 min per model on an Apple-silicon Mac (full cohort), 1-2 h on an 8-core CPU; probes ~1 min; evaluation ~20 s per run |
| M1 at 518 px | `SIZE=518 bash v2/scripts/run.sh m1 eval` | laptop with an Apple GPU, or the GPU session | ~1 h per model on an Apple-silicon Mac; minutes on CUDA |
| Check **G1** | `output/v2/reports/summary.md` | | |
| M2 (optional) | `bash v2/scripts/run.sh m2 eval` (accept the MedSigLIP terms first, see §6) | laptop or GPU | like M1 |
| M3 | `bash v2/scripts/run.sh smoke m3 eval` (§3); `gpu` also runs M1 and M2 there | rented GPU | 2-3 h, under $10 |
| M4 (optional) | `bash v2/scripts/run.sh m4 eval` | laptop | minutes |

`run.sh` writes a log per step to `output/v2/logs/` and stops at the first
failure. Its settings (`SIZE`, `MAX_TRAIN`, `FT_BATCH`/`FT_ACCUM`,
`GRAD_CKPT`, `FT_HOURS`, `STAGE_HOURS`, `TOTAL_HOURS`, `WORKERS`) are listed
at the top of [`scripts/run.sh`](scripts/run.sh). The default size is 518 px on CUDA and
224 px on Apple MPS or CPU.

**Fine-tuning locally instead of renting a GPU.** On Apple silicon or a local
NVIDIA card, at 224 px, m3 runs three fine-tunes (RAD-DINO seeds 0 and 1,
DenseNet-121), each 10 epochs of ~26k images. Run `bash v2/scripts/run.sh smoke`
first and read the images/s in its log to estimate the time. Then give the
fine-tunes enough budget:
`FT_HOURS=6 TOTAL_HOURS=24 bash v2/scripts/run.sh m3 eval`. A run stopped by
its budget keeps its best epoch, so it does not fail. This is the same recipe
at a lower resolution, so compare it only with runs made at 224 px. CPU-only
fine-tuning is too slow (about 1.5 images/s on 4 cores); use a GPU.

## 3. The GPU session (M3; docs/V2_PLAN.md §8)

1. Rent one A100 or H100 (40-80 GB) with CUDA 12 and at least 50 GB of disk.
   x86-64 is simplest. On an ARM machine (GH200), check that PyTorch still sees
   the GPU after setup (step 2).
2. On the GPU box:
   ```bash
   git clone <repo> && cd MIMIC-CXR-Anomaly-Preprocessing && git checkout v2
   bash v2/scripts/setup.sh && source .venv-v2/bin/activate
   python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'   # must print True
   mkdir -p output/v2
   hf auth login                  # only if M2 (MedSigLIP) runs here
   ```
   Don't copy your `.env` there; it can hold API keys, and the defaults work.
3. From your machine, upload the prepared data (not the raw JPGs). Plain
   `rsync -a` prints nothing per file, so no study IDs reach the console:
   ```bash
   rsync -a output/v2/manifest.parquet output/v2/measurable_classes.json \
       output/v2/audit.json output/v2/images <gpu>:MIMIC-CXR-Anomaly-Preprocessing/output/v2/
   ```
4. On the GPU box, inside `tmux` so a dropped connection doesn't stop it:
   ```bash
   bash v2/scripts/run.sh smoke m3 eval   # if M1 and M2 ran on your machine
   bash v2/scripts/run.sh gpu             # or: smoke m1 m2 m3 eval, all here
   ```
   Each fine-tune trains for at most `FT_HOURS` (1.5 h), then predicts from
   its best epoch. Other steps have a hard limit (`STAGE_HOURS`, 2 h), and the
   run skips the remaining steps after `TOTAL_HOURS` (4 h). Fine-tunes save a
   checkpoint every epoch. To resume after an interruption, run the same
   command again.
   Under 40 GB of GPU memory, use `FT_BATCH=8 FT_ACCUM=4` (same effective batch) or `GRAD_CKPT=1`.
5. Bring the fine-tunes home, then delete the data and the instance. Only the
   fine-tune outputs come back, so nothing at home is overwritten. Then rebuild
   the summary at home, where the probe runs are (G2 compares the two):
   ```bash
   ssh <gpu> 'cd MIMIC-CXR-Anomaly-Preprocessing/output/v2 && tar -czf ../m3.tgz predictions/*_ft_s[0-9].* \
       runs/*_ft_s[0-9]/best.pt runs/*_ft_s[0-9]/*.json embeddings/*_ft_s[0-9].npz logs'
   scp <gpu>:MIMIC-CXR-Anomaly-Preprocessing/output/m3.tgz output/v2/
   tar -xzf output/v2/m3.tgz -C output/v2 && rm output/v2/m3.tgz
   ssh <gpu> 'rm -rf MIMIC-CXR-Anomaly-Preprocessing/output'
   python -m v2.evaluate --all
   ```
   If you ran `gpu`, bring home `predictions`, `embeddings` and `reports` too.

The MIMIC data stays under your PhysioNet credential. Use a provider you
trust, and keep the data on the machine only for the run.

## 4. Outputs (`output/v2/`)

```
manifest.parquet           one row per study: ids, view, split, test-set flags, 12 raw labels
images/{study_id}.png      grayscale, shorter side 518 px (input to every later step)
audit.md / audit.json      split sizes, label counts, drops, overlaps, RAD-DINO check, gate G0
measurable_classes.json    frozen measurable classes per test set (§3.2)
embeddings/{model}_{size}.npz             frozen embeddings (M1, M2) and fine-tuned ones (M3)
predictions/{run}.parquet + {run}.json    one format for every model: val and test scores + how they were made
runs/{run}/                               fine-tunes: last.pt (resume), best.pt, history.json, config.json
reports/{run}.md|json                     per-class AUROC [95% CI], AUPRC, prevalence, per test set
reports/summary.md                        all runs side by side, gates G1 and G2, rows for the results log
reports/compare_A__B__{test set}.md       paired bootstrap comparisons
logs/                                     one log per run.sh step
```

Runs are named `{model}_{size}_probe`, `{model}_{size}_zeroshot`,
`{model}_{size}_ft_s{seed}` and `{embeddings}_fuse_{imageonly|vitals}`. Runs
ending in `_smoke` only check the setup and are left out of the summary.

## 5. Reading the results

- **Headline:** macro AUROC on `test_clean`, over its measurable classes,
  with a 95% patient-bootstrap CI. `test_clean` holds the official MIMIC-CXR
  validate/test patients, whom no model here trained on.
- **v1 comparison:** `test_v1` with the v1 metric, over v1's 8 measurable
  classes. v1 scored 0.792 there. RAD-DINO saw these images (without labels)
  in pretraining, so this is a go/no-go number, not a headline.
- **Model choices use `val` only.** The summary picks the best probe and the
  best fine-tune by val, then runs the gates:
  - **G1:** the best probe on `test_v1` (v1 metric) is ≥ 0.77, so the GPU run is worth it.
  - **G2:** the fine-tune beats the best probe on `test_clean`, with a paired CI above 0.
- To compare any two runs: `python -m v2.evaluate --compare RUN_A RUN_B --test-set test_clean`.
- Paste the summary's results-log rows into `docs/V2_PLAN.md` §9.

## 6. Individual commands

Every step is a module with `--help`. All of them take `--out` (default
`$OUTPUT_PATH/v2`).

```bash
python -m v2.prepare_data --mimic-cxr-jpg PATH [--cohort-dir DIR] [--extend-test-clean] [--limit N] [--v1-hdf5 DIR]
python -m v2.embed --model {rad-dino,dinov2-base,medsiglip,densenet121} --size 224 [--max-train N] [--smoke]
python -m v2.probe --embeddings output/v2/embeddings/rad-dino_224.npz
python -m v2.zero_shot --model medsiglip
python -m v2.finetune --model rad-dino --size 518 --seed 0 [--resume] [--smoke] [--max-hours H] [--save-embeddings]
python -m v2.fuse --embeddings output/v2/embeddings/rad-dino_518.npz --cohort-dir output/cohorts
python -m v2.evaluate --all | --run RUN... | --compare RUN_A RUN_B
python -m v2.doctor [--load-models]
python -m v2.smoke                # the synthetic end-to-end wiring check
```

MedSigLIP (M2) is gated. Accept its terms at
<https://huggingface.co/google/medsiglip-448>, then run `hf auth login`.
`run.sh m2` skips itself if access is missing.

Useful `prepare_data` options:
- `--extend-test-clean`: the §3.1 fallback if G0 fails.
- `--v1-hdf5 <v1 preprocessed dir>`: adds to the audit the share of lateral
  views among v1's training images. v1 picked views by filename, which never
  matched.
- `--limit 200`: a quick rehearsal on about 200 real studies.

## 7. Tests

```bash
pytest tests/v2 -q                                   # ~30 s, synthetic data and tiny random models
python -m v2.smoke                                   # end-to-end wiring check, ~30 s
V2_TEST_NETWORK=1 pytest tests/v2/test_real_models.py  # downloads RAD-DINO, DINOv2, SigLIP (~1.5 GB)
```

The wiring check builds a fake MIMIC-CXR-JPG tree with a planted signal and
runs every stage through the same entry points as the real data. It checks
that the signal is recovered end to end: the probe scores about 1.0 and the
fine-tune is above 0.7. It also checks that misaligned embeddings drop to
chance, which shows the check would catch a broken join.

## 8. What has been verified, and what hasn't

Verified, on a CPU-only Linux container:
- All unit tests and the wiring check pass on two stacks: torch 2.14.1 with
  transformers 5.20.1 (pinned), and torch 2.2.0 with transformers 4.38.0.
- RAD-DINO and DINOv2 load with every weight. v2's preprocessing matches their
  Hugging Face image processors, and the CLS embeddings agree with the
  reference pipeline (cosine > 0.99).
- With the real weights, `run.sh m0 smoke m1 m3 m4 eval` runs end to end
  on a synthetic MIMIC-shaped dataset: RAD-DINO, DINOv2 and DenseNet-121
  embeddings, probes, fine-tunes stopped by their time budget, fusion and the
  summary. `doctor --load-models` passes for RAD-DINO, DINOv2 and
  DenseNet-121.
- The SigLIP code path works with the open SigLIP-base weights and its real
  tokenizer.

Not verified here:
- Your MIMIC data (it is on your drive). The audit and `--smoke` stages exist to catch problems there.
- MedSigLIP's own weights (gated).
- CUDA/bf16 and Apple MPS execution, since the container has no GPU. `run.sh gpu` starts with a smoke step for this reason.

## 9. Working on MIMIC data with an AI assistant

PhysioNet's credentialed data use agreement forbids sharing the data with third
parties. Its [LLM guidance](https://physionet.org/news/post/llm-responsible-use/)
(September 2025) says this includes sending data through APIs or online
platforms. Third-party services must not retain the data, use it for training
or allow human review of it. A coding assistant such as Claude Code sends what
it reads (opened files, command output) to its servers. Unless you have
verified that your account's data handling meets those requirements, keep MIMIC
content out of the conversation:

- Let the assistant run the v2 commands, but not open, print or list the
  dataset, the cohort files, `manifest.parquet`, embeddings or predictions.
- v2's console output is aggregate only: counts, metrics and paths.
  Record-level diagnostics, such as which images were unreadable, go to local
  files like `output/v2/unreadable_images.csv`, and those stay local.
- Share `audit.md` and `reports/*.md`. They hold aggregate counts and metrics,
  the kind of tables a paper publishes.
- Never commit or upload `output/`; it is gitignored.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor` reports slow JPG reads | Expected on USB drives; M0 reads each JPG once, later steps use the PNGs. On a spinning disk, `WORKERS=2` can be faster than many parallel readers |
| `doctor` says sampled cohort JPGs are missing | The download is incomplete for the cohort's studies; M0 drops and counts them, but more than a few percent changes the population |
| CUDA out of memory in M3 | `FT_BATCH=8 FT_ACCUM=4` (same effective batch), or `GRAD_CKPT=1` |
| An op is not implemented on MPS | v2 sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so such ops run on the CPU; if you import v2 after torch in your own code, export it yourself |
| DataLoader worker errors on macOS | `WORKERS=0` |
| "bad steps ... aborting" in M3 | Non-finite loss or gradients in over 1% of steps; check the data and learning rate before spending more GPU time |
| `run.sh` stops with "transformers X, but v2 pins Y" | It found a Python other than the project venv. It uses `.venv-v2/bin/python` when that exists; run `bash v2/scripts/setup.sh`, or set `PYTHON=` to the venv's python. `ALLOW_UNPINNED=1` overrides the check, but embeddings at 224 px then differ from pinned runs |
| `measurable classes changed` warning | `prepare_data` was re-run with different data; the lists should stay frozen after M0 |
| M2 skipped | No MedSigLIP access, see §6 |
