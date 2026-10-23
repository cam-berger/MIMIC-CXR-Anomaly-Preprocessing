# v2: pretrained chest X-ray encoders on the 12-class task

v2 measures what modern pretrained models (RAD-DINO, MedSigLIP, DINOv2) do on
v1's 12 CheXpert classes. It uses images only, under a fixed evaluation
protocol, and runs from a laptop with the data drive or a rented GPU. The plan,
task definition and decision gates are in [`docs/V2_PLAN.md`](../docs/V2_PLAN.md).
This file covers how to run it.

```
M0 prepare_data -> M1 embed + probe -> (M2 zero-shot) -> M3 fine-tune -> M4 pre-imaging text -> M4b fusion follow-ups -> evaluate
     laptop           laptop or GPU          optional         GPU (or Apple GPU)    laptop               laptop (Apple GPU)      anywhere
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
# M4 only: text and triage data from before each X-ray
MIMIC_CXR_REPORTS_PATH=/Volumes/<drive>/mimic-cxr/2.1.0/mimic-cxr-reports.zip   # from the MIMIC-CXR project
MIMIC_IV_ED_PATH=/Volumes/<drive>/mimic-iv-ed/2.2   # the folder with ed/edstays and ed/triage
MIMIC_IV_PATH=/Volumes/<drive>/mimiciv/3.1          # optional: hosp/patients, for age
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
| M3 | `bash v2/scripts/run.sh smoke m3 eval` (§3); `gpu` also runs M1 and M2 there | rented GPU | about 2 h on an A100-40GB (measured: 2.15 instance hours, about $3-4) |
| M4: text written before the X-ray | `SIZE=518 bash v2/scripts/run.sh m4 eval`; read `reports/m4_rad-dino_518.md` (red flags, then **G4**) | laptop | about 15 min |
| M4b: text-steered attention, prior-study labels, val combiner | `SIZE=518 TOTAL_HOURS=8 bash v2/scripts/run.sh m4b eval`; read `reports/m4b_attention_rad-dino_518.md` (the head check, then **G4b**), `reports/m4b_rad-dino_518.md` and `reports/m4b_stack.md` | laptop with an Apple GPU, ~7 GB free | measured: 71 min (patch tokens 53 min, 15 attention heads 8.5 min) |
| M5a: line up with published results | `TOTAL_HOURS=20 bash v2/scripts/run.sh bench`; read `$OUTPUT_PATH/v2_bench/reports/lineup.md` (red flags, then the like-for-like table) | laptop with an Apple GPU and the data drive, ~65 GB free | measured: prepare 9 h from a USB hard drive (1-3 h from a fast disk), embeddings and tokens 6.1 h, probe and fusion 27 min, attention heads 2.3 h |
| M5b, on the Mac: text through the val combiner | `bash v2/scripts/run.sh bench_stack`; read `reports/m5_stack.md` (the check first), then `lineup.md` | laptop | about 30 min; run it again after the GPU part |
| M5b, on a GPU: two fine-tunes on the benchmark | `bash v2/scripts/run.sh bench_ft` (§3, M5b) | rented GPU | about 3.5 h of an A100 for seed 0 of both heads, plus uploading ~16 GB |
| M7a: a radiologist's labels on the official test split | `bash v2/scripts/run.sh m7a`; read `reports/m7a_radiologist.md` in `BENCH_OUT` | laptop (needs `mimic-cxr-2.1.0-test-set-labeled.csv` in `MIMIC_CXR_JPG_PATH`) | minutes |
| M7b: the CheXpert test set | `bash v2/scripts/run.sh m7b`; read `reports/m7b_chexpert.md` in `EXT_OUT` (red flags, then **G7**) | laptop with an Apple GPU; `CHEXPERT_TEST_PATH` holds the test images and `groundtruth.csv` (CheXlocalize's `test_labels.csv`, if there, is checked against it) | about 6 min |
| M6: BERT-family encoders for the pre-imaging text | `TOTAL_HOURS=16 bash v2/scripts/run.sh bench_text`; read `reports/m6_text.md` (red flags, then **G6**), then `lineup.md` | laptop with an Apple GPU, ~8 GB free (text embeddings ~6 GB); downloads three public models (~2 GB) | estimated 8-12 h (not yet measured): three fine-tunes first, then frozen embeddings and probes per encoder |

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

**M5b (§6 M5b amendment)** follows the same steps with the benchmark directory.
Upload about 16 GB while `setup.sh` runs; if the connection drops, the same
`rsync` picks up where it stopped:
```bash
ssh <gpu> 'mkdir -p MIMIC-CXR-Anomaly-Preprocessing/output/v2_bench'
rsync -a output/v2_bench/manifest.parquet output/v2_bench/measurable_classes.json output/v2_bench/audit.json \
    output/v2_bench/images <gpu>:MIMIC-CXR-Anomaly-Preprocessing/output/v2_bench/
```
Then run `TOTAL_HOURS=8 bash v2/scripts/run.sh bench_ft` in `tmux`. It runs a smoke
fine-tune of each head on a few real studies, then the mean-pool and the
attention-head fine-tunes (3 epochs each, at most `BENCH_FT_HOURS`, default 3).
Bring the results home and run the Mac part:
```bash
ssh <gpu> 'cd MIMIC-CXR-Anomaly-Preprocessing/output && tar -czf m5b.tgz v2_bench/predictions/*_ft_*s[0-9].* \
    v2_bench/runs/*_ft_*s[0-9]/best.pt v2_bench/runs/*_ft_*s[0-9]/*.json v2_bench/embeddings/*_ft_s[0-9].npz v2/logs'
scp <gpu>:MIMIC-CXR-Anomaly-Preprocessing/output/m5b.tgz output/
tar -xzf output/m5b.tgz -C output && rm output/m5b.tgz
ssh <gpu> 'rm -rf MIMIC-CXR-Anomaly-Preprocessing/output'
bash v2/scripts/run.sh bench_stack
```

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
context.parquet                           M4 inputs per study: indication, chief complaint, prior report, vitals (record-level: keep local)
context_audit.md / .json                  M4 coverage and leak-guard counts (aggregate)
reports/{run}.md|json                     per-class AUROC [95% CI], AUPRC, prevalence, per test set
reports/summary.md                        all runs side by side, gates G1 and G2, rows for the results log
reports/compare_A__B__{test set}.md       paired bootstrap comparisons
reports/m4_{embeddings}.md                M4: leakage red flags, every arm against `image`, gate G4
embeddings/{model}_{size}_tokens{G}.npy   M4b: CLS + patch tokens pooled to G x G, float16 (+ .json with ids; record-level)
reports/m4b_attention_{model}_{size}.md   M4b: head validity check, gate G4b, contrasts, how far text moves attention
reports/m4b_{embeddings}.md               M4b: prior-study label arms against M4's arms
reports/m4b_stack.md                      M4b: the combiner fit on val, and its check against concatenation
reports/m5_stack.md                       M5b (BENCH_OUT): the combiners on the benchmark, and their checks
embeddings/text_{encoder}_{source}.npz    M6 (BENCH_OUT): a language model's mean-pooled text embedding per study and source (record-level)
reports/m6_text.md                        M6 (BENCH_OUT): red flags, gate G6, the encoders compared, image + text through the combiner
reports/m7a_radiologist.md                M7a (BENCH_OUT): every run against a radiologist's labels, paired with the labeler's
```

M7b's CheXpert test set lives in its own directory, `EXT_OUT` (default
`$OUTPUT_PATH/v2_chexpert`), with the usual layout: `manifest.parquet` (500
studies, all `test`), `images/`, `audit.md` (frontal picks, label counts,
RAD-DINO's training list checked for test patients), the benchmark's models'
`predictions/` and `reports/m7b_chexpert.md` (gate G7). Nothing is trained
there.

M5's benchmark (§6 M5) lives in its own directory, `BENCH_OUT` (default
`$OUTPUT_PATH/v2_bench`), with the same layout: every frontal study, the
official split (val is official validate; `test_clean` is official test),
images as JPEG, and `reports/lineup.md`, our runs next to the published ones.
`prepare_data` refuses to replace a manifest built for another population or
split scheme, so v2.0's directory can't be overwritten by mistake.

```logs/                                     one log per run.sh step
```

Runs are named `{model}_{size}_probe`, `{model}_{size}_zeroshot`,
`{model}_{size}_ft_s{seed}` (`_ft_attn_s{seed}` with the attention head, M5b), `{embeddings}_ctx_{arm}`
(M4 and M4b's prior-label arms; arms in `v2/fuse.py`), `{model}_{size}_attn_{h0,h1,h2}` (M4b) and
`{image run}_stack_{text arm}` (M4b, M5b). M6 adds `{embeddings}_ctx_{arm}_{encoder}` (frozen language-model
features in place of TF-IDF; `_tfidf-{encoder}` with both), `{encoder}_ft_{source}_s{seed}` (a fine-tuned
language model) and `{image run}_stack_{arm}_{encoder}` or `_stack_{encoder}_ft_{source}`. Runs ending in
`_smoke` only check the setup and are left out of the summary.

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
  - **G4** (M4): image + pre-imaging text beats the image alone on `test_clean` (frozen features).
  - **G4b** (M4b): `h2` minus `h1` on `test_clean`, read only after the head validity check passes.
  - **G6** (M6, benchmark): on the indication alone, the fine-tuned medical encoder with the better val
    score (`biomedbert` or `clinicalbert`) minus the fine-tuned `bert`, on the official test split.
  - **G7** (M7b): on the CheXpert test set, the benchmark image model with the best val score, mean AUROC
    over the five usual findings, against CheXzero's 0.889 (also trained on MIMIC-CXR): PASS unless 0.889 is
    above our 95% interval.
- **M5 lineup:** published results use "positive vs rest" (uncertain and
  unmentioned count as negative), so `lineup.md` scores the official test
  split that way (`--metric u0` in `evaluate`). A published number reads as
  below, level with or above ours by where it falls against our 95% CI.
  From M5b on, text on the stronger image models (the attention head, the
  fine-tunes) is added by the val combiner: `*_stack_indication` runs are the
  image + indication rows, like for like with MMBT (§6 M5b amendment).
- **Text gains: read the v2 policy first.** The v1 metric counts only findings
  the report mentions. Text the radiologist reads makes a mention more likely,
  so under v1 text gains look smaller than they are (§6 M4b amendment).
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
python -m v2.finetune --model rad-dino --size 518 --seed 0 [--head attention] [--resume] [--smoke] [--max-hours H] [--save-embeddings]
python -m v2.context [--reports ZIP_OR_DIR] [--mimic-iv-ed DIR] [--mimic-iv DIR]   # M4 inputs -> context.parquet
python -m v2.fuse --embeddings output/v2/embeddings/rad-dino_518.npz [--arms image,image_text]
python -m v2.embed --model rad-dino --size 518 --tokens 12      # M4b: patch tokens (resumable)
python -m v2.attend --tokens output/v2/embeddings/rad-dino_518_tokens12.npy [--arms h0,h1,h2] [--seeds 0,1,2,3,4]
python -m v2.stack --pair rad-dino_518_ft_s1 rad-dino_518_ctx_text [--pair IMAGE_RUN TEXT_RUN ...]
python -m v2.prepare_data --population all --split-scheme official --image-format jpg --out output/v2_bench   # M5
python -m v2.stack --pair rad-dino_518_attn_h0 rad-dino_518_ctx_indication --out output/v2_bench   # M5b
python -m v2.lineup --out output/v2_bench                       # M5: reports/lineup.md
python -m v2.textft --encoder biomedbert --out output/v2_bench  # M6: fine-tune on the indication
python -m v2.textenc --encoder biomedbert --out output/v2_bench # M6: frozen text embeddings per source
python -m v2.fuse --embeddings output/v2_bench/embeddings/rad-dino_518.npz --text-encoder biomedbert --out output/v2_bench
python -m v2.textft --report --out output/v2_bench              # M6: reports/m6_text.md, gate G6
python -m v2.external radiologist --bench output/v2_bench       # M7a: reports/m7a_radiologist.md
python -m v2.external chexpert --chexpert DIR --bench output/v2_bench --out output/v2_chexpert [--steps prepare,apply,report]
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
pytest tests/v2 -q                                   # a few minutes, synthetic data and tiny random models
python -m v2.smoke                                   # end-to-end wiring check, ~2 min
V2_TEST_NETWORK=1 pytest tests/v2/test_real_models.py  # downloads RAD-DINO, DINOv2, SigLIP and the M6 text encoders (~3 GB)
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
- M4b on synthetic data only: patch tokens (resume included), the attention
  heads (with a planted signal that only text-steered attention can solve: `h2`
  scores > 0.95, `h1` < 0.85), prior-study labels (a leak trap: the arm fed each
  study's own labels scores > 0.95) and the val combiner (cross-fitted val
  scores). Not yet run on real data or on MPS.
- M5b on synthetic data only: the attention-head fine-tune (learns the planted
  signal in the wiring check) and the val combiner on the benchmark. The
  published per-class tables are checked against the papers' own macros.
- M7 on synthetic data only: the radiologist-label comparison, and a
  CheXpert-style test set (frontal picks, the label join, the RAD-DINO
  exposure trap, the probe refit reproducing exactly, fine-tunes reloaded from
  their checkpoints). The real CheXpert layout is checked only from
  `groundtruth.csv` (`CheXpert-v1.0/test/patient#/study#`).
- M6 on synthetic data, with a random two-layer BERT: frozen embeddings, the
  probes, the fine-tune (it learns a planted word, and in the wiring check,
  fed whole reports, it scores > 0.9: proof that it can learn from text) and
  the report. The three real encoders load at their pinned revisions with
  transformers 5.20.1 and give 768-wide embeddings (checked on a made-up
  sentence); not yet run on real data or on MPS.

Verified on the real data (October 2026; results in `docs/V2_PLAN.md` §9):
- M0-M2 on an Apple-silicon Mac (MPS), including MedSigLIP's gated weights.
- M3 on a Lambda A100-SXM4-40GB (CUDA, bf16): three fine-tunes with no
  non-finite steps, in about 2 h of instance time. Lambda's preinstalled torch
  needed `numpy<2`, which `setup.sh` now installs when needed.
- M4b and M5a on an Apple M5 Pro (MPS, 64 GB): patch tokens, attention heads,
  prior labels and the val combiner; the whole benchmark (218k studies) with
  its 49 GB token cache.
- M5b on a Lambda GH200 (aarch64, CUDA 12.8, the provider's torch 2.7.0): both
  benchmark fine-tunes in 102 min at about 220 img/s, with no non-finite steps.
- M6 and M7a on the M5 Pro: the three text encoders' fine-tunes, embeddings and
  probes in 9 h 25 min; the radiologist-label comparison in 88 s.
- M7b on the M5 Pro: the CheXpert test set (668 images from the AIMI Azure
  release, via azcopy) prepared, scored by three benchmark models and reported
  in 375 s.

Not verified: other providers' machine images. Start every new machine with
`bash v2/scripts/run.sh smoke`.

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
  dataset, the cohort files, `manifest.parquet`, `context.parquet`, embeddings
  (including the M4b token caches and their `.json` sidecars) or predictions.
- v2's console output is aggregate only: counts, metrics and paths.
  Record-level diagnostics, such as which images were unreadable, go to local
  files like `output/v2/unreadable_images.csv`, and those stay local.
- Share `audit.md` and `reports/*.md`. They hold aggregate counts and metrics,
  the kind of tables a paper publishes.
- Never commit or upload `output/`; it is gitignored.
- Never commit anything derived from a single study either: X-ray images
  (originals, crops or reconstructions, even as figures in the docs),
  tensors, per-study feature files. A public repository redistributes them,
  and the agreement forbids that.
- File names hold study IDs (`images/`, rsync's temporary files, run
  outputs). Check disk use and counts with commands that print totals only,
  and silence their per-file errors: `du -sh DIR 2>/dev/null`,
  `find DIR -type f | wc -l`.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor` reports slow JPG reads | Expected on USB drives; M0 reads each JPG once, later steps use the PNGs. On a spinning disk, `WORKERS=2` can be faster than many parallel readers |
| `doctor` says sampled cohort JPGs are missing | The download is incomplete for the cohort's studies; M0 drops and counts them, but more than a few percent changes the population |
| CUDA out of memory in M3 | `FT_BATCH=8 FT_ACCUM=4` (same effective batch), or `GRAD_CKPT=1` |
| Tests fail with `RuntimeError: Numpy is not available` | The machine's torch was built for NumPy 1.x and the venv has NumPy 2. `setup.sh` checks for this and installs `numpy<2`; by hand: `.venv-v2/bin/python -m pip install 'numpy<2'` |
| An op is not implemented on MPS | v2 sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so such ops run on the CPU; if you import v2 after torch in your own code, export it yourself |
| DataLoader worker errors on macOS | `WORKERS=0` |
| "bad steps ... aborting" in M3 | Non-finite loss or gradients in over 1% of steps; check the data and learning rate before spending more GPU time |
| `run.sh` stops with "transformers X, but v2 pins Y" | It found a Python other than the project venv. It uses `.venv-v2/bin/python` when that exists; run `bash v2/scripts/setup.sh`, or set `PYTHON=` to the venv's python. `ALLOW_UNPINNED=1` overrides the check, but embeddings at 224 px then differ from pinned runs |
| `measurable classes changed` warning | `prepare_data` was re-run with different data; the lists should stay frozen after M0 |
| M2 skipped | No MedSigLIP access, see §6 |
| `m4 needs MIMIC_CXR_REPORTS_PATH` | Download `mimic-cxr-reports.zip` from the MIMIC-CXR project (not MIMIC-CXR-JPG) and set the path in `.env`; `python -m v2.doctor` checks it |
| `RED FLAG` in the M4 report | A text-only arm is as good as the image, or a class scores above 0.90 on text alone. Don't read G4: inspect the extraction locally, without sharing the text |
| `RED FLAG` for `prior_labels` in the M4b report | The prior study's labels alone reach the image, or a class above 0.90. Check locally that `prior_study_id` is never the study itself and is at least 24 h older; then read the copy-forward caveat (§6 M4b) |
| `predates M4b` from doctor, or fuse says the context has no prior-study labels | `run.sh m4b` rebuilds `context.parquet` (by hand: `python -m v2.context`); its text columns don't change |
| `prepare_data` says a directory "holds a manifest for population ..." | You pointed M5's benchmark (or another population) at an existing output dir. Use its own `--out` (`BENCH_OUT`) |
| `prepare_data` stops with "is the drive still mounted?" | More than 0.5% of JPGs failed to read, which usually means the drive dropped off the bus or the mount went stale. Remount it (read-only is fine), check with `python -m v2.doctor`, keep the Mac awake (`caffeinate -ims`), and rerun: written images are kept |
| `bench_stack` says it needs the bench stage's text arms | Run `bench` with `MIMIC_CXR_REPORTS_PATH` and `MIMIC_IV_ED_PATH` set: the combiner needs `ctx_indication` |
| `bench_text` stopped by `TOTAL_HOURS` | Run it again with the same settings: finished fine-tunes are kept, and `textenc` skips sources already embedded. The fuse arms are refit |
| `m7b` says there are no CheXpert test images | The CheXlocalize copy on Redivis has only annotations. The images are in the original CheXlocalize release on Stanford AIMI's Azure portal (download with azcopy and the link on your dataset page; treat the link as a password and keep it out of files); put them (any layout with `patient#/study#/view#_frontal.jpg`) and `groundtruth.csv` under `CHEXPERT_TEST_PATH` |
| `RED FLAG` in `m7b` about RAD-DINO's training list, the probe refit or a reloaded fine-tune | Stop and report the line: the external test isn't clean, or a model isn't the one the benchmark scored |
| `RED FLAG` in `m6_text.md` | A language model's text-only run reaches the image, or a class scores above 0.90. Don't read G6: inspect locally, without sharing the text |
| A `bench_ft_smoke_*` step fails on the GPU | Don't start the long runs: report the error line. Under 40 GB of GPU memory use `FT_BATCH=8 FT_ACCUM=4` or `GRAD_CKPT=1` |
| The M5 probe runs fewer parallel fits than `--jobs` | Each fit copies the training features (2.6 GB at M5's size); the probe caps the workers to fit in memory and logs it |
| G4b `WITHHELD` | `h0` scored more than 0.02 below the M1 probe on val, or the check couldn't run. Don't read G4b: look at the per-seed histories in `predictions/*_attn_h0.json` (aggregate) first |
| `token 0 differs from ... npz` in attend | The tokens and the M1 embeddings came from different images, preprocessing or transformers versions. Re-run `v2.embed --tokens` in the pinned venv |
