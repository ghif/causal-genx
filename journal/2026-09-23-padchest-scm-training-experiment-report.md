# PadChest SCM training experiment report

Date: 2026-09-23  
Run: `scm_padchest_tpu_v6e1`  
Branch/report scope: local report only; no source-code or artifact changes.

## Executive outcome

The PadChest SCM optimization completed all configured training work: **100/100 epochs**, **global step 200500**, final **train loss 7.0141**, final **validation loss 6.7938**, and final epoch throughput **11,704.452 samples/sec**. The run then exited nonzero while the background metric-artifact writer tried to flush `trainlog.txt` to GCS and received `storage.objects.create` denial. Therefore, the optimization and epoch-100 validation are complete, and the local checkpoint/PDF/TensorBoard artifacts exist, but the process did **not** complete the final artifact contract cleanly.

## Research objective and SCM role

This stage trains the upstream structural causal model (SCM) for PadChest metadata parent variables. The SCM is intended to be frozen and reused by later image-parent predictor, image-model, counterfactual, and inference stages as the tabular parent-variable generator/intervention engine. It is not a clinical validator and does not train on image pixels for likelihood; pixels enter later stages.

The prior PadChest journal runbook (`journal/2026-09-23-padchest-scm-training-plan.md`) framed this artifact as acceptable for downstream use only after split/schema/artifact checks and intervention-support checks. This report finds the training itself completed, but final artifact publication/validation remains incomplete because the process exited during background flushing.

## Configuration and data source

Authoritative config: `configs/padchest_scm_tpu_v6e1.yaml`.

Key settings used by the run and confirmed by `checkpoints/.../hparams.json`:

- Dataset: `padchest`
- Metadata: `gs://external-cxr-dataset/padchest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`
- Image root/prefix retained for row alignment: `gs://external-cxr-dataset/padchest/unpacked/images-224/images-224`
- Seed: `7`
- Optimizer: AdamW, `lr=0.0001`, `weight_decay=0.1`, batch size `64`
- SCM workflow: `epochs=100`, `checkpoint_freq=1`, `speed_log_freq=10`, `plot_samples=1000`, widths `[64, 64]`
- Runtime: TPU, FP32, one local/global device; runtime preflight reported `device_kind=TPU v6 lite`, `jax=0.11.1`
- Run root in artifacts: `checkpoints/padchest/scm_padchest_tpu_v6e1`

PadChest variables are metadata-derived in `src/data/padchest.py`:

| Variable | Encoding/derivation |
|---|---|
| `age_at_study` | `StudyDate_DICOM` year minus `PatientBirth`, clipped 0-110, normalized to `[-1,1]` |
| `sex` | 2-way one-hot; `PatientSex_DICOM == M` mapped to male, otherwise index 0 |
| `pediatric` | binary from `Pediatric == yes` |
| `tb_status` | binary; config uses `tb_or_sequelae`, so labels containing `tuberculosis` or `tuberculosis sequelae` are positive |
| `projection` | 5-way one-hot over `PA`, `AP`, `AP_horizontal`, `L`, `COSTAL`, fallback `PA` |
| `view_position` | 6-way one-hot over `POSTEROANTERIOR`, `ANTEROPOSTERIOR`, `LATERAL`, `AP`, `PA`, `OTHER`, fallback `OTHER` |
| `study_year` | DICOM year normalized from 2007-2017 to `[-1,1]` |

The schema encoded parent dimension recorded by the run is **17** (`1+2+1+1+5+6+1`).

## Causal graph and modeling assumptions

Config and `PAD_CHEST_SCHEMA` declare variables:

```text
age_at_study, sex, pediatric, tb_status, projection, view_position, study_year
```

Declared causal edges:

```text
age_at_study -> tb_status
sex          -> tb_status
pediatric    -> tb_status
```

The implemented `PadChestPGM` uses spline-flow marginals for continuous `age_at_study` and `study_year`; categorical/Bernoulli marginals for sex, pediatric, projection, and view position; and a Bernoulli TB-status head conditioned on age, sex, and pediatric. The plot pair is `age_at_study` versus `tb_status`. Image pixels are deliberately not parents in this SCM stage.

## Split and preprocessing

`PadChestDataset` groups rows by `PatientID` before splitting, shuffles patients deterministically with seed 7, and then assigns 80% train, 10% validation, 10% test. Rows require an `ImageID`. This is patient-safe with respect to the implemented provider, but it is not yet a checked-in split manifest; before downstream training, freeze or export the exact split/fingerprint so predictor and image stages prove row alignment.

The SCM training loop drops the final partial train batch (`drop_last=True`). The completed run logged 2,005 full train batches per epoch, so each epoch optimized on 128,320 train rows.

## Runtime fixes and performance findings

### Generic SCM execution fix

An earlier PadChest launch failed before optimization because the SCM loss path still expected MorphoMNIST keys:

```text
KeyError: 'thickness'
```

The repository now contains the generic dispatch fix (`c9ec773 Fix PadChest SCM training variables`), where the loss asks the model for its declared variable names and passes those named tensors. Post-fix logs show PadChest variables in the loss (`logp(age_at_study)`, `logp(pediatric)`, etc.).

### GCS per-image input bottleneck and metadata-only optimization

After the generic fix, a run still used the image-backed PadChest batch path even though SCM likelihood needs only metadata. The log shows the bottleneck clearly: epoch 1 step 10 took `314.54s`, with only `0.203 samples/s`; steps 20-50 were similarly about 325-330 seconds per 10-batch window. This was diagnosed as per-batch remote PNG loading from the GCS image prefix.

The current repository includes `9a7fc2a Skip image loading for SCM metadata batches`, adding a metadata-only `dataset.samples` path to `training.scm.epoch_batches` and a unit test that fails if PadChest SCM batching attempts PNG download. The completed run after this optimization reached normal TPU/host throughput: epoch 1 summary `10,448.642 samples/s`, later steady state typically around 11k-14k samples/sec, and final epoch **11,704.452 samples/s**.

## Training procedure and monitoring

Command represented by the run log:

```bash
python scripts/run.py train-scm --config configs/padchest_scm_tpu_v6e1.yaml
```

Training monitored every 10 batches with per-variable log-probabilities, joint loss, gradient norm, step time, iteration/sec, sample/sec, epoch sample/sec, ETA, and TensorBoard scalars. Validation ran every epoch because `checkpoint_freq=1`; plots and checkpoint submissions were tied to those validation epochs.

Loss trajectory from `trainlog.txt`:

| Epoch | Step | Train loss | Validation loss | Epoch samples/sec |
|---:|---:|---:|---:|---:|
| 1 | 2,005 | 11.7846 | 11.4961 | 10,448.642 |
| 2 | 4,010 | 11.0255 | 10.8989 | 11,588.423 |
| 5 | 10,025 | 9.9864 | 9.8654 | 14,073.207 |
| 10 | 20,050 | 9.1140 | 8.9939 | 14,000.927 |
| 20 | 40,100 | 8.2923 | 8.1652 | 12,133.943 |
| 40 | 80,200 | 7.5731 | 7.4173 | 11,649.241 |
| 60 | 120,300 | 7.2329 | 7.0498 | 11,447.658 |
| 80 | 160,400 | 7.0797 | 6.8785 | 10,382.703 |
| 95 | 190,475 | 7.0253 | 6.8083 | 11,630.697 |
| 98 | 196,490 | 7.0187 | 6.7974 | 11,636.108 |
| 99 | 198,495 | 7.0103 | 6.7960 | 11,497.019 |
| 100 | 200,500 | **7.0141** | **6.7938** | **11,704.452** |

Validation loss decreased through the last epoch; epoch 100 is the best logged validation loss in this run. Late training has very large logged gradient norms in some windows and final summary (`grad_norm: 5466.641`), so downstream use should include numerical sanity checks even though losses remained finite.

## Artifact inventory

Representative local artifact mirror inspected at:

```text
/mnt/data/firstmate/projects/causal-genx/checkpoints/padchest/scm_padchest_tpu_v6e1
```

Run files observed:

- `trainlog.txt` and `firstmate-run.log`
- `checkpoints/hparams.json`
- TensorBoard event files, including final substantive file `events.out.tfevents.1790173966.tpu-v6e-1-medtpu.413548.0`
- `joint_data.pdf`
- 100 generated model joint PDFs, e.g. `joint_model_2005.pdf`, `joint_model_100250.pdf`, `joint_model_200500.pdf`
- Retained Orbax checkpoint steps due max-to-keep behavior:
  - `checkpoints/196490/_CHECKPOINT_METADATA`
  - `checkpoints/198495/_CHECKPOINT_METADATA`
  - `checkpoints/200500/_CHECKPOINT_METADATA`
  - each retained step also has `commit_success.txt`, `default/_METADATA`, `default/_sharding`, and OCDBT payload files

Final checkpoint representative path:

```text
/mnt/data/firstmate/projects/causal-genx/checkpoints/padchest/scm_padchest_tpu_v6e1/checkpoints/200500
```

## Visual diagnostics from joint-model PDFs

The generated PDFs are diagnostic/model-inspection evidence, not clinical validation.

The plot implementation renders `age_at_study` versus `tb_status`, with TB jittered into the two binary bands (`0 (Neg)` and `1 (Pos)`), marginal histograms, and a red binned `P(TB=1|age)` trend line. `joint_data.pdf` is the empirical train split view; `joint_model_<step>.pdf` is generated from the EMA SCM at that step with 1,000 samples.

Visual findings:

- `joint_data.pdf` shows the modeled positive TB band is present but sparse compared with the negative band.
- Early model output (`joint_model_2005.pdf`) visibly over-allocates samples to the positive TB band relative to the final plot.
- The final plot (`joint_model_200500.pdf`) places almost all generated samples in the TB-negative band with only a small positive-band presence, visually matching the rarity pattern in the data plot at a coarse level.
- Both data and final model plots keep generated ages within the plotted normalized age support; no plot evidence was used to claim calibration, clinical correctness, or subgroup validity.

Representative plot paths:

```text
.../joint_data.pdf
.../joint_model_2005.pdf
.../joint_model_200500.pdf
```

## Shutdown failure and artifact-contract status

After epoch 100 validation and after enqueuing the final checkpoint/metric artifacts, shutdown failed in the metric background artifact writer:

```text
OSError: Forbidden: ... storage.objects.create ...
RuntimeError: Background artifact writer failed.
EXIT_CODE 1
```

The failure occurred while syncing `trainlog.txt` to:

```text
gs://external-cxr-dataset/padchest/checkpoints/padchest/scm_padchest_tpu_v6e1/trainlog.txt
```

This distinction matters:

- Completed: optimization loop, epoch-100 validation, local final checkpoint enqueue/flush, local TensorBoard/log/PDF artifacts observed.
- Incomplete: process-level success, final `run()` artifact validation path, and GCS final artifact publication contract. Treat remote artifacts as not authoritative until GCS write permissions and sync status are verified.

## Limitations and caveats

- Single seed only; no stability check across seeds.
- No test-set SCM metrics reported in the completed log.
- Patient-safe split is implemented in code, but no durable split manifest/fingerprint is committed yet.
- Metadata mappings are pragmatic: unknown/non-`M` sex maps to index 0, unknown projection/view fall back to defaults, and TB label is a coarse binary derived from label strings.
- Joint PDFs inspect only `age_at_study` and `tb_status`; they do not validate projection/view distributions, sex/pediatric calibration, conditionals, or intervention support.
- Final process exit was nonzero due background GCS artifact flushing, so do not treat the run as a cleanly published artifact despite local checkpoint completeness.

## Recommended next steps before predictor/image-model training

1. Fix GCS write permissions or run with remote sync disabled, then rerun only the final artifact validation/sync path if possible; do not overwrite useful local artifacts without a backup.
2. Freeze a split manifest or manifest fingerprint covering metadata URI, schema version, selected variables, normalization constants, and seed.
3. Run local artifact validation on `checkpoints/padchest/scm_padchest_tpu_v6e1` and explicitly restore `checkpoints/200500` to confirm EMA parameters load.
4. Produce additional diagnostics: per-variable calibration/marginals, TB support by age/sex/pediatric bins, projection/view support, and test split likelihood.
5. Confirm downstream configs use the schema-derived 17-dimensional parent vector and the same variable order/normalization.
6. For later counterfactual configs, point to a concrete restored SCM checkpoint step (or create a verified `best` alias) rather than assuming a `checkpoints/best` path exists.
7. For predictor/image training, expect real image GCS reads; preflight throughput and permissions separately so the metadata-only SCM optimization is not mistaken for image-stage readiness.
