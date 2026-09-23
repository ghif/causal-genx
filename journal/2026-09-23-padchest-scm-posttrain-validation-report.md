# PadChest SCM post-training validation report

Date: 2026-09-23  
Branch: `fm/padchest-scm-posttrain-validation`  
Run validated: `checkpoints/padchest/scm_padchest_tpu_v6e1`, final step `200500`  
Mode: local-only; no remote data was overwritten and no clinical validity is claimed.

## Artifacts added

Validation artifacts are under:

```text
journal/artifacts/padchest_scm_posttrain_validation/
```

Files:

- `padchest_scm_split_schema_manifest.json`
- `padchest_scm_posttrain_diagnostics.json`
- `gcs_publication_probe.json`

Generation command used the `med-jax` environment:

```bash
PYTHONPATH=src /mnt/data/miniconda/envs/med-jax/bin/python scripts/padchest_scm_posttrain.py \
  --checkpoint /mnt/data/firstmate/projects/causal-genx/checkpoints/padchest/scm_padchest_tpu_v6e1/checkpoints/200500 \
  --metadata /mnt/data/dataset/padchest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv \
  --output-dir journal/artifacts/padchest_scm_posttrain_validation \
  --model-samples 0
```

The local metadata cache SHA-256 is `b02e42e5f1c53609298b47b1246a5dbed725fcae88e2cff407b772dfe10d2ddc`; the configured metadata URI remains `gs://external-cxr-dataset/padchest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`.

## Checkpoint restore and compatibility

Final step path inspected:

```text
/mnt/data/firstmate/projects/causal-genx/checkpoints/padchest/scm_padchest_tpu_v6e1/checkpoints/200500
```

The step directory contains Orbax `_CHECKPOINT_METADATA`, `commit_success.txt`, `default/_METADATA`, `default/_sharding`, `default/commit_success.txt`, and OCDBT payload files.

Restore used the project-supported `utils.load_checkpoint_with_path` path and a `PadChestPGM(widths=(64,64), seed=7)` template. Result:

- resolved path: final `200500` step
- checkpoint `format_version`: `2`
- checkpoint keys include `params`, `ema_params`, `model_params`, `opt_state`, `epoch`, `step`, `best_loss`, `ema_step`, `ema_initted`, and `hparams`
- `epoch=100`, `step=200500`, `best_loss=6.793773543390678`
- `model_params`, `ema_params`, and `params` pytrees are compatible with the PadChest SCM template

## Frozen split/schema fingerprint

Manifest fingerprint: `01df017f06fec99151eeed382a22b57c05e50c3ba31913635e12b907390522f5`  
Split fingerprint: `b21463f06015f8d2d424fe836da750511d68221d27328be83851a266ef44ddcf`

Split counts from the exact patient-level algorithm, seed `7`:

| split | patients | rows with ImageID |
|---|---:|---:|
| train | 54,100 | 128,343 |
| valid | 6,762 | 16,297 |
| test | 6,763 | 16,221 |

Variable order is fixed as:

```text
age_at_study, sex, pediatric, tb_status, projection, view_position, study_year
```

Encoded parent dimension is `17`. Normalization constants and categorical levels are recorded in the manifest.

## Test-set likelihood and diagnostics

Diagnostics ran in `med-jax` with JAX `0.11.1`; detected backend was TPU. Test-set SCM likelihood from final EMA parameters:

| metric | value |
|---|---:|
| loss | 6.5571396572 |
| logp(age_at_study) | -0.6749261902 |
| logp(sex) | -0.6933306715 |
| logp(pediatric) | -0.0004802980 |
| logp(tb_status) | -0.0522295182 |
| logp(projection) | -0.7653243302 |
| logp(view_position) | -1.1898132131 |
| logp(study_year) | -3.1810354412 |

Marginal/support checks against an equal-size final-model sample (`n=16,221`):

- age: data mean 58.66 years; model mean 59.77 years; no normalized support violations.
- sex: data male rate 48.94%; model male rate 50.34%; TVD 0.0141.
- pediatric: data has no positive pediatric rows in the test split; model sampled 0.080% positives; TVD 0.0008. This is a support limitation.
- TB status: data positive rate 0.9186%; model positive rate 0.7953%; TVD 0.00123.
- projection: TVD 0.00826; `AP_horizontal` has zero support in both data and model test/sample diagnostics.
- view position: TVD 0.01258; all configured categories have test support.
- study year: data mean 2012.21; model mean 2012.01; no normalized support violations.

TB calibration on test metadata parents:

- observed positive rate: 0.0091856
- mean predicted probability: 0.0083006
- Brier score: 0.0090993
- 10-bin ECE: 0.000885; all predictions fell in the `[0.0, 0.1]` bin.

TB support by subgroup is recorded in JSON. Notable limitation: the test split has `pediatric_1 count=0`, so pediatric-positive TB calibration/support cannot be evaluated locally for this split.

## GCS publication investigation

Current active account: `mghifary@gmail.com`.

A non-mutating IAM permission probe against `external-cxr-dataset` failed with HTTP 403 because the account lacks `serviceusage.services.use`; the completed run also logged `storage.objects.create` denial while syncing `trainlog.txt`. I did not create, delete, or overwrite any GCS object.

Safe remote-sync prerequisite: verify the publishing principal has `serviceusage.services.use` plus `storage.objects.create` on the exact `gs://external-cxr-dataset/padchest/checkpoints/padchest/scm_padchest_tpu_v6e1/` prefix, then sync only missing immutable artifacts. For local-only reruns, leave `artifacts.remote_root` empty to use the already-supported no-remote path.

## Remaining blockers/limitations

- Remote artifact publication is not verified with current credentials.
- GCS metadata URI equality was not freshly read during validation; diagnostics used the local cached CSV and record its SHA-256.
- Single final checkpoint and single split seed only; no multi-seed stability check.
- Diagnostics are metadata-level SCM checks only and do not claim clinical validity or image-model validity.
