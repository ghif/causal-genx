# PadChest SCM training plan

Date: 2026-09-23

## Scope and current-code context

This plan covers the PadChest structural causal model (SCM) stage only. It is the upstream tabular causal model that later predictor, image-model, and counterfactual stages treat as a frozen parent-variable generator/intervention engine.

Relevant project facts from the current checkout:

- Research runs use the single CLI entrypoint `scripts/run.py`; SCM runs are invoked with `python scripts/run.py train-scm --config <yaml>`.
- The typed SCM config contract in `src/config.py` expects `workflow.type: train-scm`, `runtime.precision: fp32`, optimizer `lr`, `weight_decay`, `batch_size`, artifact roots, and SCM workflow fields such as `epochs`, `speed_log_freq`, `checkpoint_freq`, `plot_samples`, `widths`, and `benchmark_steps`.
- `src/training/scm.py` currently validates `dataset` against `morphomnist` and `cxr_rait`, trains a flow-style PGM, logs TensorBoard scalars plus `trainlog.txt`, plots a data/model joint PDF, and writes Orbax checkpoints containing live params, EMA params, optimizer state, epoch, step, best validation loss, and hparams with `format_version: 2`.
- The task references `configs/padchest_scm_tpu_v6e1.yaml` and `src/data/padchest.py`, but those files are not present in this local worktree. Treat this document as the PadChest-specific runbook to apply once those paths are present in the active branch or restored from the PadChest research branch.
- The active PadChest run target is the `med-jax` conda environment on TPU v6e-1. Do not stop, restart, attach to, or otherwise disturb any live training process or tmux session while using this plan.

## Data and metadata inputs

The SCM stage should consume metadata and split manifests, not pixels, except where the data provider needs image paths to align patient/study identifiers with downstream stages. The expected PadChest inputs are:

1. PadChest metadata table with stable image/study/patient identifiers.
2. Image-path column or manifest that downstream predictor/image stages can reuse for exact row alignment.
3. Causal parent columns selected for the PadChest SCM, including demographic variables and label variables used as parents for image generation/counterfactual interventions.
4. Split assignment column or deterministic split manifest.
5. Optional exclusion flags for unusable views, missing labels, duplicates, non-frontal images, or records outside the supported causal schema.

Before training, freeze a manifest fingerprint made from the metadata URI, selected columns, filter policy, split policy, schema version, and normalization constants. Store that fingerprint in hparams or a sidecar note so later stages can prove they used the same population.

## Causal variables and encoding contract

Use the variables declared by the PadChest causal schema in the config/code as the source of truth. The provider should expose each variable separately in batches, matching the pattern `src/training/scm.py` expects for dataset-specific loss dispatch, and should also expose a concatenated `pa` vector for image-stage conditioning.

Recommended variable categories:

- Continuous demographics: e.g. age, normalized to `[-1, 1]` with documented physical min/max and clipping policy.
- Binary/categorical demographics: e.g. sex/gender, one-hot encoded with a documented unknown/missing policy.
- Pathology labels: one-hot or multi-hot depending on the graph assumption. If a pathology is modeled as a categorical node, use explicit class probabilities; if modeled as independent Bernoulli labels, document that the likelihood factorization differs from the CXR-RAIT categorical example.
- Acquisition/view variables only if they are causal parents needed by downstream image models; otherwise use them as filters/strata rather than intervention targets.

Every encoded variable must have an inverse interpretation for diagnostics and counterfactual reporting. Missingness should be handled before training, not silently converted into a valid clinical state unless that state is explicitly modeled.

## Graph assumptions

Document the PadChest DAG next to the config `causal_schema` and keep it minimal enough for identifiable counterfactual interventions. A starting assumption for chest X-ray metadata is:

- Demographics precede pathology labels.
- Pathology labels and selected acquisition/view variables are direct parents of the image mechanism.
- Age may influence some pathology labels.
- Sex/gender may influence some pathology labels if clinically justified by the research question.
- Image pixels are not parents of metadata in the SCM stage; the image-to-parent predictor is trained separately and must not leak into SCM fitting.

For each edge, record whether it is a causal assumption, a pragmatic dependency for density modeling, or an image-conditioning dependency. Avoid adding edges solely to improve likelihood if they create unsupported intervention semantics.

## Split policy

Use one patient-level split shared by SCM, predictor, image, counterfactual, and inference evaluation:

1. Group by patient identifier before splitting to prevent same-patient leakage.
2. Keep train/valid/test proportions deterministic and seed-controlled, or consume a checked-in split manifest.
3. Stratify on high-priority labels and demographics where feasible, especially rare findings used as intervention targets.
4. Validate that every split has nonzero support for each modeled category and enough positive cases for each pathology intervention.
5. Never resplit independently in later stages; later stages should load the same split assignment or an identical deterministic provider.

If the current `src/data/padchest.py` provider uses an internal shuffle/split, confirm it is patient-level and stable before considering the SCM artifact valid for downstream work.

## Preprocessing and validation

Preflight checks before the TPU job:

```bash
conda activate med-jax
python scripts/run.py train-scm \
  --config configs/padchest_scm_tpu_v6e1.yaml --dry-run
```

Then run a small bounded smoke job by overriding benchmark/epoch fields if the config supports it, writing to a separate run name. Validate:

- Config loads with `runtime.accelerator: tpu`, `runtime.precision: fp32`, and TPU v6e-1 expected device counts if specified.
- Dataset provider returns `train`, `valid`, and `test` splits with nonzero lengths.
- Batch keys match the PadChest SCM loss and model signature.
- Continuous variables are finite after normalization; categorical variables are valid one-hot/multi-hot encodings.
- Split-level summary statistics match the frozen manifest.
- GCS reads and writes work from the active shell.

GCS note: if `fsspec/gcsfs` fails because the authenticated principal lacks `serviceusage.services.use` on a quota project, use Application Default Credentials without a quota-project override for this run rather than changing training code mid-run.

## Training command and config

Use the dedicated TPU config referenced by the task:

```bash
conda activate med-jax
python scripts/run.py train-scm \
  --config configs/padchest_scm_tpu_v6e1.yaml
```

Do not reuse the same artifact run name for smoke tests and full training. For temporary validation runs, override `artifacts.run_name` to include `smoke` and set a very small `workflow.benchmark_steps` if supported.

Key config fields to verify before launch:

- `dataset.name` is the PadChest dataset id used consistently in artifact paths.
- `dataset.root` points to the metadata/images or manifests expected by `src/data/padchest.py`.
- `causal_schema.variables` and `causal_schema.edges` exactly match the model implementation.
- `model.context_dim` equals the encoded parent vector dimension.
- `optimizer.batch_size` fits TPU v6e-1 memory and leaves no unexpected tiny tail batch if `drop_last` semantics are used.
- `artifacts.root` and `artifacts.remote_root` are distinct enough to recover local and GCS outputs.
- `workflow.checkpoint_freq` is frequent enough to preserve progress but not so frequent that GCS sync dominates training.

## Logging and monitoring

The SCM runner writes:

- Console logs and `<run>/trainlog.txt`.
- TensorBoard event files with `train/*`, `valid/*`, `elbo/*`, `epoch/*`, and speed metrics.
- Data/model joint diagnostic PDFs.
- Orbax checkpoints under `<run>/checkpoints/<step>/`.

Monitor loss, per-variable log probabilities, gradient norm, iteration/sample throughput, validation loss, and checkpoint enqueue/sync messages. On TPU, initial compilation latency is expected; monitor steady-state epoch throughput after compilation rather than first-step wall time.

Do not interrupt live sessions to inspect them. Prefer read-only artifact checks such as listing the run directory or opening copied logs after confirming they are flushed.

## Checkpoint and artifact contract

A PadChest SCM artifact is usable downstream only if it contains:

- Orbax checkpoint metadata under `checkpoints/<step>/_CHECKPOINT_METADATA`.
- `format_version: 2` or the PadChest-specific successor accepted by downstream loaders.
- EMA inference params (`ema_params` and `params` alias if retained), live model params, optimizer state, epoch, global step, best validation loss, EMA step/init state, and full hparams.
- `hparams.json`, `trainlog.txt`, TensorBoard events, and diagnostic plots.
- Schema/version/fingerprint information sufficient to match later predictor/image/counterfactual configs.

Downstream stages should restore EMA parameters for inference/counterfactual sampling, not raw in-training parameters.

## Evaluation diagnostics

Evaluate beyond validation negative log likelihood:

1. Per-variable log probabilities and calibration for categorical/pathology nodes.
2. Marginal distribution comparison by split and against generated SCM samples.
3. Conditional checks for every declared graph edge, e.g. pathology prevalence by age/sex bins if those edges exist.
4. Support checks for intervention targets: each target class should have enough observed support and plausible generated support.
5. Missingness/exclusion audit comparing raw metadata counts with trainable rows.
6. Stability across seeds or at least a short repeated smoke run if the full run is expensive.

Replace the current two-variable joint PDF pattern with PadChest-appropriate diagnostics: selected pairwise plots, prevalence tables, calibration curves, and generated-vs-observed summaries.

## Intervention support checks

Before handing the SCM to counterfactual training, run explicit counterfactual support checks:

- Each allowed `do(...)` intervention can be encoded, passed through `counterfactual`, and converted back into `pa` without shape/dtype changes.
- Non-intervened exogenous noise is preserved for continuous variables where the model supports abduction.
- Interventions do not create impossible combinations unless the experiment intentionally studies off-support edits.
- Generated parent vectors stay in the same normalization range and categorical simplex/multi-hot constraints as real batches.
- Counterfactual targets used by the image stage match predictor output heads exactly.

## Failure modes and mitigations

- Missing PadChest config/provider in the active branch: do not launch; restore or merge the branch containing `configs/padchest_scm_tpu_v6e1.yaml` and `src/data/padchest.py` first.
- Dataset leakage: reject artifacts if splitting is image-level instead of patient-level.
- Rare-label collapse: add diagnostics and consider graph/label grouping before relying on interventions for rare findings.
- Schema drift: block downstream stages if `context_dim`, variable order, normalization, or label coding differs.
- GCS auth failures: use ADC compatible with `fsspec/gcsfs`; avoid quota-project override when `serviceusage.services.use` is not granted.
- TPU topology mismatch: fix config/runtime environment before launch; do not work around by silently running a full job on CPU.
- Non-finite losses or exploding gradients: stop only the job you own, preserve logs/checkpoint, and inspect preprocessing ranges and categorical encodings before retrying.

## Handoff to later stages

The SCM feeds later stages as follows:

- Predictor stage: uses the same PadChest parent variables as supervised targets from images; its variable order and normalization must match the SCM `pa` vector.
- Image model stage: conditions the VAE/HVAE on the same parent maps expanded from `pa`; it does not update the SCM.
- Counterfactual stage: freezes SCM EMA parameters, abducts observed parent noise, applies `do(...)` interventions, and supplies counterfactual parent vectors to the image mechanism.
- Inference/reporting: decodes normalized parents back to physical/clinical labels so generated counterfactuals are auditable.

Only promote the SCM artifact when the checkpoint contract, split manifest, schema fingerprint, evaluation diagnostics, and intervention support checks all pass.
