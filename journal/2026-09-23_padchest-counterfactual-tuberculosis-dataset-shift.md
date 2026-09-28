# PadChest Counterfactual Tuberculosis Dataset-Shift Research Plan

**Date:** 2026-09-23

## Research objective

Develop a robust counterfactual chest-X-ray generator using PadChest to stress-test tuberculosis classification under clinically plausible dataset shifts.

The generator should separate disease-related image evidence from acquisition and population variation.

The primary downstream task is tuberculosis classification.

## Motivation and research gap

A review of available literature found no clearly identified published study that specifically trains a causal or counterfactual image generator on PadChest for tuberculosis dataset-shift analysis.

Related work includes counterfactual contrastive learning, causal image synthesis for robust medical representations, controllable chest-X-ray diffusion models, and generative reconstruction of patient-specific confounders.

The proposed work combines these directions with PadChest-specific metadata and explicit interventions.

## Available PadChest attributes

The dataset metadata includes the following groups.

### Patient attributes

- `PatientID`
- `PatientBirth`
- `PatientSex_DICOM`
- `Pediatric`

### Acquisition and view attributes

- `StudyDate_DICOM`
- `ViewPosition_DICOM`
- `Projection`
- `MethodProjection`

### Device and image-physics attributes

- `Modality_DICOM`
- `Manufacturer_DICOM`
- `PhotometricInterpretation_DICOM`
- `PixelRepresentation_DICOM`
- `PixelAspectRatio_DICOM`
- `SpatialResolution_DICOM`
- `BitsStored_DICOM`
- `WindowCenter_DICOM`
- `WindowWidth_DICOM`
- `Rows_DICOM`
- `Columns_DICOM`
- `XRayTubeCurrent_DICOM`
- `Exposure_DICOM`
- `ExposureInuAs_DICOM`
- `ExposureTime`
- `RelativeXRayExposure_DICOM`

### Disease and annotation attributes

- `Labels`
- `Localizations`
- `LabelsLocalizationsBySentence`
- `labelCUIS`
- `Report`
- `ReportID`
- `MethodLabel`

The PadChest copy contains 160,845 PNG images and a metadata CSV.

The tuberculosis label is sparse: the metadata summary reports 308 `tuberculosis` labels and 1,070 `tuberculosis sequelae` labels.

## Proposed causal variables

Start with a compact, well-supported schema.

- `age_at_study`: derived from `StudyDate_DICOM` and `PatientBirth`.
- `sex`: derived from `PatientSex_DICOM`.
- `pediatric`: derived from `Pediatric`.
- `tb_status`: positive when `Labels` contains `tuberculosis` or `tuberculosis sequelae`.
- `projection`: derived from `Projection`.
- `view_position`: derived from `ViewPosition_DICOM`.
- `study_year`: derived from `StudyDate_DICOM`.
- `manufacturer`: derived from `Manufacturer_DICOM`.
- `image_physics`: a grouped representation of exposure, resolution, dimensions, windowing, and photometric attributes.

Keep the initial schema small enough to maintain overlap between intervention groups.

Add physics variables only after measuring missingness, cardinality, and support.

## Proposed causal graph

```text
age_at_study ───► tb_status ───► image
sex ────────────► tb_status ───► image
pediatric ──────► tb_status ───► image

projection ─────────────────────► image
view_position ───────────────────► image
study_year ──────────────────────► image
manufacturer ────────────────────► image
image_physics ───────────────────► image
```

This graph treats acquisition and device variables as image-domain causes rather than causes of tuberculosis.

It avoids using radiology reports and derived labels as image-generation parents because they can leak the target.

## Step-by-step training plan

### Step 1: Freeze the research contract

Define the primary claim before implementation.

The claim is that a generator trained on PadChest can produce clinically plausible images under observed-support interventions and expose tuberculosis-classifier sensitivity to domain shifts.

Define two separate intervention families.

- Disease interventions change `tb_status` while holding nuisance attributes fixed.
- Nuisance interventions change acquisition or population attributes while holding `tb_status` fixed.

Do not treat generated images as clinical evidence or use them for patient care.

### Step 2: Inventory and validate the PadChest copy

Read the bucket manifest, README, CSV, and image prefix.

Verify object counts, file integrity, image readability, metadata coverage, and agreement between CSV image IDs and image objects.

Create a versioned inventory containing:

- Image count.
- Patient count.
- Study count.
- Missingness by attribute.
- Category frequencies.
- Tuberculosis and tuberculosis-sequelae counts.
- Image dimensions and intensity summaries.

Keep the bucket private and preserve the PadChest research-use restrictions.

### Step 3: Build a leakage-safe metadata table

Parse the CSV into one canonical row per image.

Normalize list-valued fields such as `Labels` and `Localizations`.

Derive `age_at_study`, `study_year`, and `tb_status`.

Retain the original fields for auditability, but expose only the approved causal variables to the generator.

Do not pass `Report`, `ReportID`, `Localizations`, or report-derived label fields into the primary image model.

### Step 4: Define tuberculosis labels and sensitivity analyses

Create a primary binary label where `tb_status=1` includes both `tuberculosis` and `tuberculosis sequelae`.

Create secondary labels for:

- Tuberculosis only.
- Tuberculosis sequelae only.
- Any tuberculosis-related label.
- Excluding uncertain or weakly labeled examples.

Report all definitions because the positive class is sparse and the labels are not a perfect clinical reference standard.

### Step 5: Create patient-level data splits

Split by `PatientID`, never by image row.

Use train, validation, and test partitions with no patient overlap.

Create additional evaluation partitions for:

- Projection holdout.
- View-position holdout.
- Temporal holdout.
- Manufacturer holdout.
- Combined shift holdout.

Keep the ordinary patient-level test set as the baseline.

### Step 6: Establish a tuberculosis-classification baseline

Train or select a classifier using only the training partition.

Evaluate on the ordinary test set and every shift-aware partition.

Record AUROC, AUPRC, sensitivity, specificity, calibration, and subgroup metrics.

Use AUPRC and confidence intervals because tuberculosis is rare.

This baseline identifies the shift problem before counterfactual generation is introduced.

### Step 7: Implement the PadChest dataset provider

Add a PadChest provider that reads PNG images and the canonical metadata table from GCS.

Support streaming or bounded caching so the full archive does not need to be copied locally.

Return NCHW images and typed causal variables through the existing dataset contract.

Add tests for metadata parsing, missing values, image loading, patient-level splitting, and label definitions.

### Step 8: Define the initial structural causal model

Train the metadata SCM on the approved variables.

Use the initial graph:

```text
age_at_study ───► tb_status ───► image
sex ────────────► tb_status ───► image
pediatric ──────► tb_status ───► image

projection ─────────────────────► image
view_position ───────────────────► image
study_year ──────────────────────► image
manufacturer ────────────────────► image
image_physics ───────────────────► image
```

For the first experiment, keep demographic variables and acquisition variables distinct.

Do not infer causal relationships between manufacturer, year, and disease unless the research design explicitly models those confounding paths.

### Step 9: Estimate support before allowing interventions

For every categorical intervention, measure the number of patients and images in each category.

For continuous interventions, estimate observed quantiles and restrict sampling to the common-support region.

Reject interventions that require extrapolation or have insufficient overlap.

This is essential for manufacturers, rare views, exposure variables, and tuberculosis-positive examples.

### Step 10: Train an image-to-variable predictor

Train a predictor from images to the causal variables.

Predict at least:

- `tb_status`.
- `projection`.
- `view_position`.
- `study_year` bucket.
- `manufacturer` group where support allows.

Use the predictor for preflight checks and evaluation, not as proof that a generated image is clinically correct.

Hold out patient IDs during validation.

### Step 11: Train the conditional image model

Train the conditional VAE/HVAE image mechanism on PadChest.

Condition on the approved causal and acquisition variables.

Preserve patient-specific latent information where the model supports abduction and reconstruction.

Monitor reconstruction quality, conditional attribute accuracy, tuberculosis classification behavior, and training stability.

Do not optimize only for pixel similarity because a model can reconstruct shortcuts without learning clinically meaningful disease structure.

### Step 12: Train counterfactual fine-tuning

Fine-tune the image mechanism using paired factual and counterfactual objectives.

For each factual sample:

1. Encode the observed image and metadata.
2. Infer patient-specific latent noise or exogenous variables.
3. Sample one supported intervention.
4. Use the SCM to propagate the intervention to downstream variables.
5. Decode the counterfactual image while preserving non-intervened latent factors.
6. Re-encode or classify the generated image.
7. Apply reconstruction, conditional consistency, and counterfactual objectives.

Use disease-preserving nuisance interventions frequently enough to learn domain changes.

Use disease interventions conservatively because positive tuberculosis data are sparse.

### Step 13: Train nuisance-shift counterfactuals

Start with interventions that have the strongest support:

- `do(projection = PA)` versus `do(projection = AP)`.
- `do(view_position = POSTEROANTERIOR)` versus `do(view_position = LATERAL)`.
- Supported `study_year` buckets.
- Supported manufacturer groups.

Keep `tb_status`, patient identity factors, and non-intervened variables fixed.

Evaluate whether the generated image changes its acquisition attributes without changing tuberculosis predictions.

### Step 14: Train disease counterfactuals

Generate `do(tb_status=0)` and `do(tb_status=1)` only for samples and attribute combinations with adequate support.

Use tuberculosis-specific localization annotations only for held-out evaluation, not as primary generator inputs.

For disease counterfactuals, require:

- Correct directional change in tuberculosis prediction.
- Stable projection and acquisition attributes.
- Stable non-target anatomy.
- Plausible changes under expert or localization-based review.

### Step 15: Add CXR-RAIT as a small target-domain evaluation

Do not train a high-capacity generator from CXR-RAIT initially.

Instead, estimate its domain profile using compatible attributes and frozen image representations.

Compare PadChest and CXR-RAIT on:

- Age and sex distributions.
- Tuberculosis prevalence and label definitions.
- Image intensity and size statistics.
- View or acquisition fields where available.
- Frozen image-embedding distributions.

Use the PadChest generator to translate samples toward the estimated CXR-RAIT domain only within supported overlap.

Compare real CXR-RAIT images with translated PadChest images using attribute classifiers, embedding distances, and tuberculosis predictions.

### Step 16: Evaluate counterfactual validity

For nuisance interventions, test that:

- The target acquisition classifier changes as intended.
- Tuberculosis status remains stable.
- The tuberculosis classifier prediction is invariant within a predefined tolerance.

For disease interventions, test that:

- The tuberculosis classifier changes in the intended direction.
- Acquisition classifiers remain stable.
- Changes are localized and plausible.

Use multiple random seeds and report uncertainty.

### Step 17: Evaluate dataset-shift robustness

Run the classifier on:

- Real PadChest test images.
- Real shift-holdout images.
- Generated nuisance-shift images.
- Generated images translated toward CXR-RAIT.
- Disease counterfactual pairs.

Report performance and calibration by intervention, patient subgroup, projection, view, year, and manufacturer.

Use counterfactual consistency as a diagnostic, not as a replacement for external validation.

### Step 18: Run ablations

Compare:

- No causal conditioning.
- Demographic variables only.
- Acquisition variables only.
- Demographic plus acquisition variables.
- Without manufacturer.
- Without year.
- Without physics variables.
- Disease interventions only.
- Nuisance interventions only.

These ablations show which metadata groups provide genuine robustness rather than merely increasing model capacity.

### Step 19: Perform safety and memorization checks

Check nearest neighbors between generated and training images.

Measure whether generated outputs copy distinctive source images.

Inspect whether the model generates artifacts associated with labels, reports, or filenames.

Do not publish or expose identifiable data or generated outputs that could undermine the dataset agreement.

### Step 20: Define the final research claims

Separate claims about:

- Image realism.
- Attribute controllability.
- Disease counterfactual validity.
- Nuisance-shift robustness.
- Transfer to CXR-RAIT.

Do not claim causal identification from observational PadChest metadata alone.

Frame the generator as an intervention-based stress-testing and representation-learning tool.

## Counterfactual interventions

### Acquisition and view shifts

- `do(projection = PA)` versus `do(projection = AP)`.
- `do(view_position = POSTEROANTERIOR)` versus `do(view_position = LATERAL)`.
- `do(projection = AP_horizontal)` where sufficient support exists.

These interventions should preserve disease status while changing the acquisition domain.

### Temporal shifts

- `do(study_year = y)` for supported years or predefined historical periods.

This tests changes in acquisition practice and population composition over time.

### Device shifts

- `do(manufacturer = m)` for manufacturers with adequate sample size and overlap.

This tests scanner and vendor domain effects.

### Image-physics shifts

- Intervene on exposure or windowing profiles only within observed support.
- Avoid extrapolating to physically implausible values.

### Disease counterfactuals

- `do(tb_status = 0)`.
- `do(tb_status = 1)`.

Disease interventions are central to the generator but require stricter evaluation because the tuberculosis labels are sparse and noisy.

## Evaluation plan

### Classification performance

Report AUROC, AUPRC, sensitivity, specificity, calibration, and subgroup performance.

Because tuberculosis is rare, AUPRC and confidence intervals are especially important.

### Counterfactual validity

For disease-preserving acquisition interventions:

- Tuberculosis status should remain unchanged.
- The generated image should reflect the target view, projection, year, or manufacturer domain.
- Predictions should remain stable within a clinically reasonable tolerance.

For disease interventions:

- The tuberculosis classifier should respond in the expected direction.
- Non-target anatomy and acquisition characteristics should remain stable.
- Changes should be localized and clinically plausible where localization annotations support evaluation.

### Dataset-shift robustness

Compare classifier performance on:

- Observed test images.
- Generated acquisition-shift images.
- Temporal and manufacturer holdouts.
- Cross-intervention consistency sets.

Measure prediction invariance under nuisance interventions and sensitivity under disease interventions.

## Leakage controls

Do not use `Report`, `LabelsLocalizationsBySentence`, or report-derived fields as generator inputs in the primary experiment.

Keep report text and localization fields for held-out evaluation and label-quality analysis.

Ensure image filenames, patient IDs, and study IDs cannot leak across splits.

Do not treat `tb_status` derived from labels as an unquestionable ground truth.

Run sensitivity analyses separating tuberculosis from tuberculosis sequelae and excluding uncertain or weakly labeled examples.

## Risks and limitations

- Tuberculosis prevalence is very low and labels may be noisy.
- PadChest metadata is observational, so intervention semantics are not automatically causal.
- Manufacturer and acquisition variables may be confounded with hospital, time, and patient population.
- Generated counterfactual images can encode model artifacts rather than clinically valid changes.
- Exposure and windowing interventions may create unrealistic images if applied outside observed support.
- The research-use agreement must be respected, the bucket must remain private, and the data must not be used for diagnosis or patient care.

## Expected contribution

The project can provide a PadChest-specific benchmark for causal counterfactual generation and dataset-shift stress testing in tuberculosis classification.

The central contribution is not merely synthetic augmentation.

It is an intervention-based evaluation framework that distinguishes disease changes from acquisition, temporal, device, and demographic shifts.
