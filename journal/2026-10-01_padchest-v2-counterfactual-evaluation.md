# PadChest v2 counterfactual image generator: quantitative and qualitative evaluation

Date: 2026-10-01  
Project: Causal-GenX  
Dataset: PadChest v2  
Artifact evaluated: `gs://external-cxr-dataset/padchest/checkpoints_v2/padchest/counterfactual_padchest_v2_tpu_v6e1`

## Executive summary

This report evaluates the finetuned PadChest v2 counterfactual image generator. The model starts from the PadChest v2 conditional HVAE and is finetuned while the structural causal model (SCM) and image-parent predictor remain frozen. The generator receives a factual chest radiograph and its observed causal context, abducts image-specific latent residuals, intervenes on one causal variable, and decodes a counterfactual image while preserving the residual anatomy as far as the model allows.

The latest complete uploaded counterfactual checkpoint is step **80,200** (epoch 20). The training process was still running beyond this point when the artifacts were inspected; therefore, this report intentionally evaluates step 80,200 rather than an incomplete live checkpoint.

The main findings are:

- The training-time validation ELBO at step 80,200 was **2.2432** (NLL **1.6826**, KL **0.5606**).
- Across seven interventions on each of five real PadChest studies, the SCM returned the requested intervened value in **35/35 cases (100%)**.
- After correcting the 16-bit PadChest input conversion and panel layout, the generated images had mean normalized pixel MAE **0.01867** and mean PSNR **72.76 dB** in this small diagnostic set.
- The corrected panels have normal chest-radiograph contrast. The previous report version used a direct `I;16` to `L` conversion and a single eight-column strip, which made the images appear almost white and visually compressed.
- The small pixel changes are expected for several variables because the model is constrained to preserve the abducted patient-specific residual. They should not be interpreted as evidence that the images contain clinically correct disease edits.
- The strongest training-time auxiliary accuracy was for `tb_status` (**1.0000** for all logged intervention types), while projection, view position, and scanner accuracies were materially weaker. This suggests that causal metadata intervention is functioning, but image-level realization of acquisition variables remains incomplete.

## 1. Evaluation target and artifact provenance

### 1.1 Restored checkpoints

The counterfactual generator was evaluated together with the exact upstream artifacts referenced by its checkpoint metadata:

| Component | Restored checkpoint | Role |
|---|---:|---|
| Counterfactual HVAE | `counterfactual_padchest_v2_tpu_v6e1/checkpoints/80200` | Trainable image mechanism after CF finetuning |
| SCM | `scm_padchest_v2_tpu_v6e1/checkpoints/198495` | Generates parent variables under interventions |
| Image-parent predictor | `predictor_padchest_v2_tpu_v6e1/checkpoints/26065` | Auxiliary likelihood for image/parent consistency |
| Base image model initialization | `image_model_padchest_v2_tpu_v6e1/checkpoints/380950` | Pre-CF HVAE referenced by the CF training configuration |

All restored checkpoints were complete Orbax checkpoints. The evaluation ran on CPU in FP32 after downloading only the required checkpoint artifacts and five PNG samples to temporary storage; the source dataset remains in GCS.

### 1.2 Input-format correction

The PadChest PNG export used by this evaluation is stored as 16-bit `I;16` images. The first evaluator version called `Image.open(path).convert("L")` directly. That clips the 16-bit values instead of applying the `/256` scaling used by `PadChestDataset`, producing an almost-white input image. The evaluator now follows the dataset loader exactly: `uint16 → float32 / 256 → uint8 → resize → [-1, 1]`.

For example, the corrected TB-sequela input has raw statistics `I;16/uint16`, min `0`, max `57594`, mean `33971.54`, and standard deviation `14988.53`; after conversion and model normalization its mean is `0.03693` and standard deviation is `0.45687`, consistent with a usable grayscale radiograph.

The qualitative panel was also changed from one eight-column strip to a 2×4 grid. This preserves the radiograph's spatial scale in Markdown previews.

### 1.3 Model and causal schema

The model uses a 21-dimensional PadChest v2 context composed of:

| Variable | Encoding | Causal role |
|---|---|---|
| `age_group` | 5-way one-hot: `0-4`, `5-17`, `18-39`, `40-64`, `65+` | Parent of `tb_status` |
| `sex` | 2-way one-hot: `non_m_or_unknown`, `M` | Parent of `tb_status` |
| `tb_status` | Binary: tuberculosis or sequelae | Target causal health variable |
| `projection` | 5-way one-hot: `PA`, `AP`, `AP_horizontal`, `L`, `COSTAL` | Acquisition variable |
| `view_position` | 6-way one-hot: `POSTEROANTERIOR`, `ANTEROPOSTERIOR`, `LATERAL`, `AP`, `PA`, `OTHER` | Acquisition variable |
| `scanner` | 2-way one-hot: `ImagingDynamicsCompanyLtd`, `PhilipsMedicalSystems` | Acquisition variable |

The SCM edges are `age_group → tb_status` and `sex → tb_status`. Intervening on a parent of `tb_status` allows the SCM to update the downstream TB status rather than manually overwriting every context field.

## 2. Quantitative evaluation

### 2.1 Training-time validation at the evaluated checkpoint

The counterfactual training log reports the following validation values at global step 80,200:

| Validation intervention | Loss | Auxiliary loss | ELBO | NLL | KL | Age acc. | Sex acc. | TB acc. | Projection acc. | View acc. | Scanner acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `do_age_group` | 30.4938 | 2.3144 | 2.2432 | 1.6826 | 0.5606 | 0.4688 | 0.8125 | 1.0000 | 0.4062 | 0.4375 | 0.6562 |
| `do_sex` | 30.5194 | 2.3170 | 2.2432 | 1.6826 | 0.5606 | 0.5625 | 0.6250 | 1.0000 | 0.4062 | 0.4062 | 0.6562 |
| `do_tb_status` | 29.8137 | 2.2464 | 2.2432 | 1.6826 | 0.5606 | 0.5625 | 0.8750 | 1.0000 | 0.4062 | 0.4375 | 0.6562 |
| `do_projection` | 30.6226 | 2.3273 | 2.2432 | 1.6826 | 0.5606 | 0.5312 | 0.8438 | 1.0000 | 0.3438 | 0.4375 | 0.6562 |
| `do_view_position` | 34.8036 | 2.7454 | 2.2432 | 1.6826 | 0.5938 | 0.8438 | 1.0000 | 0.4062 | 0.2188 | 0.6562 | 0.4375 |
| `do_scanner` | 34.5402 | 2.7191 | 2.2432 | 1.6826 | 0.5938 | 0.8438 | 1.0000 | 0.4062 | 0.4375 | 0.4375 | 0.4375 |
| observational | 31.0445 | 2.3695 | 2.2432 | 1.6826 | 0.5606 | 0.5938 | 0.8438 | 1.0000 | 0.2812 | 0.4688 | 0.6562 |

These are the model's own held-out validation summaries, not metrics computed from radiologist annotations of generated images. The predictor is trained to reconstruct the metadata variables, so the auxiliary accuracies measure internal consistency with the learned predictor rather than clinical realism.

### 2.2 Direct intervention-consistency and image-change audit

For an independent artifact-level audit, five real PadChest studies were selected to cover TB sequelae, normal PA, male PA, male lateral, and male AP examples. Seven interventions were applied to each study: `tb_status=1`, `tb_status=0`, `age_group=65+`, `sex=M`, `projection=AP`, `view_position=LATERAL`, and `scanner=Philips`.

| Metric | Result |
|---|---:|
| Factual studies | 5 |
| Interventions per study | 7 |
| Factual-to-counterfactual pairs | 35 |
| Requested SCM target satisfied | 35/35 (100.0%) |
| Mean pixel MAE, normalized [-1, 1] | 0.018671 |
| Mean PSNR | 72.76 dB |
| Mean single-sample ELBO | 2.3440 |
| Mean single-sample NLL | 1.7322 |
| Mean single-sample KL | 0.6119 |

The corrected direct audit confirms that the SCM intervention interface is working and that the input/output images are being evaluated in the intended intensity range. It does not establish that the image generator has made a clinically valid transformation: a target can be satisfied in the metadata tensor while the pixels change too little, change in the wrong anatomy, or introduce an implausible acquisition artifact. The earlier direct-audit MAE/PSNR values from the incorrectly converted inputs are superseded by the values above.

The per-intervention image-change pattern is informative. Projection edits produced the largest average changes among the tested acquisition variables, whereas TB-status edits were often subtle. This is compatible with the current residual-preserving design, but it also means that a TB classifier could receive little additional disease evidence from the counterfactual images unless the generator is evaluated with stronger image-level and clinical criteria.

## 3. Qualitative evaluation

Each panel contains one factual image followed by seven generated counterfactuals in a 2×4 layout. The captions identify the factual study and the intervention applied to each generated image. The intervention label describes the requested target; the SCM may also update downstream variables, especially `tb_status` after an age or sex intervention.

### Figure 1. TB-sequela PA study

![TB-sequela PA counterfactual panel](artifacts/padchest_cf_v2_eval/tb_sequela_pa.png)

**Caption.** PadChest image `144161045940829338272357413255772911245_6ca9jm.png`, a female 86-year-old PA study labeled with tuberculosis sequelae. The top row shows the factual input, `tb_status=1` (positive), `tb_status=0` (negative), and `age_group=65+`; the bottom row shows `sex=M`, `projection=AP`, `view_position=LATERAL`, and `scanner=PhilipsMedicalSystems`. The image-specific latent residual is held fixed across each intervention, so the panel is a paired counterfactual comparison rather than seven unconditional samples.

### Figure 2. Normal PA study

![Normal PA counterfactual panel](artifacts/padchest_cf_v2_eval/normal_pa.png)

**Caption.** PadChest image `20536686640136348236148679891455886468_k6ga29.png`, a female 84-year-old PA study labeled normal. The top row shows the factual input, `tb_status=1` (positive), `tb_status=0` (negative), and `age_group=65+`; the bottom row shows `sex=M`, `projection=AP`, `view_position=LATERAL`, and `scanner=PhilipsMedicalSystems`. The panel illustrates the magnitude and localization of changes produced by the finetuned model for disease-status, demographic, and acquisition-variable interventions.

### Figure 3. Male PA study

![Male PA counterfactual panel](artifacts/padchest_cf_v2_eval/male_pa.png)

**Caption.** PadChest image `135803415504923515076821959678074435083_fzis7b.png`, a male 86-year-old PA study with chronic/fibrotic findings. The top row shows the factual input, `tb_status=1` (positive), `tb_status=0` (negative), and `age_group=65+`; the bottom row shows `sex=M`, `projection=AP`, `view_position=LATERAL`, and `scanner=PhilipsMedicalSystems`. The unchanged-looking `sex=M` panel is expected because the factual study already has the requested sex value and therefore does not represent a substantive intervention.

### Figure 4. Male lateral study

![Male lateral counterfactual panel](artifacts/padchest_cf_v2_eval/male_lateral.png)

**Caption.** PadChest image `135803415504923515076821959678074435083_fzis7d.png`, a male 86-year-old lateral study from the same patient series as Figure 3. The top row shows the factual input, `tb_status=1` (positive), `tb_status=0` (negative), and `age_group=65+`; the bottom row shows `sex=M`, `projection=AP`, `view_position=LATERAL`, and `scanner=PhilipsMedicalSystems`. Because the factual view is already lateral, the `view_position=LATERAL` counterfactual is a no-op in the metadata and serves as a negative control for spurious visual change.

### Figure 5. Male AP study

![Male AP counterfactual panel](artifacts/padchest_cf_v2_eval/male_ap.png)

**Caption.** PadChest image `313903302629300007485735352869488750471_75sg0k.png`, a male 39-year-old AP study with an `OTHER` DICOM view-position value. The top row shows the factual input, `tb_status=1` (positive), `tb_status=0` (negative), and `age_group=65+`; the bottom row shows `sex=M`, `projection=AP`, `view_position=LATERAL`, and `scanner=PhilipsMedicalSystems`. The `projection=AP`, `sex=M`, and `scanner=PhilipsMedicalSystems` panels are metadata no-op controls because those values already match the factual study.

### Figure 6. Factual-to-counterfactual pixel-difference heatmaps

These maps show the absolute per-pixel difference between each factual input and its corresponding generated counterfactual, computed directly from model output tensors in normalized `[-1, 1]` intensity space (not from the rasterized panels). The factual image is shown as the first tile; the remaining tiles follow the intervention order in Figures 1–5. Within each study, all seven maps share a display ceiling set to that study's 99th-percentile difference, so brighter regions indicate larger changes. The scales are study-specific and must not be compared quantitatively between studies. Black `tb_status=1` in Figure 6a is expected because the factual TB-sequela study already has that target and its generated image is effectively unchanged. These maps visualize image-space changes only; they are not saliency maps or clinical explanations.

![Factual-to-counterfactual absolute pixel-difference heatmaps for the TB-sequela PA study](artifacts/padchest_cf_v2_eval/tb_sequela_pa_difference_heatmap.png)

![Factual-to-counterfactual absolute pixel-difference heatmaps for the normal PA study](artifacts/padchest_cf_v2_eval/normal_pa_difference_heatmap.png)

![Factual-to-counterfactual absolute pixel-difference heatmaps for the male PA study](artifacts/padchest_cf_v2_eval/male_pa_difference_heatmap.png)

![Factual-to-counterfactual absolute pixel-difference heatmaps for the male lateral study](artifacts/padchest_cf_v2_eval/male_lateral_difference_heatmap.png)

![Factual-to-counterfactual absolute pixel-difference heatmaps for the male AP study](artifacts/padchest_cf_v2_eval/male_ap_difference_heatmap.png)

## 4. Interpretation

The checkpoint demonstrates a functioning end-to-end counterfactual pipeline: the model loads the finetuned image mechanism, uses the SCM to generate intervened parent contexts, preserves a factual image's latent residual, and produces finite decoded images for all tested interventions. The perfect target-satisfaction rate in the direct audit is a property of the SCM output and should be separated from image realism.

The main scientific concern is intervention identifiability. `tb_status` is the variable of interest, but the PadChest labels identify TB or sequelae at the study level and do not supply pixel-level evidence for what a valid counterfactual “remove TB” or “add TB” image should look like. In addition, acquisition variables such as projection, view position, and scanner are not purely causal disease factors. They can be useful nuisance/domain variables, but changing them may require geometric or acquisition-consistent transformations that a 128×128 conditional HVAE is not guaranteed to learn.

Accordingly, the present results support the claim that the model can perform metadata-conditioned counterfactual decoding, not the stronger claim that it generates medically faithful TB counterfactuals. Before using generated images to improve CXR-RAIT TB classification, the next evaluation should test whether augmentation improves held-out, site-stratified TB performance without increasing shortcut reliance.

## 5. Recommended next experiments

1. **Pair-level clinical review.** Have radiologists score TB plausibility, anatomy preservation, projection/view correctness, and artifacts for blinded factual/counterfactual pairs.
2. **Classifier-based counterfactual tests.** Measure whether a fixed external TB classifier changes in the intended direction under `tb_status` interventions while remaining stable under sex, age, and scanner interventions.
3. **Domain-robustness evaluation.** Train CXR-RAIT classifiers with and without CF augmentation and report AUROC, AUPRC, sensitivity at fixed specificity, calibration, and subgroup/site performance.
4. **Acquisition-variable controls.** Treat projection, view position, and scanner as nuisance/domain variables. Require improvements on external domains before using them as augmentation targets.
5. **Scale-up beyond the five-study audit.** Repeat the direct audit over a fixed, patient-disjoint test subset, stratified by TB label, sex, age group, scanner, projection, and view position.

## 6. Reproducibility and files

The evaluation script is [evaluate_padchest_counterfactual.py](../scripts/evaluate_padchest_counterfactual.py). It writes:

- [metrics.json](artifacts/padchest_cf_v2_eval/metrics.json), containing all 35 pair-level records and aggregate metrics.
- [tb_sequela_pa.png](artifacts/padchest_cf_v2_eval/tb_sequela_pa.png)
- [normal_pa.png](artifacts/padchest_cf_v2_eval/normal_pa.png)
- [male_pa.png](artifacts/padchest_cf_v2_eval/male_pa.png)
- [male_lateral.png](artifacts/padchest_cf_v2_eval/male_lateral.png)
- [male_ap.png](artifacts/padchest_cf_v2_eval/male_ap.png)
- [tb_sequela_pa_difference_heatmap.png](artifacts/padchest_cf_v2_eval/tb_sequela_pa_difference_heatmap.png)
- [normal_pa_difference_heatmap.png](artifacts/padchest_cf_v2_eval/normal_pa_difference_heatmap.png)
- [male_pa_difference_heatmap.png](artifacts/padchest_cf_v2_eval/male_pa_difference_heatmap.png)
- [male_lateral_difference_heatmap.png](artifacts/padchest_cf_v2_eval/male_lateral_difference_heatmap.png)
- [male_ap_difference_heatmap.png](artifacts/padchest_cf_v2_eval/male_ap_difference_heatmap.png)

The evaluator uses the PadChest v2 metadata derivation and 16-bit image conversion already implemented in `src/data/padchest.py`, restores complete local copies of the GCS checkpoints, and runs the same `_cf_forward` path used by counterfactual training. It renders the heatmaps directly from generated tensors before panel rasterization. The saved `metrics.json` records raw and normalized input statistics for each factual study.

## Conclusion

At the latest complete checkpoint, the PadChest v2 counterfactual model is operational and internally consistent at the SCM level. Its current validation profile is strongest for TB-status consistency and weaker for acquisition-variable realization. The qualitative panels show controlled, generally small image changes, which is encouraging for anatomy preservation but insufficient to establish clinically meaningful TB editing. The model should therefore be treated as a research instrument for testing domain-robust training hypotheses, not yet as a validated source of synthetic TB evidence.
