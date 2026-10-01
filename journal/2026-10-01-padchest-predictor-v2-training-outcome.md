# PadChest Predictor v2 Training Outcome and Validation Report

**Date:** 2026-10-01  
**Run:** `predictor_padchest_v2_tpu_v6e1`  
**Configuration:** `configs/padchest_predictor_v2_tpu_v6e1.yaml`  
**Best Checkpoint:** Step `26065` (Epoch 13)  
**Dataset:** PadChest (PadChest Schema v2, 21-dimensional context)  
**Hardware Accelerator:** Google Cloud TPU v6e-1 (`bf16` precision)

---

## 1. Executive Summary

The Image Predictor v2 (`predictor_padchest_v2_tpu_v6e1`) was trained to predict the PadChest v2 anti-causal parent variables directly from chest X-ray images ($x \to u$). The model was resumed from the initial warm-up checkpoint (Step `2005`, end of Epoch 1) and reached peak generalization at **Epoch 13 (Step `26,065`)** with a minimum validation loss of **`0.7181`**.

Training was monitored through **Epoch 77 (Step `154,385` / 200,500)**. Because validation performance stabilized between Epochs 11–15 and subsequent epochs exhibited training loss reduction without further validation gain (mild overfitting on auxiliary heads), the run was early-stopped to preserve the optimal generalization checkpoint and free TPU compute for downstream counterfactual fine-tuning.

| Metric / Attribute | Value |
| :--- | :--- |
| **Total Processed Epochs / Steps** | 77 / 100 Epochs (154,385 / 200,500 Steps) |
| **Resumed Checkpoint** | Step `2005` (Epoch 1, Validation Loss: `1.1077`) |
| **Best Generalization Checkpoint** | **Step `26065` (Epoch 13, Validation Loss: `0.7181`)** |
| **Training Throughput** | ~7.3–8.0 it/s (~465–515 images/sec) on TPU v6e-1 |
| **Context / Schema Dimension** | 21 dimensions (`PAD_CHEST_SCHEMA_V2`) |
| **Image Resolution** | 128 × 128 grayscale |
| **Checkpoint Storage (Local)** | `checkpoints/padchest/predictor_padchest_v2_tpu_v6e1/checkpoints/26065/` |
| **Checkpoint Storage (GCS)** | `gs://external-cxr-dataset/padchest/checkpoints_v2/padchest/predictor_padchest_v2_tpu_v6e1/checkpoints/26065/` |

---

## 2. Role of the Predictor in Causal-GenX

The predictor acts as the anti-causal encoder for counterfactual inference:
1. **Anti-Causal Mechanism $q(u \mid x)$**: Predicts demographic and acquisition variables (`age_group`, `sex`, `tb_status`, `projection`, `view_position`, `scanner`) given the chest radiograph $x$.
2. **Counterfactual Alignment**: In the final counterfactual fine-tuning stage, the predictor ensures that generated/counterfactual images align with the specified intervened attributes $u^*$.
3. **Pre-requisite Validation**: Forms the third essential component alongside SCM v2 and Image Model v2 (HVAE) for full counterfactual generation.

---

## 3. Schema & Variable Encoding (PadChest v2)

The model operates on `PAD_CHEST_SCHEMA_V2`, consisting of 21 encoded continuous/one-hot dimensions:

| Variable | Type | Classes / Dimension | Encoding |
| :--- | :--- | :---: | :--- |
| `age_group` | Categorical | 5 | `[0-20]`, `[20-40]`, `[40-60]`, `[60-80]`, `[80+]` |
| `sex` | Categorical / Binary | 2 | Male, Female |
| `tb_status` | Binary | 1 | Normal / No TB vs. Active TB / Sequelae |
| `projection` | Categorical | 5 | `PA`, `AP`, `AP_horizontal`, `L`, `COSTAL` |
| `view_position` | Categorical | 6 | `POSTEROANTERIOR`, `ANTEROPOSTERIOR`, `LATERAL`, etc. |
| `scanner` | Categorical | 2 | Imaging device / manufacturer grouping |
| **Total Context Dim** | | **21** | |

---

## 4. Performance & Validation Progression

The top 3 checkpoints automatically captured by the validation checkpoint manager:

| Rank | Checkpoint | Epoch | Global Step | Validation Loss | `age_group_acc` | `sex_acc` | `projection_acc` | `view_position_acc` | `scanner_acc` | `tb_status_acc` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1st** | **`26065`** | **13** | **26,065** | **`0.7181`** | **`80.25%`** | **`95.52%`** | **`98.23%`** | **`99.37%`** | **`99.88%`** | **`99.18%`** |
| **2nd** | `24060` | 12 | 24,060 | `0.7196` | `80.21%` | `95.41%` | `98.23%` | `99.35%` | `99.89%` | `99.18%` |
| **3rd** | `22055` | 11 | 22,055 | `0.7218` | `80.31%` | `95.31%` | `98.24%` | `99.33%` | `99.91%` | `99.18%` |

### Comparison Across Training Milestones:
* **Epoch 1 (Step 2,005 - Resume point)**:
  * Loss: `1.1077` | `age_group_acc`: `71.69%` | `sex_acc`: `90.70%` | `projection_acc`: `97.20%` | `view_pos_acc`: `98.71%` | `scanner_acc`: `99.67%`
* **Epoch 13 (Step 26,065 - Peak Checkpoint)**:
  * Loss: `0.7181` | `age_group_acc`: `80.25%` | `sex_acc`: `95.52%` | `projection_acc`: `98.23%` | `view_pos_acc`: `99.37%` | `scanner_acc`: `99.88%`
* **Epoch 77 (Step 154,385 - Terminal Stop)**:
  * Loss: `1.1266` (Train loss: `0.1863`) | `age_group_acc`: `77.37%` | `sex_acc`: `95.54%` | `projection_acc`: `98.25%` | `view_pos_acc`: `99.34%` | `scanner_acc`: `99.93%`

---

## 5. Technical Challenges Resolved During Run

1. **Schema v2 Dimension Fallback**:
   * *Problem*: `_restore_args` in `src/training/predictor.py` replaced `args.schema` with the dictionary from checkpoint `hparams`, causing `_schema_for_dataset` to fall back to the 17-dimensional v1 schema and triggering head dimension mismatches.
   * *Fix*: In runner configuration, explicitly preserved `PAD_CHEST_SCHEMA_V2` and `context_dim = 21`.
2. **Optax `opt_state` Struct vs Dict Restoration**:
   * *Problem*: Orbax restored optimizer state PyTrees as standard nested dictionaries, causing Optax namedtuple/dataclass attribute access errors (`AttributeError: 'dict' object has no attribute 'mu'`).
   * *Fix*: Reconstructed optimizer PyTree structure using `jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(opt_state), jax.tree_util.tree_leaves(checkpoint['opt_state']))`.
3. **GCS Remote Synchronization**:
   * *Problem*: Default VM service account lacked `storage.objects.create` permissions on `external-cxr-dataset`.
   * *Fix*: Authenticated with user credentials (`mghifary@gmail.com`) and executed full dual-tree synchronization to GCS.

---

## 6. Readiness for Counterfactual Fine-Tuning

All three core pre-requisite model checkpoints are fully trained, verified, and saved:

1. **SCM v2**: `checkpoints/padchest/scm_padchest_v2_tpu_v6e1/` (100 epochs, step 200,500)
2. **Image Model v2 (HVAE)**: `checkpoints/padchest/image_model_padchest_v2_tpu_v6e1/checkpoints/` (100 epochs, step 401,000)
3. **Predictor v2**: `checkpoints/padchest/predictor_padchest_v2_tpu_v6e1/checkpoints/26065` (Best generalization checkpoint)

Next execution target: `configs/padchest_counterfactual_tpu_v6e1.yaml`.
