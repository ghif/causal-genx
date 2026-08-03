# Counterfactual API Deployment in `backend/`

## Summary

Create a self-contained FastAPI backend under `backend/` that serves the MorphoMNIST counterfactual pipeline from private GCS artifacts. Deploy it as a private Cloud Run CPU service in `asia-southeast2`, called only by an authenticated frontend server/BFF.

The service will load once at startup:

- CF HVAE EMA checkpoint: `gs://medical-airnd/causal-gen/checkpoints/morphomnist/cf_jax-tpu-v6e4_23-07-2026/checkpoints/15795`
- Frozen SCM checkpoint
- Frozen predictor checkpoint
- Original image-model metadata/checkpoint required to rebuild the HVAE architecture

It will accept a multipart image plus a single named causal intervention and return JSON containing a base64 PNG, factual parents, intervened parents, request seed, model version, and latency.

## Backend implementation

- Add `backend/app/` with:
  - Application factory and `GET /livez`, `GET /readyz`, and `POST /v1/counterfactual`.
  - Typed multipart request parsing and typed JSON response models.
  - A startup-owned model registry that loads artifacts once, exposes readiness only after all models compile/restore, and releases Orbax managers cleanly on shutdown.
  - A counterfactual service layer that reuses the repository's JAX/Flax model classes and Orbax restore helpers without importing or running training loops.

- Configure model sources as required environment variables, with production defaults pinned to:
  - `CF_CHECKPOINT=gs://medical-airnd/causal-gen/checkpoints/morphomnist/cf_jax-tpu-v6e4_23-07-2026/checkpoints/15795`
  - SCM, predictor, and image-model checkpoint roots from `configs/morphomnist_counterfactual_tpu_v6e4.yaml`.
  - `TRUST_INCOMPLETE_CHECKPOINT=true`, because the target artifact has no `commit_success.txt`.

- Reconstruct the HVAE from its saved metadata, then restore only the CF checkpoint's `ema_params`; restore SCM EMA parameters and predictor EMA parameters plus BatchNorm statistics. Validate the upstream artifact roles and checkpoint tree compatibility before marking the service ready.

- Implement inference as:
  1. Decode a grayscale PNG/JPEG upload; accept only 28x28 or 32x32 images.
  2. Apply the existing deterministic MorphoMNIST evaluation preprocessing: 28x28 inputs receive a two-pixel border; normalize pixels to `[-1, 1]`.
  3. Run the predictor to obtain factual normalized thickness, intensity, and digit probabilities; use argmax digit.
  4. Convert the requested physical-unit intervention to model scale, preserving the factual exogenous variables through `MorphoMNISTPGM.counterfactual`.
  5. Abduct latents from the factual image with the fine-tuned HVAE, render using counterfactual parents, preserve residual noise as in `_cf_forward`, clip to `[-1, 1]`, and encode the resulting 32x32 grayscale PNG.
  6. Return both physical and normalized parent values. Use the exact training ranges: thickness `0.87598526-6.255515`; intensity `66.601204-254.90317`.

- Define `POST /v1/counterfactual` as multipart form data:
  - `image`: required PNG/JPEG file.
  - `intervention_name`: required one of `digit`, `thickness`, `intensity`.
  - `intervention_value`: required physical-unit float for thickness/intensity, integer `0-9` for digit.
  - `seed`: optional non-negative integer; default to a service-configured value.
  - Reject malformed files, unsupported dimensions/types, out-of-range interventions, oversized uploads, and unavailable/unready models with structured JSON errors.

- Return:
  - `model_version` pinned to the CF GCS path and resolved step.
  - `factual_parents` and `counterfactual_parents`, each in `physical` and `normalized` forms.
  - `intervention`, `seed`, `latency_ms`, and `image_png_base64`.
  - No checkpoint contents, internal GCS paths beyond the immutable model version, or stack traces.

## Container and Cloud Run deployment

- Add `backend/Dockerfile`, `.dockerignore`, `requirements.txt`, and a documented deployment command/script inside `backend/`.
  - Build from the repository root so `src/` is available in the image.
  - Install the CPU JAX dependency set plus `uvicorn` and `python-multipart`.
  - Set CPU-only JAX environment before app import.
  - Run one Uvicorn worker; model state must not be duplicated across workers.

- Deploy the container to Artifact Registry and Cloud Run in `asia-southeast2`.
  - Set request concurrency to `1`, CPU always allocated during a request, a model-appropriate request timeout, and `min-instances=0`.
  - Set a bounded maximum instance count to contain CPU cost and avoid untested parallel JAX execution.
  - Allow scale-to-zero; `/readyz` reports unavailable while the model is still loading/compiling after a cold start.
  - Keep the service private. Its runtime service account receives only `roles/storage.objectViewer` on the exact GCS prefixes needed by the four model artifacts.
  - Do not place any GCP credentials, checkpoint copies, or signed object URLs in the image.

- Document the frontend-server integration:
  - The browser uploads to the frontend server.
  - The frontend authenticates the user, forwards the multipart request to private Cloud Run using its service identity, and returns the JSON result.
  - The browser never receives Google Cloud tokens and never accesses the model bucket directly.

## Testing and acceptance criteria

- Unit-test request validation, image preprocessing, physical/normalized conversions, all three intervention types, deterministic seed behavior, response encoding, and typed error responses.
- Unit-test model registry failure paths: missing metadata, wrong artifact role/version, incomplete checkpoint restore behavior, incompatible parameter tree, and readiness state.
- Add a mocked end-to-end API test that verifies predictor -> SCM -> HVAE invocation order and validates factual versus counterfactual parent outputs.
- Add a CPU smoke test, guarded by GCS credentials, that restores the four real artifacts, sends one valid 28x28 image, and asserts a finite 32x32 PNG response.
- Verify container startup, private Cloud Run deployment, service-account GCS read access, authenticated BFF invocation, cold-start readiness behavior, and an end-to-end browser-facing request.

## Assumptions and defaults

- First release supports MorphoMNIST only, one image and one intervention per request; batching, CXR-RAIT, artifact hot-reload, persistent output storage, and caller-supplied factual parents are out of scope.
- The backend is placed entirely under `backend/`; existing training stages and YAML experiment contracts remain unchanged.
- Counterfactual inference is synchronous and CPU-backed; production latency is measured before considering a GPU or GKE migration.
