# Causal-GenX counterfactual backend

This FastAPI service serves the MorphoMNIST SCM, image-parent predictor, and
fine-tuned counterfactual HVAE from GCS. It accepts one PNG/JPEG image plus one
physical-unit intervention through `POST /v1/counterfactual`.

For the architecture, model artifacts, and request flow, see
[docs/arch.md](docs/arch.md).

## Prerequisites

- Run commands from the repository root.
- Use the `med-jax` Conda environment (Python 3.10+).
- Authenticate locally to Google Cloud with an account that has read access to
  the four configured checkpoint prefixes. Application Default Credentials are
  used by `gcsfs`; no credential file is configured in this repository.

```bash
gcloud auth application-default login
```

## Install dependencies

The backend requirement file includes the repository's CPU JAX stack and adds
the HTTP runtime dependencies (`uvicorn` and `python-multipart`).

```bash
conda activate med-jax
python -m pip install -r backend/requirements.txt
```

Or run the same commands without activating Conda in the current shell:

```bash
conda run -n med-jax python -m pip install -r backend/requirements.txt
```

## Run locally

Start a single-worker server bound only to localhost:

```bash
./backend/run.sh
```

Override the defaults when needed:

```bash
HOST=127.0.0.1 PORT=8081 CONDA_ENV=med-jax ./backend/run.sh
```

The first start downloads and restores the SCM, predictor, image metadata, and
counterfactual HVAE checkpoint from GCS. It can take noticeably longer than
later starts due to Orbax restore and JAX compilation.

In a second terminal, verify process liveness and model readiness:

```bash
curl -i http://127.0.0.1:8080/livez
curl -i http://127.0.0.1:8080/readyz
```

`/livez` returns `200` as soon as FastAPI is running. `/readyz` returns `200`
only after all model artifacts restore successfully; otherwise it returns `503`
with the startup error.

## Call the counterfactual API

The API accepts a grayscale or colour PNG/JPEG that is exactly 28x28 or 32x32
pixels. A 28x28 image is padded to 32x32 using the MorphoMNIST evaluation
transform. Numeric interventions use physical MorphoMNIST units:

- `thickness`: `0.87598526` through `6.255515`
- `intensity`: `66.601204` through `254.90317`
- `digit`: integer `0` through `9`

For example, generate a counterfactual that sets thickness to `3.0`:

```bash
curl --max-time 300 -X POST http://127.0.0.1:8080/v1/counterfactual \
  -F image=@/absolute/path/to/morphomnist-image.png \
  -F intervention_name=thickness \
  -F intervention_value=3.0 \
  -F seed=11
```

The JSON response contains `image_png_base64`, factual and counterfactual
parents in both normalized and physical units, the immutable model version,
seed, and request latency. Invalid image formats, dimensions, payloads, or
interventions return HTTP `422`; requests before successful model startup
return HTTP `503`.

## Test the API layer

These tests mock the heavyweight model registry, so they do not download GCS
artifacts:

```bash
conda run -n med-jax env PYTHONPATH=src:. \
  pytest -q tests/unit/test_counterfactual_backend.py
```

For Cloud Run packaging and deployment details, see `Dockerfile`, `deploy.sh`,
and [docs/cloudrun-plan.md](docs/cloudrun-plan.md). Keep production deployments
private and grant their service account `roles/storage.objectViewer` plus the
read-only `roles/storage.legacyBucketReader` required by `gcsfs` bucket metadata
lookups.
