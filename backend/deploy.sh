#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project ID}"
SERVICE_NAME="${SERVICE_NAME:-causal-genx-counterfactual}"
REGION="${REGION:-asia-southeast2}"
REPOSITORY="${REPOSITORY:-causal-genx}"
: "${SERVICE_ACCOUNT:?Set SERVICE_ACCOUNT to the Cloud Run runtime service account email}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://127.0.0.1:8000,http://localhost:8000}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:$(git rev-parse --short HEAD)"
gcloud builds submit --config backend/cloudbuild.yaml --substitutions "_IMAGE=${IMAGE}" .
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --no-allow-unauthenticated \
  --cpu 4 \
  --memory 8Gi \
  --concurrency 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --cpu-boost \
  --set-env-vars "TRUST_INCOMPLETE_CHECKPOINT=true,DEFAULT_SEED=7,RATE_LIMIT_REQUESTS=30,RATE_LIMIT_WINDOW_SECONDS=60,ALLOWED_ORIGINS=${ALLOWED_ORIGINS}" \
  --startup-probe=httpGet.path=/readyz,httpGet.port=8080,initialDelaySeconds=0,periodSeconds=10,timeoutSeconds=5,failureThreshold=30
