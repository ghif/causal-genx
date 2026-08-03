#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CONDA_ENV="${CONDA_ENV:-med-jax}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; install Conda or add it to PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "Conda environment '${CONDA_ENV}' was not found." >&2
  exit 1
fi

cd "${REPOSITORY_ROOT}"
echo "Starting Causal-GenX counterfactual API at http://${HOST}:${PORT} using Conda environment ${CONDA_ENV}."
exec conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH=src:. \
  uvicorn backend.app.main:app --host "${HOST}" --port "${PORT}" --workers 1
