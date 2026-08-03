#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
cd "${ROOT}"
echo "Serving frontend at http://${HOST}:${PORT}/frontend/; API base URL is configured in frontend/config.js"
exec python3 -m http.server "${PORT}" --bind "${HOST}"
