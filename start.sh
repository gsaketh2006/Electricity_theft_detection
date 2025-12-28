#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Allow overriding PORT by env var (Render provides $PORT)
PORT=${PORT:-5000}
WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}

echo "Starting gunicorn with PYTHONPATH=$PYTHONPATH" >&2
exec gunicorn wsgi:app --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY} --log-level debug --log-file -
