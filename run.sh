#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

if [ ! -f "app.py" ]; then
  echo "ERROR: app.py was not found in $(pwd)"
  exit 1
fi

if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt was not found in $(pwd)"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating local virtual environment in .venv..."
  "$PYTHON_BIN" -m venv .venv
fi

echo "Installing/updating dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

PORT="${NEURAL_EVOLUTION_PORT:-5000}"

echo
echo "Server will be available at: http://127.0.0.1:${PORT}"
echo "Press Ctrl+C to stop the server."
echo "============================================"
echo

.venv/bin/python app.py
