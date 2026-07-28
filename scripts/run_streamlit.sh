#!/usr/bin/env bash
# Lance Streamlit avec le Python du venv (évite Anaconda sans cv2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Erreur : .venv introuvable. Crée-le avec : python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
export PYTHONPATH=src
exec "$ROOT/.venv/bin/python" -m streamlit run streamlit_app.py "$@"
