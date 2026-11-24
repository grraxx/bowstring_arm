#!/usr/bin/env bash
# Start the bowstring console: http://localhost:8765 (from WSL, open it in the Windows browser).
# Interpreter: $BOWSTRING_PYTHON if set, else the sofa-env conda env if present (needed only for the
# SOFA plants), else python3 on PATH (enough for the model plant and the Live tab).
cd "$(dirname "$0")/.." || exit 1
PY="${BOWSTRING_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$HOME/miniforge3/envs/sofa-env/bin/python" "$HOME/mambaforge/envs/sofa-env/bin/python" \
           "$HOME/miniconda3/envs/sofa-env/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
exec "$PY" console/server.py "$@"
