#!/usr/bin/env bash
set -euo pipefail

TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$TOOL_ROOT/.venv"
REQ="$TOOL_ROOT/requirements.txt"
PYTHON_BIN="${PYTHON:-python3}"

unset PYTHONPATH

if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REQ"
"$VENV/bin/python" -c "import numpy, pandas, patsy, nibabel, nilearn, statsmodels, scipy"

echo "Neuroimaging group analysis setup complete."
echo "Run an analysis with:"
echo "$VENV/bin/python run_analysis.py my_analysis.txt"
