#!/usr/bin/env bash
# One-command environment setup for the SQD spin-audit reproduction kit.
# Linux or WSL recommended (qc-pyci ships Linux wheels); Python 3.11.
#
#   ./setup.sh          full stack  -- reproduce every number in the paper
#   ./setup.sh --min    minimal     -- numpy + scipy + pyscf, enough for the
#                                       quick reproducers (Level 1, the
#                                       sampling-wall calculator, the moment
#                                       example, and spin_inversion_is_not_singlet)
#
# Override the interpreter with PYTHON=/path/to/python3.11 ./setup.sh
set -euo pipefail

PY="${PYTHON:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || { echo "error: $PY not found (set PYTHON=...)"; exit 1; }

"$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
# Always drive pip through the venv's python (a stale ./.venv/bin/pip shebang can
# install into the wrong interpreter).
python -m pip install --upgrade pip

if [ "${1:-}" = "--min" ]; then
  python -m pip install numpy==2.3.0 scipy==1.17.1 pyscf==2.13.1
  echo
  echo "Minimal environment ready (numpy, scipy, pyscf)."
else
  python -m pip install -r requirements.txt
  echo
  echo "Full environment ready."
fi

echo "Activate it with:  . .venv/bin/activate"
echo "Then start with:   python reproduce_level1.py   (see START_HERE.md)"
