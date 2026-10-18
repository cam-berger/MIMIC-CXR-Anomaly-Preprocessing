#!/usr/bin/env bash
# Set up and verify a v2 environment on this machine (a laptop with the data, or a rented GPU).
#
#   bash v2/scripts/setup.sh            # .venv-v2 (sees a preinstalled torch), install, run the tests
#   bash v2/scripts/setup.sh --cpu      # Linux without a GPU: CPU-only torch (much smaller download)
#   bash v2/scripts/setup.sh --system   # install into the current Python environment instead
#   bash v2/scripts/setup.sh --no-tests # skip pytest and the wiring check
#
# Afterwards: source .venv-v2/bin/activate && python -m v2.doctor --mimic-cxr-jpg <path>
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV=.venv-v2
CPU=0
SYSTEM=0
TESTS=1
for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    --system) SYSTEM=1 ;;
    --no-tests) TESTS=0 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10+ is required ($("$PY" --version 2>&1) found)." >&2
  echo "macOS: brew install python@3.11 && PYTHON=python3.11 bash $0" >&2
  echo "conda: conda create -n v2 python=3.11 && conda activate v2 && bash $0 --system" >&2
  exit 1
fi

if [ "$SYSTEM" = 0 ]; then
  # --system-site-packages keeps a GPU provider's preinstalled CUDA torch visible inside the venv.
  [ -d "$VENV" ] || "$PY" -m venv --system-site-packages "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  PY=python
fi
"$PY" -m pip install -q --upgrade pip

torch_ok() {
  "$PY" - <<'EOF' 2>/dev/null
import sys
import torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
sys.exit(0 if (major, minor) >= (2, 6) else 1)
EOF
}
if torch_ok; then
  echo "Using installed torch $("$PY" -c 'import torch; print(torch.__version__)')"
elif [ "$CPU" = 1 ]; then
  "$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -m pip install -q torch torchvision   # CUDA build on Linux, MPS-enabled build on macOS
fi
"$PY" -m pip install -q -r v2/requirements.txt

"$PY" - <<'EOF'
import torch, transformers
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"torch {torch.__version__}, transformers {transformers.__version__}, device: {dev}")
EOF

if [ "$TESTS" = 1 ]; then
  "$PY" -m pytest tests/v2 -q
  "$PY" -m v2.smoke
fi

cat <<EOF

Setup done.
  1. source $VENV/bin/activate        (unless you used --system)
  2. python -m v2.doctor --mimic-cxr-jpg /path/to/mimic-cxr-jpg/2.1.0 --cohort-dir output/cohorts
  3. Optional, for MedSigLIP (M2): accept the terms at https://huggingface.co/google/medsiglip-448, then: hf auth login
  4. bash v2/scripts/run.sh m0        (see v2/README.md)
EOF
