#!/bin/bash
# Minimal setup: assumes python3.11 + venv already exist.
# Clones the repo (optional) and installs only the Python deps.
# Usage:
#   ./scripts/setup_mac.sh            # deps into CURRENT activated venv
set -e

command -v git >/dev/null || { echo "git missing"; exit 1; }

if [ ! -f schema/dsl.py ]; then
  echo "==> cloning repo"
  git clone https://github.com/Zenilos/LILM.git && cd LILM
fi
cd "$(dirname "$0")/.." 2>/dev/null || true

echo "==> python: $(python3 --version 2>/dev/null || python --version)"
echo "==> venv : $VIRTUAL_ENV"

echo "==> installing python deps"
pip install -q --upgrade pip
pip install -q "cactus-needle[metal]" tensorflow pydantic httpx numpy

echo "==> verifying needle engine"
python - <<'EOF'
import jax, needle
print("jax devices:", jax.devices())
EOF

python schema/dsl.py && echo "OK"
DONE=$'
------------------------------------------------------------
Ready. Remember:
  - robot.cact goes in repo root after Colab training
  - run scripts/run_model.py "go to kitchen"
'
echo "$DONE"
