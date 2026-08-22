#!/bin/bash
# One-shot macOS (Apple Silicon) setup for the LILM project.
# Usage: git clone https://github.com/Zenilos/LILM.git && cd LILM && ./scripts/setup_mac.sh
set -e

echo "==> 1/7 Homebrew"
if ! command -v brew >/dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "    already installed"
fi

echo "==> 2/7 Python 3.11"
brew list --formula | grep -q "^python@3.11$" || brew install python@3.11
PY311="$(brew --prefix)/opt/python@3.11/bin/python3.11"

echo "==> 3/7 Virtual env at ~/p3.11"
if [ ! -d "$HOME/p3.11" ]; then
  "$PY311" -m venv "$HOME/p3.11"
else
  echo "    ~/p3.11 exists, reusing"
fi
source "$HOME/p3.11/bin/activate"

echo "==> 4/7 Python packages"
pip install -q --upgrade pip
pip install -q "cactus-needle[metal]" tensorflow pydantic httpx numpy

echo "==> 5/7 Ollama + local validator model"
if ! command -v ollama >/dev/null; then
  brew install ollama
fi
brew services start ollama || true
sleep 3
ollama list | grep -q "qwen2.5:7b-instruct" || ollama pull qwen2.5:7b-instruct

echo "==> 6/7 Verify Needle engine (downloads ~14 MB from HF once)"
python - <<'EOF'
import needle, jax
print("needle:", needle.__name__, "| jax devices:", jax.devices())
EOF

echo "==> 7/7 Smoke tests"
python schema/dsl.py
python generator/build_dataset.py /tmp/lilm_smoke.jsonl && rm /tmp/lilm_smoke.jsonl

cat <<'DONE'

============================================================
Setup complete. Activate the env in every new shell:
  source ~/p3.11/bin/activate

Quick commands:
  python scripts/run_model.py "go to my room"     # needs robot.cact in repo root
  python augment/run_validation.py data/generated/v2.jsonl "" 300   # local LLM validation
  ./finetune/run_finetune.sh                      # metal finetune (see notes)

NOTE: stop Ollama before Metal finetune: brew services stop ollama
DONE
