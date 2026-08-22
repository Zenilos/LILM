#!/bin/zsh
# Fine-tune Needle 2 on the robot DSL dataset.
# Verified working config on a 16GB M3 Pro (macOS):
#   - backend: METAL (uppercase). Lowercase 'metal' errors out.
#   - batch-size 4 is REQUIRED: 8+ OOM-kills (SIGKILL) during XLA compile.
#   - max-len 192: our commands are short; 1024 wastes memory.
#   - Stop Ollama first if running: it holds ~5-6 GB and contributes to the kill.
#   - CPU fallback works too (JAX_PLATFORMS=cpu) but compiles/steps much slower.
#
# usage: ./finetune/run_finetune.sh [dataset.jsonl] [out.pkl]
set -e
source ~/p3.11/bin/activate

DATA=${1:-data/finetune/train_v2.jsonl}
OUT=${2:-checkpoints/needle_lora_v2.pkl}
EPOCHS=${EPOCHS:-10}

brew services stop ollama 2>/dev/null || true

JAX_PLATFORMS=METAL needle finetune "$DATA" \
  --epochs "$EPOCHS" \
  --val-split 0.1 \
  --max-len 192 \
  --batch-size 4 \
  --out "$OUT"

# then build the deployable model:
#   needle build checkpoints/needle2.pkl --lora "$OUT" --bits 2 --out robot.cact
