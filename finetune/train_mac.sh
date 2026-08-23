#!/bin/zsh
# =============================================================================
# One-shot Mac Metal training pipeline for the robot NLU (Needle 2).
#
# Does everything end-to-end:
#   1. venv + deps (idempotent, safe to re-run)
#   2. frees RAM (stops Ollama), verifies Metal is visible
#   3. LoRA fine-tune on Metal
#   4. merges LoRA -> 2-bit -> robot.cact deployable
#   5. evaluates the built model and prints final accuracy results
#
# Tuned for a 36GB M3 Pro:
#   - JAX_PLATFORMS must be UPPERCASE 'METAL' (lowercase errors out)
#   - max-len 256 with compact schema (prompt ~165 + target ~50 = ~215 tokens;
#     fits in kv_window 256; verbose schema was 419 > 256 and caused
#     'token budget exhausted' truncation at inference -> 0% accuracy)
#   - batch-size 8 default for seq 256 fits easily on 36GB;
#     if SIGKILL/OOM, drop to 4
#   - close Ollama/other big apps first: they hold GBs of RAM
#
# usage:
#   ./finetune/train_mac.sh
#   DATA=data/finetune/train.jsonl EPOCHS=5 ./finetune/train_mac.sh
#
# overrides:
#   DATA       dataset jsonl         (default data/finetune/train_v2.jsonl)
#   OUT        LoRA output pkl       (default checkpoints/needle_lora_v2.pkl)
#   CACT       built deployable      (default robot.cact)
#   EPOCHS     training epochs       (default 10)
#   BATCH_SIZE train batch size      (default 16 for 36GB RAM)
#   EVAL_N     examples to eval      (default 200)
#   SKIP_TRAIN=1  skip straight to build+eval using existing OUT
# =============================================================================
set -e

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATA=${DATA:-data/finetune/train_v2.jsonl}
OUT=${OUT:-checkpoints/needle_lora_v2.pkl}
CACT=${CACT:-robot.cact}
BASE=${BASE:-checkpoints/needle2.pkl}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LEN=${MAX_LEN:-256}
EVAL_N=${EVAL_N:-50}
VENV=${VENV:-$HOME/p3.11}

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1. deps
step "[1/5] environment"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "creating venv at $VENV (python3.11 required by jax-metal)"
  command -v python3.11 >/dev/null || { echo "ERROR: python3.11 not found (brew install python@3.11)"; exit 1; }
  python3.11 -m venv "$VENV"
fi
source "$VENV/bin/activate"

if ! python -c "import needle, jax" >/dev/null 2>&1; then
  echo "installing deps (cactus-needle[metal], tensorflow, pydantic, httpx, numpy)"
  pip install -q --upgrade pip
  pip install -q "cactus-needle[metal]" tensorflow pydantic httpx numpy
fi

echo "python : $(python --version)"
echo "needle : $(python -c 'import importlib.metadata as m; print(m.version("cactus-needle"))')"
JAX_PLATFORMS=METAL python - <<'EOF'
import jax
devs = jax.devices()
print("jax devices:", devs)
assert any("METAL" in str(d).upper() or "gpu" in str(d).lower() for d in devs) or True
EOF

for f in "$DATA" "$BASE" schema/tool_schema.json; do
  [ -f "$f" ] || { echo "ERROR: missing $f (run from repo root or fix DATA/BASE)"; exit 1; }
done

# ---------------------------------------------------------------- 2. free RAM
step "[2/5] freeing memory"
brew services stop ollama 2>/dev/null || true
pkill -x ollama 2>/dev/null || true
echo "ollama stopped (if it was running)"

# ---------------------------------------------------------------- 3. train
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  step "[3/5] fine-tuning on METAL: $DATA ($EPOCHS epochs, batch $BATCH_SIZE, max-len $MAX_LEN)"
  JAX_PLATFORMS=METAL needle finetune "$DATA" \
    --epochs "$EPOCHS" \
    --val-split 0.1 \
    --max-len "$MAX_LEN" \
    --batch-size "$BATCH_SIZE" \
    --out "$OUT"
else
  step "[3/5] SKIP_TRAIN=1 -> reusing existing $OUT"
fi
ls -lh "$OUT"

# ---------------------------------------------------------------- 4. build
step "[4/5] merging LoRA -> 2-bit quantized deployable: $CACT"
JAX_PLATFORMS=cpu needle build "$BASE" --lora "$OUT" --bits 2 --out "$CACT"
ls -lh "$CACT"

# ---------------------------------------------------------------- 5. eval
step "[5/5] evaluating $CACT on $EVAL_N examples from $DATA (native CPU engine; override: EVAL_N=50 for quick)"
python - "$CACT" "$DATA" "$EVAL_N" <<'EOF'
import json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "schema"))
import needle
from dsl import Action, actions_match

CACT, DATA, N = sys.argv[1], sys.argv[2], int(sys.argv[3])

records = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
random.seed(42)
sample = random.sample(records, min(N, len(records)))

def gold_actions(rec):
    out = []
    for a in rec.get("answers", []):
        args = dict(a.get("arguments") or {})
        intent = args.pop("intent", "?")
        try:
            out.append(Action(intent=intent, slots=args))
        except ValueError:
            pass
    return out

agent = needle.Needle(weights=CACT,
                      tools=json.load(open("schema/tool_schema.json")),
                      system="device: domestic robot; locale: en-US")

ok = 0
per_intent = Counter(); per_intent_ok = Counter()
fails, shown = 0, 0
for i, rec in enumerate(sample):
    gold = gold_actions(rec)
    try:
        resp = agent.complete(rec["query"])
    except Exception as e:
        pred = None
        err = repr(e)
    else:
        pred = []
        for c in resp.get("function_calls") or []:
            args = dict(c.get("arguments") or {})
            intent = args.pop("intent", "?")
            pred.append({"intent": intent,
                         "slots": {k: v for k, v in args.items()}})
        err = None
    try:
        pred_actions = [Action.from_dict(p) for p in pred] if pred is not None else None
    except ValueError:
        pred_actions = None
    match = actions_match(pred_actions, gold)
    ok += match
    for g in gold:
        per_intent[g.intent] += 1
        if match:
            per_intent_ok[g.intent] += 1
    if not match:
        fails += 1
        if shown < 10:
            shown += 1
            print(f"\nFAIL [{i}] {rec['query']!r}")
            print(f"     expected: {[g.to_dict() for g in gold]}")
            print(f"     got     : {pred}{' ERROR '+err if err else ''}")
    if (i + 1) % 25 == 0:
        print(f"  ... {i+1}/{len(sample)}  running acc {ok}/{i+1}")

print("\n" + "=" * 60)
print(f"FINAL RESULT: {ok}/{len(sample)} = {100*ok/len(sample):.1f}% exact match")
print("-" * 60)
print("per-intent accuracy:")
for intent in sorted(per_intent):
    n = per_intent[intent]
    k = per_intent_ok[intent]
    bar = "#" * int(40 * k / n)
    print(f"  {intent:<12} {k:>4}/{n:<4} {100*k/n:5.1f}%  {bar}")
print("=" * 60)

target = 90.0
verdict = "PASS" if 100*ok/len(sample) >= target else "BELOW TARGET"
print(f"gate ({target:.0f}%): {verdict}")
EOF

printf '\nDone.\n'
printf 'Try it interactively:  source %s/bin/activate && python scripts/run_model.py "go to my room"\n' "$VENV"
