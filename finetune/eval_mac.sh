#!/bin/zsh
# Standalone eval for robot.cact on Mac Metal — no training.
# Usage:
#   ./finetune/eval_mac.sh                    # 50 random examples from train_v2 on METAL
#   EVAL_N=200 ./finetune/eval_mac.sh         # 200 examples
#   CACT=/tmp/robot4.cact DATA=data/finetune/train_v2.jsonl EVAL_N=50 ./finetune/eval_mac.sh
set -e
cd "$(dirname "$0")/.."
CACT=${CACT:-robot.cact}
DATA=${DATA:-data/finetune/train_v2.jsonl}
EVAL_N=${EVAL_N:-50}
VENV=${VENV:-$HOME/p3.11}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
[ -f "$CACT" ] || { echo "missing $CACT"; exit 1; }
[ -f "$DATA" ] || { echo "missing $DATA"; exit 1; }
echo "==> evaluating $CACT on $EVAL_N examples from $DATA (METAL)"
JAX_PLATFORMS=METAL python - "$CACT" "$DATA" "$EVAL_N" <<'PY'
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
    out=[]
    for a in rec.get("answers", []):
        args=dict(a.get("arguments") or {}); intent=args.pop("intent","?")
        try: out.append(Action(intent=intent, slots=args))
        except ValueError: pass
    return out
agent = needle.Needle(weights=CACT, tools=json.load(open("schema/tool_schema.json")), system="device: domestic robot; locale: en-US")
ok=0; per=Counter(); per_ok=Counter(); shown=0
for i, rec in enumerate(sample):
    gold=gold_actions(rec)
    try:
        resp=agent.complete(rec["query"])
        pred=[{"intent": (args:=dict(c.get("arguments")or{})).pop("intent","?"), "slots": {k:v for k,v in args.items()}} for c in resp.get("function_calls") or []]
        err=None
    except Exception as e:
        pred=None; err=repr(e)
    try: pred_actions=[Action.from_dict(p) for p in pred] if pred is not None else None
    except ValueError: pred_actions=None
    m=actions_match(pred_actions, gold)
    ok+=m
    for g in gold:
        per[g.intent]+=1
        if m: per_ok[g.intent]+=1
    if not m and shown<10:
        shown+=1
        print(f"\nFAIL [{i}] {rec['query']!r}\n     expected: {[g.to_dict() for g in gold]}\n     got     : {pred}{' ERROR '+err if err else ''}")
    if (i+1)%25==0: print(f"  ... {i+1}/{len(sample)}  running acc {ok}/{i+1}")
print("\n"+"="*60)
print(f"FINAL RESULT: {ok}/{len(sample)} = {100*ok/len(sample):.1f}% exact match")
print("-"*60)
for intent in sorted(per):
    n=per[intent]; k=per_ok[intent]
    print(f"  {intent:<12} {k:>4}/{n:<4} {100*k/n:5.1f}%  {'#'*int(40*k/n)}")
print("="*60)
tgt=90.0; print(f"gate ({tgt:.0f}%): {'PASS' if 100*ok/len(sample)>=tgt else 'BELOW TARGET'}")
PY
