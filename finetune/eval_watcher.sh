#!/bin/zsh
# Polls .eval_trigger written by finetune_patched.py after each epoch,
# builds temp cact and runs 5-shot eval, appends to eval_quick.log
# Usage: ./finetune/eval_watcher.sh &  (started automatically by train_mac.sh)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIGGER="$ROOT/checkpoints/.eval_trigger"
LOG="$ROOT/eval_quick.log"

# allow custom checkpoint dir via OUT
if [ -n "$OUT" ]; then
  TRIGGER="$(dirname "$OUT")/.eval_trigger"
fi

echo "eval_watcher polling $TRIGGER -> $LOG" > "$LOG"
while true; do
  if [ -f "$TRIGGER" ]; then
    # atomically grab and clear trigger
    TMP=$(mktemp)
    mv "$TRIGGER" "$TMP" 2>/dev/null || { sleep 1; continue; }
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      epoch=$(echo "$line" | awk '{print $1}')
      ckpt=$(echo "$line" | awk '{print $2}')
      base=$(echo "$line" | awk '{print $3}')
      data=$(echo "$line" | awk '{print $4}')
      total=$(echo "$line" | awk '{print $5}')
      [ -f "$ckpt" ] || continue
      echo "$(date '+%H:%M:%S') quick $epoch/$total  building..." | tee -a "$LOG"
      # build temp cact
      tmp_cact="${ckpt%.pkl}.cact"
      JAX_PLATFORMS=cpu python3 - <<PY 2>&1 | tee -a "$LOG"
import pickle, json, random, os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from needle.model.run import load_checkpoint
from needle.model.export import write_export
from needle.model.tokenizer import get_tokenizer
from needle.model.architecture import effective_kv_window
from needle.model.finetune import merge_lora
ckpt="$ckpt"
base="$base"
data="$data"
epoch=int("$epoch")
total=int("$total")
with open(ckpt,"rb") as f: pack=pickle.load(f)
params, cfg, _ = load_checkpoint(base, return_run=True)
lora={tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])} for k,v in pack["lora"].items()}
merged=merge_lora(params, lora, pack["scale"])
tmp_cact=ckpt.replace(".pkl",".cact")
write_export(merged, cfg, tmp_cact, bits=2, bits_map=None, tokenizer=get_tokenizer(cfg.vocab_size), kv_window=effective_kv_window(cfg))
import needle
tools=json.load(open("schema/tool_schema.json"))
agent=needle.Needle(weights=tmp_cact, tools=tools, system="device: domestic robot; locale: en-US")
recs=[json.loads(l) for l in open(data) if l.strip()]
random.seed(42+epoch)
sample=random.sample(recs, min(5, len(recs)))
ok=0
for r in sample:
    try:
        resp=agent.complete(r["query"])
        pred=[]
        for c in resp.get("function_calls") or []:
            a=dict(c.get("arguments") or {})
            pred.append({"intent": a.pop("intent","?"), "slots": {k:v for k,v in a.items()}})
        gold=[]
        for a in r.get("answers", []):
            ag=dict(a.get("arguments") or {})
            intent=ag.pop("intent","?")
            try:
                from schema.dsl import Action
                gold.append(Action(intent=intent, slots=ag))
            except: pass
        if gold:
            from schema.dsl import Action as A2, actions_match
            try:
                pa=[A2.from_dict(p) for p in pred]
                if actions_match(pa, gold): ok+=1
            except: pass
    except: pass
print(f"quick {epoch}/{total}  5-shot {ok}/5 = {100*ok/5:.0f}%")
PY
      echo "$(date '+%H:%M:%S') quick $epoch/$total  done" | tee -a "$LOG"
      rm -f "$ckpt" "$tmp_cact"
    done < "$TMP"
    rm -f "$TMP"
  fi
  sleep 2
  # exit when training done and no trigger
  if [ -f "$ROOT/checkpoints/needle_lora_v2.pkl" ] && [ ! -f "$TRIGGER" ]; then
    # keep watching for a bit then exit if no new triggers
    sleep 5
    [ -f "$TRIGGER" ] || break
  fi
done
