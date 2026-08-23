# HANDOFF — LILM_V1 Mac Metal Training

## Goal
Fine-tune Needle 2 (90 MB, `cactus-needle` 2.0.9, `checkpoints/needle2.pkl`) to `robot.cact` (12 MB, W2A8, 405 tensors) for XIAO ESP32-S3 N16R8. Input: typed text -> `robot_action` tool calls (10 intents, `schema/dsl.py` + `schema/tool_schema.json`). Target ≥90% exact `actions_match` on Test C (not yet created; current eval uses `data/finetune/train_v2.jsonl` 200-sample).

## Repo State (commit 320fd95..2b16d1e, main)
- **Data**: `data/generated/v2.jsonl` (3297, `{"text","actions"}`) -> `finetune/to_needle_jsonl.py` -> `data/finetune/train_v2.jsonl` (3297, `{"query","tools","answers","reasoning"}`), `data/finetune/toy.jsonl` (60). No `data/tests/` yet. `train_v2` 3297 = 3004 with reasoning, 293 UNAVAILABLE, 800 compound.
- **Model**: No local model def; `needle` = JAX `SimpleAttentionNetwork` (d_model 512, 27 layers, vocab 8192, kv_window 256, max_seq 2048) + LoRA rank 16 alpha 32 (5 groups, scale 2.0) via `needle/model/finetune.py`.
- **Mac**: M3 Pro 36 GB, Python 3.11 venv `~/p3.11`, `jax 0.4.38`, `jax-metal 0.1.1`, `ENABLE_PJRT_COMPATIBILITY=1`, `JAX_PLATFORMS=METAL` (uppercase). Ollama must be stopped (holds 5-6 GB).
- **Checkpoints**: `checkpoints/needle2.pkl` (86 MB base), `checkpoints/needle_lora_v2.pkl` (7.6 MB, overwritten each run), `robot.cact` (12 MB, root, also `robot_toy.cact`).

## What Was Fixed
1. **Batch**: 4 -> 16 -> 8. 8 OOM-killed on 16 GB during XLA compile; 16 was tried on 36 GB but `loss 0.0000` was not cosmetic — see next.
2. **max-len 192 -> 512 -> 1024 -> 256**: `fit_max_len` buckets (128,256,512...). Prompt with verbose schema = 419 toks + target ~50 = 470. `cap 192` => mask 0.0 for all 3297 (verified: `load_jsonl` mask mean 0, 3297 zeros) => loss 0.0000 and 0% accuracy. `cap 512` gave mask 50, `cap 1024` bucketed to 1024 (2× compile, 2h stall at `(compiling...)`), settled on `256` after compact schema.
3. **KV window overflow**: `cfg.kv_window=256` (effective 256). Verbose schema 419 >256 => `needle_complete` returns `tool call truncated: token budget exhausted` for most queries (even `hi` at 176+16=192). Created `schema/tool_schema_compact.json` (165 toks, no descriptions, total 203 with system) and swapped `schema/tool_schema.json` to compact (backup `tool_schema_verbose.json`), regenerated `train_v2.jsonl` with compact tools (cap 256 now mask 50, zeros 0).
4. **Eval**: `finetune/train_mac.sh` does 5 steps (deps, free RAM, `needle finetune`, `needle build --bits 2`, eval). Eval is **native CPU** `libneedle.dylib` (ctypes, not JAX/METAL) — ~0.5s/prefill 1000 tok/s + decode 700 tok/s, 5 examples ~2s + build 8s. Default `EVAL_N 200->50`. Added `finetune/eval_mac.sh` standalone. Added file-signal quick eval: training writes `checkpoints/.tmp_lora_epochN.pkl` + `checkpoints/.eval_trigger` each epoch, `finetune/eval_watcher.sh` polls and appends 5-shot to `eval_quick.log` (bg, non-blocking). Patched `finetune/finetune_patched.py` (copied into `~/p3.11/.../needle/model/finetune.py` by `train_mac.sh`). Last commit `2b16d1e` = quick eval every epoch (was every 2).

## Current Failure (still 0% after all fixes)
- Training at `seq 256, batch 8, 10 epochs, compact` completes: `val` drops `0.1276 (ep1) -> 0.0462 -> 0.0241 (ep3) -> ~0.0118 (ep10)` — looks healthy. `adapter` saved, `robot.cact` built (12 MB).
- Inference still `0/5` (seed 42): all `got: []` or `UNAVAILABLE` or `truncated`. Raw `a.complete("go to kitchen")` with new compact cact gives `UNAVAILABLE` or `tool call truncated` even for `hi` (176+16). Manual rebuild of same LoRA with `write_export(..., kv 256/512/1024)` also gives `UNAVAILABLE` (not truncated for kv 256, but wrong intent).
- Toy (60 examples, verbose) also 0% and malformed JSON: `PLAY` with `location` instead of `file`, e.g. `{"intent":"PLAY","message":"","location":"beep.wav"}`.

## Hypotheses for Next AI
1. **Slot mapping bug**: model predicts `location` for `PLAY` — check `to_needle_jsonl.py` `example_to_answer` / `SLOT_KEYS` and whether compact schema dropped `description` needed for disambiguation.
2. **Reasoning field**: training target is `<think>reasoning</think><tool_call>answers</tool_call>`. Reasoning is 3004/3297. Maybe model learns to emit reasoning but inference expects no reasoning, or reasoning length pushes budget.
3. **Tokenizer / `render_example` mismatch**: training prompt is `IM_START user TOOLS_START tools_json TOOLS_END query IM_END IM_START assistant` — verify `needle` inference builds identical prompt (system handling).
4. **LoRA merge / quant**: `merge_lora` scale 2.0, bits 2 — try `bits 4` or no quant to rule out destruction. Manual `bits 2` vs `needle build --bits 2` gave different MD5s (b514 vs 8512) — investigate `bits_map`.
5. **Overfit test**: toy should overfit 60 examples to >90% if pipeline is correct — it doesn't, so pipeline is broken, not data size.

## How to Reproduce
```bash
cd ~/codes/esp32/LILM_V1  # or /Users/M2/codes/esp32/LILM_V1 on this Mac
git pull
./finetune/train_mac.sh   # defaults: DATA=train_v2.jsonl EPOCHS=10 BATCH_SIZE=8 MAX_LEN=256 EVAL_N=50, QUICK_EVAL bg
# or smoke:
EPOCHS=3 BATCH_SIZE=8 MAX_LEN=256 ./finetune/train_mac.sh
# eval only (old cact is stale after retrain):
./finetune/eval_mac.sh
EVAL_N=5 ./finetune/eval_mac.sh
# watcher log:
tail -f eval_quick.log
# raw probe:
python3 - <<'PY'
import json, needle
a=needle.Needle(weights='robot.cact', tools=json.load(open('schema/tool_schema.json')), system="device: domestic robot; locale: en-US")
print(a.complete("go to kitchen"))
print(a.complete("play audio lullaby.mp3"))
PY
```

## Files to Share
- `robot.cact` (root, 12 MB) + `checkpoints/needle_lora_v2.pkl` (7.6 MB) for testing here. After any retrain, these are overwritten.
- `data/finetune/train_v2.jsonl` is now compact (165 toks); to revert to verbose: `cp schema/tool_schema_verbose.json schema/tool_schema.json && python3 finetune/to_needle_jsonl.py data/generated/v2.jsonl data/finetune/train_v2.jsonl`.

## Open Issues
- Quick eval watcher only logs header (`polling...`) on this Mac — `checkpoints/.eval_trigger` is written (`quick N queued`) but watcher never processes (ps shows it, trigger file exists but not consumed). Training's `cp finetune_patched.py` to venv may race; verify `grep -c eval_trigger ~/p3.11/.../finetune.py`.
- Eval with `EVAL_N=5` fast (~10s), `EVAL_N=10` hangs >2 min on 6th sample (compound WAKEUP+GET) — sequential native engine, one slow sample blocks all.
- No Test A/B/C, no `eval/report.py` yet.

## Next Steps Suggested
1. Fix toy overfit first (simplest signal). Try removing `reasoning`, or training with `MAX_LEN=256` compact on toy for 20 epochs, LR default (`args.lr` — check `finetune.py` optax warmup cosine).
2. Try `needle build --bits 4` and eval to isolate quant.
3. Log actual `render_example` prompt for a failing query vs what `Needle` sends.
4. Consider regenerating `train_v2` without descriptions but keeping intents, or with even shorter schema.
