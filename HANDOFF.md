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

## RESOLVED (Aug 23 evening) — actual root causes
1. **Model/training was NEVER broken.** Float32 JAX greedy decode from `render_example` prompt with merged LoRA produces PERFECT output (`<think>...</think><tool_call>[{"name":"robot_action","arguments":{"intent":"MOVE","location":"kitchen"}}]</tool_call>`). Diagnostic script pattern: load params+LoRA, `merge_lora`, `SimpleAttentionNetwork.apply`, greedy argmax.
2. **Root cause A — 2-bit export destroys the fine-tune.** Engine at `--bits 2`: loops/wrong intents/"token budget exhausted"/[]. At `bits=4` (23.2 MB): matches float32 behavior. 3-bit (17.8 MB) and attn-only-4-bit mixed (19.4 MB): intermediate degradation — intents flip to UNAVAILABLE, slots survive. LoRA delta is not quantization-aware (base was QAT'd for its own map; fine-tune in float32 leaves that manifold).
3. **Root cause B — agent reuse poisons eval.** Calling `complete()` repeatedly on one `Needle` accumulates context -> later queries return [] (budget death). FIX (verified): `needle._lib().needle_reset()` after each query == fresh-agent quality. Both `train_mac.sh` and `eval_mac.sh` patched (commit b1c7f64).
4. **Engine is deterministic** (5 identical runs) — not sampling noise.
5. Current `checkpoints/needle_lora_v2.pkl` (16:41, compact, 3 epochs, batch 8): float32 truth = 5/7 perfect, `tidy up the living room`->MOVE (should be CLEAN), `do you love me`->MOVE (should be UNAVAILABLE) — undertrained, hence the fresh 10-epoch run.

## In flight
- `nohup env EPOCHS=10 BATCH_SIZE=4 MAX_LEN=256 EVAL_N=50 ./finetune/train_mac.sh > /tmp/train_run.log` on this Mac (M1 16GB, batch 4 for RAM safety), started 17:26, ~3h ETA. Builds BITS=4 cact and runs eval-with-reset automatically.
- Size problem OPEN: b4 = 23 MB > 16 MB ESP32 flash. Options to explore: QAT (quantization-aware finetune), ternary for MLP + 4-bit attn, prune layers, or ship b3 if more training widens margins enough (re-test ladder after 10-epoch).

## Verification commands (after training)
```bash
tail -f /tmp/train_run.log          # progress
python3 - <<'PY'
import json, needle
a=needle.Needle(weights='robot.cact', tools=json.load(open('schema/tool_schema.json')), system="device: domestic robot; locale: en-US")
for q in ["go to kitchen","play audio lullaby.mp3","tidy up the living room","vacuum the balcony","do you love me","please hold on for 30 minutes","fetch teh glasses right now"]:
    r=a.complete(q); print(q, '->', r.get('function_calls'), r.get('error')); needle._lib().needle_reset()
PY
./finetune/eval_mac.sh              # 50 examples, reset between queries
```


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

---

# SESSION UPDATE (Aug 24, 2026): MYSTERY SOLVED — 99.5% THROUGH C ENGINE

## TL;DR
The fine-tuned model was perfect all along. The official native engine
(`libneedle.dylib` 2.0.3) breaks during *generation* on our W4 blob even
though its prefill matches JAX exactly. We vendored the independent
`andrisgauracs/needle-2-esp32` C99 engine, found and fixed a real bug in it
(LUT kernel hardcoded to the 2-bit codebook), and now score:

    199/200 = 99.5% exact match (scripts/eval_c_engine.py)
    Only miss: "wait for a minute" -> duration_amount=60/unit=minutes (model nuance)

## Evidence chain (all verified this session)
1. JAX float32 greedy on full_v1: perfect outputs incl UNAVAILABLE/compounds.
2. JAX with engine numerics simulated (W4 weights + A8 activations + KV8): still perfect.
3. Blob forensics: `ref_forward.py` (NumPy over raw blob bytes) == JAX == dylib step-1 logits; weight tensors bit-equal between read_export and C dequant (<=2e-4 fp16 noise); directory metadata identical to official needle2.cact except bits width.
4. Official dylib on OFFICIAL needle2.cact: reproduces JAX exactly (get_weather/Paris) -> engine is faithful for base model.
5. Official dylib on OUR robot.cact: step-1 top5 matches ([8042,-3.99],[6,-15.89],...), but generated text drifts into base dialect within a few tokens -> generation-path bug specific to finetuned blobs. Black box: only 4 exported symbols, NEEDLE_DEBUG gives only first-step top5.
6. Independent C99 engine had the SAME symptom but is source-available. Stage-by-stage diff vs NumPy ref pinned divergence to engram k/v projections -> nd_cq_lut_build()/dot_group_lut2() hardcode cb[0:4] (2-bit) while our tensors are bits=4 -> every LUT-projected gemv was garbage. Upstream never sees it because official model ships CQ2 projections.
7. Fix: branch on t->bits==2 at all LUT sites, else generic nd_cq_gemv(). Post-fix C == NumPy == JAX (8042:-4.44/-6/-17.19... vs ref max -4.3864).
8. Full greedy generation through fixed engine = perfect calls; eval 50/50 then 199/200.

## What this means practically
- Deploy via the vendored C99 engine (third_party/needle2-esp32/). It builds
  for ESP32-S3 upstream (their demo runs on-device; production quant ~13MB fits 16MB flash).
- robot.cact (W4 uniform) is 23MB — too big for 16MB flash as-is. Options:
  a) QAT retrain at lower width (--qat 3/2 + export bits matching), or
  b) mixed-width export (embed/mhc stay 4, rest 2/3) — engines support per-tensor bits.
- Official dylib bug: worth reporting to Cactus with evidence chain above.

## Key files added
- third_party/needle2-esp32/ (engine src patched, host_runner.c CLI, README with full patch notes)
- scripts/eval_c_engine.py — N-query eval through the C engine (same seed-42 sample + actions_match scoring as eval_mac.sh)

## Gotchas learned (zsh/shell/engine)
- $(cat file) strips trailing newlines -> tokenizer count off-by-one; read files in-process.
- nd_tok_encode adds dummy-prefix token; _ex(...,add_dummy=0) only for continuations after markers.
- Engine debug: NEEDLE_DEBUG=1 dumps prefix/turn ids + first-step top5 (only step!).
- ENGINE_VERSION is pinned "2.0.3" inside pip package 2.0.9 — no newer binary exists.
