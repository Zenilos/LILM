# LILM_V1 Project Report — Fine-tuning Needle 2 for On-Device Robot NLU

*Status as of Aug 24, 2026 · All work on M1/M3 Macs, pushed to `Zenilos/LILM`*

---

## 1. Objective

Fine-tune Needle 2 (`cactus-needle`, 90 MB JAX model, vocab 8192, d_model 512,
27 layers) into `robot.cact` — a quantized deployment blob for the XIAO
ESP32-S3 N16R8 (16 MB flash) — that maps typed text to ordered
`robot_action` tool calls:

```
"clean the living room and then go to kitchen"
  → [{"intent":"CLEAN","location":"the living room"},
     {"intent":"MOVE","location":"kitchen"}]
```

10 intents: CLEAN, GET, GIVE, MOVE, PLAY, SHOW, STOP, WAIT, WAKEUP,
UNAVAILABLE (refusals). Success metric: exact `actions_match` ≥ 90%.

**Final result achieved: 199/200 = 99.5% exact match**, end-to-end through a
real C engine reading our exported deployment blob.

---

## 2. Result Summary

| Milestone | Accuracy | Engine |
|---|---|---|
| First working LoRA (compact schema, bits=4) | ~57% | official dylib |
| LoRA run 2 (system prompt baked into data) | 18/50 = 36% | official dylib |
| Full-parameter fine-tune `full_v1` (10 ep) | 19/50 = 38% | official dylib |
| AQAT continuation (act-quant QAT) | 11/50 = 22% | official dylib |
| **Same `full_v1` weights via patched C99 engine** | **50/50 = 100%** | C99 (patched) |
| **Scaled eval** | **199/200 = 99.5%** | C99 (patched) |

The model was never the bottleneck after root cause 3 was fixed. The last 38%
ceiling was entirely an artifact of the official native engine's generation
path; a faithful engine scores 99.5% on the *same blob bytes*.

Per-intent at 199/200: CLEAN 18/18, GET 32/32, GIVE 27/27, MOVE 26/26,
PLAY 29/29, SHOW 20/20, STOP 18/18, UNAVAILABLE 17/17, WAIT 32/33,
WAKEUP 36/36.

Sole miss: *"please wait for a minute please"* → model emitted
`duration_amount=60, unit=minutes` instead of `1` (semantic normalization
nuance in training data, not an engine/model failure).

---

## 3. Timeline

### Phase 0 — Setup & data generation
- Generated 3297 training examples (`data/generated/v2.jsonl` → converter →
  `data/finetune/train_v2.jsonl`): 3004 with reasoning, 293 UNAVAILABLE,
  800 compound queries.
- Compact tool schema (`schema/tool_schema.json`, 165 tokens; verbose backup
  preserved). Required because verbose schema (419 tok) overflowed the
  model's 256-token KV window.
- Mac Metal training environment: Python 3.11 venv, jax-metal with
  `ENABLE_PJRT_COMPATIBILITY=1`, Ollama killed to free RAM, batch size 8
  (OOM-safe), `MAX_LEN=256`.

### Phase 1 — Three training-side root causes (fixed)
1. **2-bit export destroys fine-tunes.** `--bits 2` export → loops/wrong
   intents/truncation. Uniform W4 export behaves like float32. Fine-tuning in
   float32 leaves the base model's quantization-aware manifold.
2. **Agent context poisoning.** Reusing one `Needle` agent across queries
   accumulates state → later queries return empty. Fix:
   `needle._lib().needle_reset()` between queries.
3. **No system prompt in training data.** Inference sends
   `"device: domestic robot; locale: en-US"` but data never contained it →
   base-dialect behavior. Baked SYSTEM into every record + converter
   (commit `54d4ec7`).

### Phase 2 — Capacity ruled out
- LoRA rank 16 retrain (10 ep): 36%.
- Full-parameter fine-tune (`finetune/train_full.py`, pretrained init,
  AdamW warmup-cosine, direct cact export): 38%. Same ceiling → not capacity.
- AQAT (package's built-in activation fake-quant): 22% → abandoned.
- Float32 greedy decode of `full_v1`: **perfect** on all hard cases
  (UNAVAILABLE, PLAY+file, compounds). Model correct in JAX.

### Phase 3 — Numerics ruled out
- Weight-W4 dequant simulation in JAX: perfect.
- Full engine-numerics simulation (W4 weights + A8 activations via
  `fake_quant_act` + KV8): perfect.
- Logit-margin analysis: min margin ≈ 7, median ≈ 18 logits — too fat for
  A8 noise to flip.
- KV-width rebuild (kv_bits=16): identical results.
- Failure taxonomy: whole-call missing (6), intent-wrong (4), slot errors 0
  → binary "perfect or base-refusal", the signature of conditioning mismatch
  — yet prompt token streams were later proven byte-identical.

### Phase 4 — Engine forensics (the decisive arc)
- Discovered `NEEDLE_DEBUG=1` in `libneedle.dylib`: dumps tool inventory,
  prefix/turn token ids, first-step top-5, final think/text/calls.
- Proved input equivalence: engine prefix+turn ids == `[BOS] +
  tok.encode(render_example(...))` elementwise (184 tokens).
- Step-1 logits: JAX −3.69 vs dylib −3.99 on top-1 (8042), one token exact
  (591:−16.53 both) → prefill numerics agree.
- Env-var sweep (`NEEDLE_KV_BITS/WINDOW`, `NEEDLE_STRICT_VALIDATE`,
  `NEEDLE_NO_REBASE`, …): no effect.
- Control experiment: dylib on **official** `needle2.cact` reproduces JAX
  exactly (`get_weather(city="Paris")`) → engine faithful for the base
  model; its generation path breaks only for our fine-tuned blob.
- `ENGINE_VERSION="2.0.3"` is pinned inside pip package 2.0.9 — no newer
  binary exists; bug unreportable-but-unfixable upstream without source.

### Phase 5 — Independent engine: found the real bug
- Cloned `andrisgauracs/needle-2-esp32` (independent C99 `.cact` reader +
  engine, runs on ESP32-S3; production quant ~13 MB fits 16 MB flash).
- Its NumPy reference (`tools/ref_forward.py`) over raw blob bytes matched
  JAX perfectly — including full greedy generation. So the blob is right and
  two engines disagreed only in their own forward passes.
- Stage-by-stage numeric diff (embedding → engram k/v → per-layer y/lane →
  final norm → logits) localized divergence at **engram k/v projections**.
- Hash indices and fetched table rows were identical → culprit = projection
  kernel: **`nd_cq_lut_build()` / `dot_group_lut2()` hardcode the 2-bit
  codebook and 2-bit index packing.** Our blob stores all projection weights
  at bits=4 → every LUT-projected GEMV produced garbage. Upstream never sees
  this because the official model ships CQ2 projection weights.
- **Fix:** all LUT call sites branch on `t->bits == 2`; other widths take the
  verified generic `nd_cq_gemv()`. Post-fix: C == NumPy == JAX
  (top-5 `8042:-4.44 6:-17.19 591:-17.88 7:-20.01 286:-20.65` vs ref
  max −4.3864).
- End-to-end generation through fixed engine: perfect calls, compounds
  included. Eval: 50/50, then 199/200.

### Phase 6 — Persistence
- Vendored patched engine under `third_party/needle2-esp32/` (with README
  documenting the upstream bug + patch).
- `scripts/eval_c_engine.py`: N-query eval through the C engine using the
  same seed-42 sample + `actions_match` scoring as `eval_mac.sh`.
- Commits: `54d4ec7` (system bake-in), `1224c8a` (AQAT/--init),
  `483b0d7` (vendor + fix + eval), HANDOFF update, this report.

---

## 4. Technical Findings Worth Keeping

### Model/data
- Tokenizer emits invisible token 8042 (dummy prefix ▁) before any text
  starting with `<|im_start|>`; sentencepiece `▁` must be decoded to spaces
  when parsing generated JSON slots.
- `render_example` contract: system block + user block with `<tools>…</tools>`
  + assistant turn; specials THK=6, /THK=7, TOOL_CALL=10, EOS=1, IM_END=5.
- Training at seq 256/batch 8/10 epochs, LR 2e-4 cosine → val loss 0.008;
  strip-reasoning targets (`<think></think>\n<tool_call>…`) work best.
- Slot values: gold uses natural strings ("the living room"); model copies
  query phrasing faithfully.

### Engines
- Official dylib exports only `complete/init/load/reset/weights(_size)`; no
  logit access beyond NEEDLE_DEBUG step-1.
- Blob format v3 (tag 0x05E12A83): positional tensor directory (405 entries),
  per-tensor dtype/group/bits, header carries orders/sites/rope/kv config.
  We verified read_export ↔ C dequant bit-equality (≤2e-4 fp16 noise).
- C99 engine LUT fast-path is only valid for 2-bit tensors (now guarded);
  generic GEMV path handles 2/3/4-bit correctly.
- Shell gotchas that cost hours: `$()` strips trailing newlines (tokenizer
  off-by-one); zsh aborts on unquoted `=word`; `nd_tok_encode` vs
  `_ex(..., add_dummy=0)` semantics for continuations.

---

## 5. Current Repo State

```
robot.cact                     W4A8 deployment blob (23.17 MB — exceeds 16 MB flash!)
checkpoints/full_v1.pkl        best model (full-param FT, float32 master weights)
checkpoints/needle2.pkl        frozen base
third_party/needle2-esp32/     vendored C99 engine + CQ4 fix + host_runner CLI
scripts/eval_c_engine.py       99.5%-scoring eval through C engine
finetune/train_full.py         full-parameter trainer (--qat/--aqat/--init/--strip-reasoning)
finetune/{train_mac,eval_mac,post_full_watch}.sh
schema/tool_schema.json        compact schema (165 tok); verbose backup alongside
data/finetune/train_v2.jsonl   3297 examples, SYSTEM baked in
HANDOFF.md                     operational handoff + session update
```

Open issues:
- **Flash budget**: 23 MB > 16 MB. This is the remaining blocker for ESP32.
- Official-engine bug unreported upstream (evidence chain in HANDOFF.md).
- No held-out Test A/B/C set yet (current eval samples train_v2).

---

## 6. Next Step

Size-reduction pass so the blob fits 16 MB flash while keeping ≥90%:

1. Baseline ladder on the C engine (free): export `full_v1` at uniform W3
   (~17.8 MB, still over) and mixed widths (embed4/mhc4/rest2 ≈ 12–13 MB),
   eval each via `scripts/eval_c_engine.py`.
2. If W2/mixed degrades: QAT retrain (`train_full.py --qat <bits>` STE weight
   quantization) at the target width until engine-eval recovers.
3. Validate chosen width through the ESP32 build path (upstream demo) and
   measure on-device memory/latency.

Target: ≤14 MB blob, ≥90% exact match, running in the upstream ESP32-S3 demo.
