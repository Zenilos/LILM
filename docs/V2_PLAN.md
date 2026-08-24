# V2 Plan — Distilled Narrower Student ("needle-robot-s")

Status: planned (v1 ships first — see PROJECT_REPORT.md)
Owner: TBD
Estimate: 2–4 days once started

## Motivation

V1 kept the full 45M-param teacher and compressed via mixed-width quantization.
The audit showed why that path has a hard ceiling:

- Model flash partition on the ESP32-S3 N16R8 is **14.68 MB** (firmware owns the
  first 2 MB; blob flashed at `0x210000`).
- The byte distribution is dominated by d²-scaling kernels
  (q/gate/out = 21.2M params) plus vocab-sized tables (embedding + 2 engram
  tables = 12.6M params).
- Squeezing under the partition forced 2-bit k/v + tables, which collapsed the
  WAIT/WAKEUP logit margin: best fitting artifact today is **t4 @ 14.65 MB,
  76%** (all non-WAIT intents 85–100%, WAIT ~0/7).
- QAT could not rescue it: training-time fake-quant does not match
  `write_export`/engine quantization, so every QAT run improved val CE while
  degrading deployed accuracy (99.5 → 60–82 → 42% trajectories).

A purpose-built narrower student removes the tradeoff entirely.

## Target architecture

| | Teacher (v1) | Student (v2) |
|---|---|---|
| d_model | 512 | **384** |
| attention heads | 8 × 64 | 6 × 64 |
| kv_heads | 4 | 3 |
| params | ~45M | ~25M |
| export | mixed 2/3-bit (cliff) | **uniform W4** (~13 MB, fits with margin) |
| speed (ESP32 @240 MHz) | 534 ms/token | ~190 ms/token (est.) |
| expected task acc | 99.5% @ W4-23MB (doesn't fit) / 76% fitting | ≥90% target |

## Step 0 — Feasibility gate (10 minutes, decides everything)

The `.cact` header carries `hada_n` (Hadamard width used by the mhc machinery)
plus `mhc_lanes`, `engram_sub_dim`, etc. If `hada_n` must equal `d_model` AND be
a power of two, d=384 is illegal and the plan pivots:

- Check `engine/src/nd_model.c`, `nd_cact.c`, and
  `needle/model/architecture.py` for how `hada_n` is derived/constrained.
- If constrained to pow2: fall back to d=256 (re-run size math; likely still
  fits at W4 ≈ 7–8 MB but capability risk) or pivot the axis entirely
  (vocab/table shrinking instead of width).
- Also verify `TransformerConfig` → `write_export` → engine round-trip at the
  new width using random weights before any training investment.

## Pipeline

### 1. Surgery (teacher → student skeleton)

Warm-start, not from-scratch — preserves pretraining leverage:

- embedding + engram tables: truncate rows along the feature dim (512→384).
- attention kernels: slice whole heads (keep first 6 query heads, first 3 kv
  heads); keep head_dim=64 so RoPE/attention internals are untouched.
- gate/out projections: select top-384 hidden units by importance
  (magnitude × activation frequency on task data), not blind slicing.
- keep tokenizer/vocab (8192) identical so the deployment stack is unchanged.

Expected state after surgery: degraded but functional model (typically within
single-digit % of teacher on the task) at zero training cost.

### 2. Recovery distillation (fp32)

- Data: scale the synthetic generator (`scripts/gen_wait_aug.py` pattern) to
  ~30–50K queries covering all 10 intents + boundary pairs; optionally add
  teacher-filtered paraphrases (self-training) for slot diversity
  (`message`, `file`, `location`).
- Loss: KL(teacher logits ‖ student logits) + CE(gold), λ≈1.
- Short run: a few thousand steps usually restores near-teacher behavior
  because the skeleton is already functional.

### 3. Quantization

- Export uniform W4 → ~13 MB, fits with ~1.5 MB margin. No 2-bit anywhere,
  no per-tensor maps, no cliff.
- Optional QAT polish *only after* fixing the fake-quant mismatch bug
  (see "Prerequisites" below). At W4-uniform this should be gentle.

### 4. Validation

- Same gates as v1, all reused as-is:
  - `tools/check_engine.py` dequant/GEMV vs NumPy (<3e-7)
  - `ref_forward.py` full forward vs NumPy
  - `nd_gtest` grammar tests
  - C-engine eval (`scripts/eval_c_engine.py`) on held-out sample; success bar
    ≥90% exact-match, then 200-query confirmation.
- On-device bring-up reuses the upstream demo path unchanged.

## Prerequisites / known bugs to fix first

1. **Fake-quant vs export mismatch** — STE training uses
   `cq_quantize_params()` per leaf while export goes through
   `_cq_unpack`/codebook path; three independent QAT runs degraded deployed
   accuracy while improving val CE. Root-cause and align before trusting any
   QAT stage (not needed if step 3 stays plain-W4 PTQ).
2. Per-epoch eval loop writes interleaved logs when sharing a file with
   training stdout — use separate files (already done in `/tmp/aug_eval_loop.sh`
   pattern).

## De-risking ladder (abort thresholds)

1. Step 0 fails constraint check → pivot axis (documented above). Cost: hours.
2. After surgery alone (no training): if task accuracy < 70%, selection method
   is wrong — try SVD truncation or gradual width steps 512→448→384.
3. After recovery distillation at 5K samples: expect ≥90% of teacher. If <80%,
   data volume/diversity is the bottleneck — scale generator before touching
   hyperparameters.
4. Final: student must beat t4's 76% fitting-artifact baseline by construction;
   ship bar ≥90%.

## Non-goals

- No change to tokenizer, schema, prompt format, host tooling, or deployment
  partition layout.
- No new engine features; the vendored patched engine stays bit-compatible.
