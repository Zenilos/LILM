# V2 Plan — Two Tracks to >90%: (a) Distilled Student, (b) Two-Board Cluster

Status: **v1 deployed on hardware** (see DEPLOYMENT.md) · v2 tracks planned
Owner: TBD
Estimate: v2a 2–4 days · v2b 3–5 days (+0.5 day shared: chain-depth data)

## V1 shipping position (context for everything below)

- Best fitting artifact: **t4 map @ 14.65 MB, 76% raw** (full_v1 weights; all
  non-WAIT intents 85–100%, WAIT 0/7 — every miss is WAIT→WAKEUP with correct
  duration slots and *no recipient*).
- **Deterministic output repair** closes most of that gap with zero ML risk:
  a post-parser rule on device — `WAKEUP without recipient + duration_amount
  ⇒ rewrite intent to WAIT` (such outputs are illegal per schema anyway).
  Counterfactual scoring: **76% → 86%** on the 50-query harness; **confirmed
  177/200 = 88.5% at 200 queries** with `--repair`
  (`/tmp/eval_t4_repair_200.txt`; WAIT 31/33, CLEAN 18/18). Remaining misses
  are slot-value noise on rare tokens and a few truncated compounds.
- Retraining does NOT reliably buy quantization quality. Evidence:
  plain FT continuation of full_v1 on augmented data moved fp32 val CE to
  0.0000 while its t4-quantized accuracy bounced 8% → 24% → 14% → 0%
  across five epochs (run `full_v6_aug`, checkpoints/full_v6_aug.pkl).
  PTQ at 2/3-bit is chaotic in the weights: full_v1 was a lucky
  draw, and tiny (<6% relative) weight drift re-rolls it.
- **Shipped**: t4 blob + repair rule running on the DevKitC-1
  (`docs/DEPLOYMENT.md`, commit `7779dd9`); smoke tests passed on board.

## Post-deployment findings (hardware + W4-vs-t4 comparison)

Live tests with hard queries, comparing the deployed t4 blob against the
full-quality W4 export (`robot.cact`, same teacher weights) on the Mac:

| Failure class | t4 deployed | W4 teacher | Diagnosis |
|---|---|---|---|
| Rare-token slot garbling ("my daughter" → "my mom", "momentumianian") | ✗ | ✓ correct | Quantization damage → both v2 tracks fix |
| 5–6 action chains: truncation, degenerate MOVE repeats, dropped hops | ✗ | ✗ **still fails** | Training distribution — corpus compounds max at ~2–3 calls |

Consequences for the plan below:

1. **Chain-depth data workstream (shared, both tracks):** scale the synthetic
   generator to compounds of depth 4–6 with distractor clauses and
   coreference ("she is at my desk"), then regenerate the fine-tune set and
   distillation set from it. Cheap (~0.5 day), independent of hardware track,
   and required for the product to accept natural multi-step commands. The
   W4 teacher must pass extended chain evals BEFORE distilling (v2a), or the
   student inherits the gap.
2. **v2b's value proposition widens:** it removes quantization slot-noise
   without retraining; combined with (1) it is also the fastest route to
   correct long chains, since the teacher only needs data, not architecture,
   changes.
3. On-board latency measured: ~1.28 s/token decode, ~40 s per short request;
   prefix priming ≈ 168 s once per boot (cached). Long chains multiply token
   counts — keep an eye on UX budgets when evaluating tracks.

## Motivation

V1 kept the full 45M-param teacher and compressed via mixed-width quantization.
The audit showed why that path has a hard ceiling:

- Model flash partition on the ESP32-S3 N16R8 is **14.68 MB** (firmware owns
  the first 2 MB; v1 ended up at blob offset `0x1B0000` after the partition
  repurpose — see DEPLOYMENT.md).
- The byte distribution is dominated by d²-scaling kernels
  (q/gate/out = 21.2M params) plus vocab-sized tables (embedding + 2 engram
  tables = 12.6M params).
- Squeezing under the partition forced 2-bit k/v + tables, which collapsed the
  WAIT/WAKEUP logit margin.
- QAT could not rescue it: training-time fake-quant does not match
  `write_export`/engine quantization, so every QAT run improved val CE while
  degrading deployed accuracy (99.5 → 60–82 → 42% trajectories).

Two independent escape routes exist. They are not mutually exclusive; pick by
product constraints after de-risking spikes.

---

## Track v2b — Two-board cluster ("split the model, keep the quality")

**Idea:** pipeline-split the FULL-quality W4 blob across two ESP32-S3 boards,
each with its own 16 MB flash. No distillation, no quantization cliff —
exact teacher accuracy (99.5%) preserved by construction.

### Precedent

Upstream ships exactly this topology (`needle2-esp32-distributed`): Needle
halves on multiple ESP32-S3s over ESP-NOW, activations passed at the layer
seam. Our vendored engine already walks layers sequentially over mmap'd
flash; cutting at a layer boundary is a natural extension.

### Budget math (2 boards)

| Resource | Per board | Notes |
|---|---|---|
| W4 blob half | ~11.6 MB of 16 MB flash | fits with margin |
| KV cache half | ~1.75 MB of 8 MB PSRAM | window 256, int8 |
| Seam traffic (decode) | ~2 KB/token | ms-level vs 534 ms/token budget |
| Seam traffic (prefill) | ~400 KB once per query | seconds over ESP-NOW; faster over WiFi |

### Engineering steps

1. Blob splitter: partition `robot.cact` tensors by layer index into two
   images with matching headers.
2. Engine surgery: split forward pass at boundary L; board A runs layers
   < L, serializes hidden state (fp32 or int8), board B continues; route
   engram/retrieval heads whole onto one side.
3. Transport: ESP-NOW first (upstream reference code), WiFi TCP fallback.
4. Sync/failure handling: sequence numbers, retry, watchdog reset of the
   pair.
5. Validation: reuse all four host gates (dequant/GEMV, ref forward,
   grammar tests, C-engine eval) with a two-process shim before hardware.

### Risks

- Radio reliability in the field (mitigation: retries + local fallback to
  single-board t4+repair image).
- Product complexity: two powered boards, pairing, provisioning.
- Upstream distributed repo latency numbers were measured on their workload;
  re-measure ours (expected ≈ +5–10 ms/token decode).

---

## Track v2a — Distilled narrower student ("needle-robot-s")

### Target architecture

| | Teacher (v1) | Student (v2) |
|---|---|---|
| d_model | 512 | **384** |
| attention heads | 8 × 64 | 6 × 64 |
| kv_heads | 4 | 3 |
| params | ~45M | ~25M |
| export | mixed 2/3-bit (cliff) | **uniform W4** (~13 MB, fits with margin) |
| speed (ESP32 @240 MHz) | 534 ms/token | ~190 ms/token (est.) |
| expected task acc | 99.5% @ W4-23MB (doesn't fit) / 76% fitting | ≥90% target |

### Step 0 — Feasibility gate (10 minutes, decides everything)

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

### Pipeline

#### 1. Surgery (teacher → student skeleton)

Warm-start, not from-scratch — preserves pretraining leverage:

- embedding + engram tables: truncate rows along the feature dim (512→384).
- attention kernels: slice whole heads (keep first 6 query heads, first 3 kv
  heads); keep head_dim=64 so RoPE/attention internals are untouched.
- gate/out projections: select top-384 hidden units by importance
  (magnitude × activation frequency on task data), not blind slicing.
- keep tokenizer/vocab (8192) identical so the deployment stack is unchanged.

Expected state after surgery: degraded but functional model (typically within
single-digit % of teacher on the task) at zero training cost.

#### 2. Recovery distillation (fp32)

- Data: scale the synthetic generator (`scripts/gen_wait_aug.py` pattern) to
  ~30–50K queries covering all 10 intents + boundary pairs; optionally add
  teacher-filtered paraphrases (self-training) for slot diversity
  (`message`, `file`, `location`).
- Loss: KL(teacher logits ‖ student logits) + CE(gold), λ≈1.
- Short run: a few thousand steps usually restores near-teacher behavior
  because the skeleton is already functional.

#### 3. Quantization

- Export uniform W4 → ~13 MB, fits with ~1.5 MB margin. No 2-bit anywhere,
  no per-tensor maps, no cliff.
- Optional QAT polish *only after* fixing the fake-quant mismatch bug
  (see "Prerequisites" below). At W4-uniform this should be gentle.

#### 4. Validation

- Same gates as v1, all reused as-is:
  - `tools/check_engine.py` dequant/GEMV vs NumPy (<3e-7)
  - `ref_forward.py` full forward vs NumPy
  - `nd_gtest` grammar tests
  - C-engine eval (`scripts/eval_c_engine.py`) on held-out sample; success bar
    ≥90% exact-match, then 200-query confirmation.
- On-device bring-up reuses the upstream demo path unchanged.

## Prerequisites / known bugs to fix first

(applies to both tracks)

1. **Fake-quant vs export mismatch** — STE training uses
   `cq_quantize_params()` per leaf while export goes through
   `_cq_unpack`/codebook path; three independent QAT runs degraded deployed
   accuracy while improving val CE. Root-cause and align before trusting any
   QAT stage (not needed if step 3 stays plain-W4 PTQ).
2. Per-epoch eval loop writes interleaved logs when sharing a file with
   training stdout — use separate files (already done in `/tmp/aug_eval_loop.sh`
   pattern).

## v2a de-risking ladder (abort thresholds)

1. Step 0 fails constraint check → pivot axis (documented above). Cost: hours.
2. After surgery alone (no training): if task accuracy < 70%, selection method
   is wrong — try SVD truncation or gradual width steps 512→448→384.
3. After recovery distillation at 5K samples: expect ≥90% of teacher. If <80%,
   data volume/diversity is the bottleneck — scale generator before touching
   hyperparameters.
4. Final: student must beat the v1 shipping position (t4 + repair ≈ 86%)
    by construction; ship bar ≥90%.

## Track comparison & decision rule

| | v2a distilled student | v2b two-board cluster |
|---|---|---|
| Quality ceiling | ~90–95% (distillation retention) | **99.5% exact** |
| ML risk | moderate (retention curve unknown) | none |
| Systems risk | low (single board unchanged) | moderate (radio, sync, 2 firmwares) |
| BOM / product | 1 board | 2 boards, pairing + power |
| Latency/token | ~190 ms (est.) | ~540 ms (unchanged) + hop overhead |
| Effort | 2–4 days, mostly ML | 3–5 days, mostly firmware |

**Decision rule:** run both Step-0 spikes (v2a constraint check; v2b
two-process seam shim). If the v2a surgery+distill prototype at 20 K samples
holds ≥93% of teacher on the task → ship v2a (cheaper product). Otherwise →
v2b, whose quality is guaranteed and whose risk is ordinary systems work.
Either track replaces the deployed t4+repair build only when it beats it on
the 200-query harness **and** the extended chain eval (depth 4–6 compounds).

**Sequencing:** do the shared chain-depth data workstream first (it changes
the fine-tune/distillation sets both tracks consume), then the two Step-0
spikes, then commit to a track.

## Non-goals

- No change to tokenizer, schema, prompt format, host tooling, or deployment
  partition layout.
- No new engine features beyond what each track strictly requires;
  the single-board vendored engine stays bit-compatible.
