# V2 Progress Report — Deep-Chain Training Experiment

**Date**: 2026-08-27 · **Branch**: v2 · **Training run**: full_v7deep

---

## What we tried

Extended the generator to produce depth 3–6 action chains with coreference
and distractors, then fine-tuned full_v1 on the enriched dataset (3,890
examples: 2,206 atomic + 800 2-clause + 600 deep-chain + 300 off-topic) for
5 epochs. Goal: close the chain-length generalization gap discovered in the
W4-vs-t4 comparison.

## Results

### Held-out eval (200 queries, zero train overlap)

| Model | Held-out accuracy | Notes |
|---|---|---|
| **Deployed t4** (full_v1 weights) | **90.0%** | Current production artifact |
| **full_v7deep t4** (same blob size) | **3.0%** | Catastrophic — model generates `</think>` loops, CONF 0.0000 |
| **full_v7deep W4** (23 MB, no quant) | **42.0%** | Evaluated on chain-specific subset |

### Chain-specific eval (100 deep-chain + 50 2-clause + 30 atomic)

| Model | Chain eval | Per-intent (GIVE strongest, WAIT weakest) |
|---|---|---|
| **Deployed t4** | **40/100 = 40%** | GIVE 13/46, SHOW 12/53, MOVE 10/37 |
| **full_v7deep W4** | **42/100 = 42%** | Nearly identical — 600 deep-chain examples insufficient |

### Training trajectory

| Epoch | fp32 val loss | t4 quantized accuracy | Diagnosis |
|---|---|---|---|
| 1 | 0.0137 | 10% | PTQ chaos: weights left the lucky basin |
| 2 | 0.0105 | — | (watcher behind) |
| 3 | 0.0082 | — | |
| 4 | 0.0081 | — | |
| 5 | 0.0080 | **3%** | Degenerate: `</think>` loops, no tool_call |

### On-board deployment (ESP32-S3 DevKitC-1)

| Metric | Deployed t4 | full_v7deep t4 |
|---|---|---|
| "clean the living room" | ✓ ACT CLEAN | TIMEOUT (no output) |
| "wait 30 minutes" | ✓ ACT WAIT (repair fires) | TIMEOUT |
| "stop" | ✓ ACT STOP | `</think>` loop, CONF 0.0000 |
| "play music.mp3" | ✓ ACT PLAY | `</think>` loop, CONF 0.0000 |

**full_v7deep t4 is completely unusable on hardware.** Restored original
t4 blob after confirming.

## Key findings

1. **PTQ chaos is worse than documented.** Previous runs (full_v6_aug) bounced
   8→24→14→0 across epochs. full_v7deep went straight to 3% and degenerated
   into EOS-loop behavior. The t4-compatible weight basin is not just lucky —
   it's *fragile to any training signal*, including deep-chain data that
   should be compatible with the teacher's existing capabilities.

2. **600 deep-chain examples are insufficient** to improve the teacher even at
   full W4 quality (42% vs 40% on chains). The chain gap is a
   generalization bottleneck that requires either (a) significantly more
   diverse chain data (thousands, not hundreds), or (b) architectural changes
   (v2a student trained from scratch on chains, or v2b cluster using full
   W4 quality).

3. **The t4 artifact must be frozen.** Every attempt to retrain full_v1
   and re-export at t4 makes things worse. The path forward is NOT
   retraining the same weights — it's building new weights (v2a student)
   or removing the quantization cliff (v2b cluster).

## Recommendation

**Do NOT pursue further retraining of full_v1 for t4.** The artifact at
`robot_t4.cact` (90% held-out) is the best this architecture can produce
at this bit-width. V2 must go through one of the two planned tracks:

- **v2a (distilled d=384 student)**: train from scratch on deep chains,
  export at uniform W4 (no mixed-width cliff), target ≥90%
- **v2b (two-board cluster)**: host full W4 blob across 2 ESP32s,
  quality = 99.5% by construction, no retraining needed for chains
  (but chain data work still needed for the teacher to handle them)

Step-0 results (both feasible) are documented in V2_PLAN.md.
The chain-depth data workstream should be scaled significantly (2K–5K
deep-chain examples) before either track's distillation/teacher step.

---

*Scores measured with `scripts/eval_c_engine.py --repair --eval-data`.
Chain eval set: `data/eval/chain_eval.jsonl` (seed 777, zero train overlap).*
