# Session Handoff — V2 Teacher Retrain with Chain Data (Depth ≤ 4)

**Date**: 2026-08-27 · **Branch**: v2 · **Session end**: training stopped, machine being logged off

---

## What this session did

1. **Settled the v2a target** as single-board **N16R8** (per user decision, answered the
   XIAO-N8R8 question — see below). User drove: *"v2a but with longer chains data
   (up to 5 intents)", later "up to 4 chains and retrain the teacher with mid training
   evaluation".*
2. **Scaled chain-depth data**: `generator/deep_chains.py` depth distribution changed
   `[3,3,4,4,5,6]` → **`[3,3,4,4]`** (up to 4 intents). Built `data/generated/train_v5chains.jsonl`
   and `data/finetune/train_v5chains_nl.jsonl` — **5,769 examples**:
   - answer-count: 1 → 2,069 · 2 → 1,100 · 3 → 1,300 · 4 → 1,300
   - kinds: atomic 1880, compound 700, deep-chain 3000, unavailable 42, scrambled 147
3. **Teacher retrain infra hardened against a Metal bug** (see "Crash" below):
   - `finetune/train_full.py`: added `--snapshot-every N` → rolling mid-epoch snapshot for
     crash resume (metadata now carries `steps` too).
   - `/tmp/train_v8_supervisor.sh`: runs **one epoch per launch** via `--max-steps 700`,
     then evaluates the epoch snapshot **only while the GPU is free** (never concurrent
     with training), then resumes from it. Failed early version deleted its own resume
     snapshot (fixed).
   - Trained `full_v8teacher` attempt (init full_v1, lr 1e-4, seq 256, batch 8,
     val-split 0.03 → 173 holdout, 5600 steps/8 epochs). Epoch 1 completed
     (loss 0.0003, val 0.0163), eval 40/100 on chain set.

## Results so far

### Mid-training W4 chain evals (all on `/tmp/robot_v8_w4.cact`, 100-query held-out chain set)

All snapshots landed in a tight band — **39–43%** with no upward trend yet:

| Snapshot (steps) | Chain eval |
|---|---|
| ~250 (attempts 1–3) | 43, 42, 43, 43% |
| ~250 (attempt 4) | 39% |
| mixed | 42, 43% |
| **epoch 1 complete (700)** | **40%** |

Baseline for comparison: deployed t4 **40%**, pre-train W4 teacher **42%** on the same
chain set. **The chain data has NOT yet moved the teacher above its ~40% ceiling.**
This is the single most important open question — it was the stated reason the chain
workstream exists, and it is **not passing yet** (mid-training, only 1 full epoch seen).

> Caveat: the chain eval set (`data/eval/chain_eval.jsonl`) still contains depth-5/6
> queries from the old generator. Training now caps at depth 4. For consistent gating,
> regenerate a depth-1–4-only chain eval before concluding whether the gate fails.

### Architecture / feasibility answers from this session (for the docs)

- **XIAO ESP32-S3 Sense N8R8 (8 MB flash) cannot host v2a.** The d=384 student W4 blob is
  ~13 MB (projected) and even the t4-map compressed models approach the 8 MB ceiling once
  firmware (~1.6 MB) is subtracted. N16R8 (16 MB flash, model partition 14.68 MB) is the
  right target — matches the existing deployed build. (Full budget table in
  `docs/V2_PLAN.md` track v2a.)
- **Why a smaller student can help where the big teacher "failed":** the t4 failure and the
  chain failure are two different problems. v2a student ships at uniform W4 (no 2-bit
  cliff), and it copies chain behavior via distillation — but ONLY if the teacher first
  learns chains. That is the un-passed gate above.

## The Metal crash (root cause + why we capped out here)

- Symptom: JAX/Metal dies ~15–20 min into an attempt with
  `failed to legalize operation 'mhlo.dot_general'` in the attention matmul
  (`needle/model/architecture.py:270`).
- Root cause: JAX preallocates **12.7 GB** of unified memory up front. Any concurrent
  second JAX process (earlier: the eval watcher firing every 250 steps; also the val-eval
  compile inside the same run at larger val-split) pushes XLA into a recompile that the
  MPS legalizer then fails.
- Fixes that worked: (a) eval **never runs while training is alive** — moved into the
  supervisor epilog; (b) `--val-split 0.03` (173 instead of 576 holdout) to shrink the
  in-run val-eval footprint; (c) one-epoch-per-launch + `--snapshot-every 250` resume.
  With this setup epoch 1 completed cleanly.
- Note: `XLA_PYTHON_CLIENT_MEM_FRACTION` was **ignored** by the Metal plugin —
  preallocation stayed 12.7 GB regardless.

## Current state on disk (what to pick up next session)

- `checkpoints/full_v8teacher.pkl` — **not yet written** (training did not finish any
  full 8-epoch run; epoch 1 eval exists but the final checkpoint was never saved because
  the run was stopped mid-epoch).
- `/tmp/snap_v8.pkl` — epoch 1, 700 steps, 180 MB (usable as `--init` if you want one
  more epoch) — **/tmp is volatile**, copy it out if you want it.
- `/tmp/robot_v8_w4.cact` — last exported epoch-1 W4 weights (23.17 MB).
- Logs: `/tmp/train_v8.log`, `/tmp/train_v8_supervisor.log`, `/tmp/eval_v8_*.txt`,
  `/tmp/eval_v8_results.txt` (contains `EVAL epoch=1 steps=700 :: 40/100`).
- Supervisor script: `/tmp/train_v8_supervisor.sh` **(/tmp — re-save if needed)**.
- Git commits this session: `17fd01c` (chain data ≤4 + retrain launch), plus the
  `train_full.py --snapshot-every` change and `deep_chains.py` depth edits (`[3,3,4,4]`)
  which should be committed.

---

## Next steps (in order)

1. **Regenerate a depth-1–4-only chain eval set** (seed 777 or new) so the gate matches
   the training distribution; keep it frozen.
2. **Let `full_v8teacher` run to completion** with the fixed supervisor (one epoch/launch,
   epilog eval, resume). Watch the per-epoch eval line — **if it stays ≤ ~45% through
   epochs 3–4, that is a failed gate**: more of the same data will not fix chains and we
   must reconsider (bigger data scale 5–10K, different data mix, or an architecture-side
   pivot — do NOT distill a teacher that fails the chain gate, or the student inherits it).
3. **If gate passes**: proceed to v2a surgery (d=384 skeleton) + recovery distillation on
   the same chain-rich data, then uniform-W4 export, then the standard validation gates
   (check_engine, ref_forward, nd_gtest, eval ≥90% held-out + extended chain eval), then
   deploy to the N16R8 DevKitC at 0x1B0000 and write the v2 performance report.
4. **If gate fails**: stop distillation attempts; re-derive data scaling (this is the
   cheapest knob; plan estimated 2–5K chains — we are at 3K and flat, so next move is
   either 5–10K with more distractor/coref variety, or accept a depth-3 ceiling and target
   the v2b cluster for depth-4+).

## Open items / notes for next session

- `train_v8_supervisor.sh` correctly resumes and evals per epoch, but verify it tolerates
  a crash *inside* a 700-step epoch (it should: snapshot every 250 steps, but `--init`
  only accepts a full checkpoint — mid-epoch crash resumes from the last *epoch* snapshot,
  losing up to 700 steps; acceptable).
- The eval currently points at `data/eval/chain_eval.jsonl` — switch to the depth-1–4 set
  from step 1.
- If Metal crashes persist even with the epilog approach, consider `jax.config.update`
  preallocation via `XLA_PYTHON_CLIENT_PREALLOCATE=false` at the *top of the script*
  (env var was not honored; config API may be) — or run training on CPU with a much
  longer time budget and a smaller `--epochs`.
- Tuya of background: the **deployed N16R8 still runs v1 t4 + repair** (unchanged, 90%
  held-out). Do not overwrite the board until a candidate beats it on both the 200-query
  harness AND the (new) depth-1–4 chain eval.