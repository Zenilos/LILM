# Final Plan v2 — Fine-tuned Needle 2 on ESP32-S3 (Pivot)

**Status:** supersedes v1 (distillation). Pivot decided after verifying `andrisgauracs/needle-2-esp32` runs Needle 2 on N16R8 (weights mmap'd from 16 MB flash, KV cache in 8 MB octal PSRAM).
**Objective:** typed text in → grammar-valid ordered action calls out, fully offline on-device.
**Compute:** everything runs on this Mac (M3 Pro 36 GB, `~/p3.11`, `cactus-needle[metal]`). Colab is no longer needed.

---

## 0. What we will have at the end

1. **Dataset factory** (LLM-agnostic): DSL schema → deterministic generator → validated `dataset.jsonl`. Reusable later for a Char-CNN if N8R8/speed ever demands Path B.
2. **A fine-tuned Needle 2** (`robot.cact`, ~13–14 MB, CQ2-bit): parses your commands into one flat tool call per atomic action, off-topic → `[]`.
3. **Verified compatibility**: tuned `.cact` proven to load and produce correct calls in the `needle-2-esp32` **host engine** before any hardware step.
4. **Flashed board**: model in its flash partition, robot tools registered in the firmware handler table, end-to-end demo over serial TUI.
5. **Acceptance**: ≥ 300 hand-written Test C commands, ≥ 90% exact action-sequence accuracy through the real on-board engine.

Known accepted costs: ~25–40 s/command (~1.87 tok/s), ~51 s boot prime per power-up, confidence head disabled by fine-tuning.

---

## 1. Why no distillation / no QAT

- **Distillation** = teacher → tiny different architecture. Not needed: we ship Needle itself.
- **QAT**: not needed. LoRA trains on the frozen base; `needle build` merges the adapter and re-quantizes post-training to the checkpoint's per-layer bit map (`--bits 2`). Cactus's PTQ pipeline does quantization for us.

---

## 2. DSL — single source of truth

Nine atomic intents; HANDOVER = GET then GIVE; WAKEUP is ONE NLU action expanded by firmware planner (move-to-person + wake sound).

| Intent | Required slots | Optional |
|---|---|---|
| `MOVE` | `location` | |
| `CLEAN` | | `location` |
| `PLAY` | `file` | (SD-card filename) |
| `SHOW` | `message` | `person` |
| `GET` | `object` | |
| `GIVE` | `object`, `recipient` | |
| `STOP` | | |
| `WAIT` | `duration` | |
| `WAKEUP` | `recipient` | |

Canonical prediction = ordered `list[{intent, slots}]`.

### On-device tool schema (the 256-token window trick)

The engine fits only ~2–3 raw tools, so we declare **ONE tool**:

```json
[{"name":"robot_action","description":"Execute one atomic robot action",
  "parameters":{"type":"object","properties":{
    "intent":{"type":"string","enum":["MOVE","CLEAN","PLAY","SHOW","GET","GIVE","STOP","WAIT","WAKEUP"]},
    "location":{"type":"string"},
    "object":{"type":"string"},
    "recipient":{"type":"string"},
    "file":{"type":"string"},
    "duration":{"type":"string"},
    "message":{"type":"string"}},
  "required":["intent"]}}]
```

Flat types only (the port rejects nesting); enum constrains decoding; unused slot fields are simply omitted by the grammar-constrained decode. A compound command = multiple `robot_action` calls in one response. Slot values are raw spans ("five minutes", "my room"); duration/location resolution happens in C++ handlers, never in the model.

Fine-tuning teaches the model this exact schema, so token-window pressure drops versus zero-shot multi-tool use.

---

## 3. Pipeline

```text
MAC ─ dataset factory
  schema/dsl.py            Action dataclass, intents, normalize(), match()
  generator/*              canonical templates → synonyms → light typos
                           → compounds (stored boundaries) → off-topic
  augment/validate.py      OpenRouter LLM (free models) sanity-checks
                           generated text against known labels (PROBED:
                           base Needle zero-shot is too weak to validate;
                           see teacher/probe.py results)
        │
        ▼ convert to finetune JSONL
  {"query": "...", "tools": [<schema above>],
   "answers": [{"name":"robot_action","arguments":{...}}, ...],
   "reasoning": "'my room' -> location; ..."}
  (off-topic examples: "answers": [])
        │
MAC ─ training
  needle finetune data.jsonl --epochs 10 --lora-rank 16   # JAX/Metal
  needle build checkpoints/needle2.pkl --lora ... --bits 2 --out robot.cact
        │
MAC ─ verification gate (before any hardware)
  build host engine from andrisgauracs/needle-2-esp32 (cmake),
  run Test A/B/C through it, compare to known labels
        │
BOARD
  reflash model partition with robot.cact (esptool, ~3 min)
  paste same JSON schema into tools/robot.json,
  implement handlers in main.c TOOL_TABLE (move/clean/play/show/get/
  give/stop/wait/wakeup → planner expands wakeup)
  rebuild firmware (model partition untouched)
```

---

## 4. Phases

### Phase 0 — Schema + seeds
- `schema/dsl.py`; unit-test `match()` (case, articles, `5`↔`five`).
- Hand-write 200–400 reviewed `(text, list[Action])` seed pairs: every intent, every slot, compounds, off-topic, your real phrasings.

**Exit:** you approve `data/seed_commands.jsonl`.

### Phase 1 — Deterministic generator
- Synonyms/templates per intent, realistic typo noise (typed input, so no ASR phonetic noise), compound joins with stored boundaries, off-topic class.
- Emit 3–5k labeled examples.

**Exit:** `data/generated/v1.jsonl`.

### Phase 2 — LLM validation loop (OpenRouter, free models)
- PROBE RESULT (teacher/probe.py): base Needle zero-shot fails slot filling in both single-tool and nine-tool modes (`go to my room` → MOVE with no location; `give John the cup` → wakeup). Base Needle cannot be the validator.
- Use free OpenRouter models to check each generated example: does the text express exactly the known actions? Mismatch → reject/review.
- Also confirmed by probe: multi-call responses work even with one declared tool — compounds are safe for the on-device design.

**Exit:** validated `data/generated/v1.jsonl`, rejection rate logged.

### Phase 3 — OpenRouter augmentation (free models)
- Paraphrase diversity beyond templates; same validation gate.
- Scale to ~10–20k validated examples; carve Test A/B; hand-write Test C ≥ 300.

**Exit:** `data/finetune/train.jsonl` + tests.

### Phase 4 — Fine-tune + build (Mac)
- `needle finetune` (watch val loss each epoch; see doc/finetuning.md for sizing).
- `needle build ... --bits 2 --out robot.cact`.
- Smoke-test `robot.cact` with `needle playground --weights robot.cact` on 50 commands.

**Exit:** `robot.cact` behaves on host Python.

### Phase 5 — Host-engine verification gate ⚠ critical
- Build the C99 host engine from `andrisgauracs/needle-2-esp32`.
- Load `robot.cact` with our compacted schema; run Test A/B/C through it.
- Check: loads without error (format/bit-map compat), correct calls, latency/token stats.

**Exit:** ≥ 90% exact-sequence on Test C via host engine. If incompatible → fallback: keep base weights + rely on descriptions, or revisit bit map.

### Phase 6 — Board integration
- Reflash model partition; wire schema + 9 handlers into firmware; planner expands `WAKEUP`.
- Measure: boot prime time, ms/token, full-command latency.

**Exit:** TUI demo: type `"go to my room and wait there for 5 minutes and then go to oven"` → correct ordered actions executed/logged.

### Phase 7 — Acceptance report
- Test C through the board serial interface; report exact-sequence accuracy, latency, boot time.

---

## 5. Repository layout

```text
LILM_V1/
├── PLAN-final.md
├── schema/{dsl.py,tool_schema.json}
├── generator/{canonical.py,synonyms.py,noise.py,compound.py}
├── teacher/{needle_client.py,validate.py}     # caching + confidence gate
├── augment/openrouter_augment.py
├── data/{seed_commands.jsonl,generated/,validated/,rejected.jsonl,tests/}
├── finetune/{to_needle_jsonl.py,run_finetune.sh,build_cact.sh}
├── eval/{host_engine_eval.py,test_c.txt,report.py}
└── firmware-notes.md           # handler wiring for needle-2-esp32 fork
```

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Tuned `.cact` won't load in the C99 port (bit map/format) | Phase 5 gate before any hardware work; toy 50-example finetune tested early |
| Fine-tuned model loses off-topic rejection | Keep `answers: []` examples ~10–15% of data; test explicitly |
| Enum+7 fields degrades zero-shot but fine after tuning | Descriptions kept short & load-bearing; validate wording in Phase 2 |
| Compound responses get truncated (256-token window covers prompt+output) | Schema ≈ small; monitor output length; cap compounds at 3 actions in data |
| Metal/JAX finetune issues on M3 | Fall back to CPU JAX or Colab GPU (same scripts) |
| Slow iteration on hardware (51 s prime per boot) | Do all accuracy work via host engine; board only for final acceptance |

## 7. What not to do
- Don't train QAT pipelines — `needle build` owns quantization.
- Don't declare 9 separate tools — window overflow; one tool + intent enum.
- Don't skip Phase 5 — a `.cact` that works in Python but not in the port is the #1 failure mode.
- Don't put entity resolution (room IDs, seconds) in the model.
