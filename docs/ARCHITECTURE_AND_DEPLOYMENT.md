# LILM_V1 — Architecture & Deployment Guide (agent handover)

*Audience: a fresh agent (human or LLM) taking over deployment and LM feature
work. Read this top to bottom once; everything else is linked from here.
Status as of 2026-08-25. Companion docs: PROJECT_REPORT.md (training story),
DEPLOYMENT.md (bring-up log), V2_PLAN.md (future work).*

---

## 1. What this project is

A 45M-param Needle-2 language model ("cactus-needle", JAX) fine-tuned to
convert English commands into ordered robot tool calls, quantized into a
`.cact` blob, and running **fully offline on an ESP32-S3**:

```
"clean the living room and then go to kitchen"
   → [{"intent":"CLEAN","location":"the living room"},
      {"intent":"MOVE","location":"kitchen"}]
```

10 intents: CLEAN, GET, GIVE, MOVE, PLAY, SHOW, STOP, WAIT, WAKEUP,
UNAVAILABLE. Metric: exact-match of the full call list (`schema/dsl.py:
actions_match`).

**Current deployed state**: t4 mixed-width blob + deterministic repair rule,
88.5% (177/200) host-side, verified live on an ESP32-S3 DevKitC-1 N16R8
(commit `7779dd9`). Full-quality W4 export (`robot.cact`, doesn't fit the
board) scores 99.5%.

## 2. Repository map

```
LILM_V1/
├── third_party/needle2-esp32/     VENDORED RUNTIME — what actually runs
│   ├── engine/                    Portable C99 engine (SAN forward pass,
│   │                              grammar-constrained sampling). Patched;
│   │                              stays bit-compatible with upstream.
│   ├── esp32/needle_demo/         THE FIRMWARE (see §6)
│   │   ├── partitions.csv         flash layout (model @ 0x1B0000)
│   │   ├── main/main.c            app: mmap blob → prime → generate → dispatch
│   │   └── components/needle/     CMake wrapper compiling engine/ as IDF component
│   ├── host/                      host_runner + test harnesses (CMake)
│   └── tools/robot.json           copy of canonical tool schema for firmware build
├── generator/                     synthetic data generator (canonical.py, noise.py,
│   │                              build_dataset.py)
├── augment/                       validation of generated data
├── finetune/train_full.py         JAX/Metal full-parameter trainer (+ gotchas §8)
├── scripts/
│   ├── export_snapshot.py         pkl checkpoint → .cact blob (quantize here!)
│   ├── eval_c_engine.py           THE eval harness: builds host_runner, scores
│   │                              exact-match; --repair applies the WAIT fix
│   └── gen_wait_aug.py            augmentation generator (WAIT/WAKEUP focus)
├── schema/dsl.py                  Action dataclass + slot rules + actions_match
│                                  (single source of truth for legality)
├── schema/tool_schema.json        canonical tool schema (grammar + training +
│                                  firmware all compile THIS file)
├── checkpoints/*.pkl              JAX param pytrees (full_v1 = shipped weights)
├── robot.cact                     full-quality W4 export, 23.17 MB, 99.5% — Mac-only
├── docs/                          this file, DEPLOYMENT.md, V2_PLAN.md, PROJECT_REPORT.md
└── data/, archive/, colab_bundle/, teacher/, notebooks/
```

Upstream references: `needle2-esp32` (engine+demo), `needle` python package
(architecture/training; lives inside colab_bundle if needed).

## 3. Model architecture (what the 45M params are)

Needle-2 is a Simple Attention Network — NOT a transformer stack:

- **d_model=512, 27 layers, vocab 8192**, ~534 ms/token class on ESP32 @240 MHz.
- Per layer: input splits into 4 "lanes"; each lane gets RMS-normed, RoPE'd
  attention over **8 query heads / 4 KV heads (head_dim 64)**, then a
  Hadamard-modulated MLP (`mhc`: lanes × d_model element-wise products passed
  through phi projections). No feedforward matrices in the transformer sense;
  byte cost is dominated by q/gate/out projections (21.2M params) plus
  vocab-sized tables.
- **Engram memory**: 2 key/value tables addressed by a hash of recent tokens
  (site-local history), giving the model cheap long-range context beyond the
  KV window (trained window = 256).
- **MTP/confidence heads**: auxiliary multi-token-prediction combine +
  probe-based confidence scalar (`nd_model_confidence`), used by the runtime
  to report CONF lines.
- Quantization: per-tensor-group codebook scheme ("cq") in
  `engine/include/nd_quant.h`; weights stay packed in flash and are dequantized
  row-at-a-time during GEMV — RAM never holds the model.

## 4. Task contract (the single source of truth chain)

```
schema/tool_schema.json ──compiles──▶ nd_grammar (constrains generation)
        │
        ├──renders──▶ training prompts (generator/ + finetune/to_needle_jsonl.py)
        └──copied──▶ third_party/needle2-esp32/tools/robot.json (firmware build)
```

- Prompt format baked into weights AND firmware:
  `<|im_start|>user\n<tools>{COMPACT_SCHEMA}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n`
  The model answers `<think></think>\n<tool_call>[{"name":"robot_action",
  "arguments":{"intent":...}}, ...]</tool_call>` — one JSON array, N calls,
  order matters. Schemas must be whitespace-free (firmware compacts at boot).
- Legality (which slots each intent accepts, required pairs like
  WAKEUP⇒recipient) is enforced by `schema/dsl.py:Action.from_dict`.
  The grammar constrains keys/intents but NOT cross-slot legality.

## 5. Deployment pipeline (reproduce end-to-end)

```sh
# 0. Environment (zsh): JAX on Apple Metal needs both
export ENABLE_PJRT_COMPATIBILITY=1
source ~/.espressif/tools/activate_idf_v5.5.4.fish   # or .sh twin; IDF v5.5.4

# 1. Train (fp32) — see §8 gotchas before touching flags!
python3 finetune/train_full.py --data data/finetune/train_v3aug.jsonl \
  --init checkpoints/full_v1.pkl --out checkpoints/new.pkl \
  --epochs 5 --batch-size 8 --max-len 256 --lr 1e-4 --seed 42

# 2. Export + quantize (quantization happens ONLY here, post-training)
python3 scripts/export_snapshot.py checkpoints/full_v1.pkl /tmp/out.cact \
  --bits-map 'engram0.tables=2,...,default=3'   # t4 string in V2_PLAN.md

# 3. Size gate: blob must fit model partition (15,007,744 B after repurpose)
ls -l /tmp/out.cact

# 4. Score through the REAL C engine before any flashing
cmake -S third_party/needle2-esp32/host -B /tmp/n2host && cmake --build /tmp/n2host
ENABLE_PJRT_COMPATIBILITY=1 python3 scripts/eval_c_engine.py 200 /tmp/out.cact --repair
#    ship bar: ≥85% with repair on 200 queries (deployed artifact: 88.5%)

# 5. Flash board (DevKitC-1 N16R8, port varies)
cd third_party/needle2-esp32/esp32/needle_demo
idf.py set-target esp32s3 && idf.py build && idf.py -p /dev/cu.usbmodemXXXX flash
python3 -m esptool --chip esp32s3 -p /dev/cu.usbmodemXXXX --baud 921600 \
        write_flash 0x1B0000 /tmp/out.cact
```

Flash layout (partitions.csv): nvs@0x9000 · phy@0xf000 · factory app
0x10000+0x1A0000 · **model 0x1B0000..0x1000000 (15,007,744 B)**. Upstream's
table does NOT fit our 14.65 MB blob — that's why the app slot was shrunk;
app binary is only ~267 KB.

## 6. Firmware architecture (main.c flow)

```
boot → compact schema → find+mmap "model" partition → nd_model_open(blob)
     → nd_grammar_compile(schema) → prime_prefix() [154-token prompt prefilled
     ONCE (~168 s), snapshot taken] → bench → REPL loop:
        read line → rewind to snapshot → prefill query (~13 tok)
        → generate ≤96 tokens, grammar-constrained, streaming TOK lines
        → CONF line → dispatch_call(generated)
```

- Dual-core GEMV: `rows_dual_core` splits matvec rows across cores via
  semaphore handshake (`nd_parallel_rows` hook).
- KV cache + activations live in octal PSRAM; weights are mmap'd flash,
  read in place. Decode ≈ 1.28 s/token; short request ≈ 40 s.
- Serial protocol: `EVT` (status/benchmarks), `TOK` (streamed pieces),
  `CONF <x>` (confidence), `ACT <parsed action>` (one line PER CALL),
  `ERR <why>`. `!think` toggles reasoning-stream display.
- **Repair rule lives in `parse_calls`/`tool_robot_action`**: WAKEUP without
  recipient + duration_amount ⇒ rewrite intent to WAIT. This is the single
  highest-value line of the deployment (76%→88.5%); keep it when refactoring.
  Same rule mirrored in `scripts/eval_c_engine.py --repair`.

## 7. How to add features to the LM (the ripple chain)

Adding an intent, slot, or tool touches EVERY layer — do them in this order:

1. **Schema**: edit `schema/tool_schema.json` (+ `schema/dsl.py` slot rules
   so `Action.from_dict` accepts the new shape). Regenerate
   `tool_schema_compact.json` if you use the verbose variant anywhere.
2. **Data**: extend `generator/canonical.py` + `build_dataset.py` (and/or
   `scripts/gen_wait_aug.py` pattern) to emit the new shapes; include
   boundary/negative cases; run `augment/validate.py`. Known gap to close
   while you're there: compound depth 4–6 chains (V2_PLAN.md workstream #0).
3. **Retrain fp32**: continuation FT is fine for small deltas BUT re-exporting
   at t4 re-rolls the PTQ dice (§8) — budget for a fresh bits-map search or
   move to uniform-W4 (v2a student) where export is stable.
4. **Export + eval**: pipeline steps 2–4 above. Never ship without the C-engine
   200-query run.
5. **Firmware**: copy schema to `third_party/needle2-esp32/tools/robot.json`;
   update `TOOL_TABLE`/handler in `main.c` (ACT line + LED/actuator mapping);
   rebuild. Grammar auto-adapts from the JSON — no engine changes.
6. **Host parity**: whatever parses ACT lines downstream must accept the new
   slots; keep eval harness gold-building in sync (`dsl.py` drives it).

## 8. Pitfalls (each one cost real time — do not rediscover)

- **`--bits-map` implies QAT**: passing `--bits-map` to train_full.py without
  understanding flags silently enables mixed-width STE QAT (qat_bits=-1);
  training looks perfect, exported blob is garbage. Quantize ONLY via
  export_snapshot.py after plain fp32 training.
- **PTQ chaos at ≤3 bits**: tiny weight drift (<6% relative) flips deployed
  accuracy wildly (observed 76%→8%→24%→14%→0% across epochs of harmless
  continuation training). full_v1's t4 result is a lucky basin; preserve the
  artifact, don't assume retraining reproduces it.
- **kv_window trap**: exports must carry the trained window (256); older
  exporter path defaulted wrong and degraded long prompts. Already fixed in
  export_snapshot.py via effective_kv_window(cfg) — don't regress it.
- **JAX on Mac**: without ENABLE_PJRT_COMPATIBILITY=1 imports crash; heavy
  training makes concurrent evals 7–11 min — pin eval/export processes to CPU
  (`JAX_PLATFORMS=cpu`).
- **zsh**: unquoted `=word` aborts the command; `$()` strips trailing
  newlines; don't `echo ---` (glob).
- **Serial testing**: firmware consumes buffered stdin lines sent before
  READY — your test script's queue shifts; wait for READY first. Use IDF venv
  python for pyserial.
- **Schema drift**: three copies exist (repo canonical, tools/robot.json,
  generated header). The build regenerates from robot.json, but YOU must
  re-copy after editing the canonical file.
- **/tmp volatility**: build dir `/tmp/n2esp32` and blobs under /tmp vanish on
  reboot; durable copies: repo (robot.cact, checkpoints) or ~/robot.

## 9. Where to go next

Read `docs/V2_PLAN.md` — two tracks past the current stopgap: v2a distilled
d=384 single-board student, v2b two-board cluster hosting the full-quality W4
blob; shared prerequisite = chain-depth data workstream. Ship gates and
de-risking ladders are spelled out there.
