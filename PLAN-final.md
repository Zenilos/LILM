# Final Plan: Needle-Distilled Intent+Slot NLU for ESP32-S3

**Status:** chosen plan — supersedes the 4 drafts. Takes the label-first discipline of `needle esp32.md`, the structure of `v3`, drops what's wrong in each.
**Objective:** text in → ordered list of atomic actions with slot values out. Fully offline on XIAO ESP32S3 Sense.
**Compute split:** dataset generation + teacher validation on this Mac (M3 Pro, `~/p3.11`), training on Colab free tier (T4), conversion/export back on the Mac.

---

## 0. What we will have at the end (definition of done)

1. **A reusable dataset factory**: DSL schema → deterministic generator → Needle-validated JSONL. Rerunnable when you add intents.
2. **A validated dataset**: ~15–30k labeled examples (`text` → `list[Action]`), plus held-out Test A/B/C splits, Test C being 300+ commands you hand-wrote.
3. **A trained student model**: Char-CNN, dual-head (multi-label intents + character-level BIO slot spans), meeting on Test C:
   - Intent F1 > 95%, Slot entity F1 > 90%, Exact action-sequence accuracy > 85%
4. **Deployment artifact**: `model_quantized.tflite` (< 200 KB, full INT8) + `model_data.cc` C array, verified within ~2 points of the float model.
5. **A host-side demo script**: type any command → get the exact action list the firmware would execute.

Not included in this phase: firmware/C++ runtime (Phase 7, separate effort), ASR noise modeling (input is typed text).

---

## 1. Verified facts about Needle 2 (checked against the repo)

| Fact | Consequence |
|---|---|
| `pip install cactus-needle`; engine auto-downloads from HF, cached | Runs locally on the M3 (use `cactus-needle[metal]`) |
| `@needle.tool` + `agent.complete(text)` → `function_calls[]`, one call per atomic action | Native fit for our compound-command output format |
| Byte-level grammar from schemas → structurally valid output, no JSON regexing | Declare tools, never parse prose |
| Calibrated `confidence` per response | Use as acceptance filter for generated data |
| Off-topic → `function_calls: []` | Built-in reject class |
| `needle generate-data --augment` uses OpenRouter (external LLMs, not Needle) | Our paraphrase-augmentation path; works with free models |
| No free-text generation in Needle itself | Variation must come from our generator / OpenRouter, never from Needle |
| `needle finetune` LoRA exists but disables the confidence head, and tuned .cact still needs ~28 MB RAM | Fine-tuning Needle is out of scope; ESP32 can't hold it anyway |

---

## 2. DSL — single source of truth

Eight atomic intents (HANDOVER decomposed into GET + GIVE; WAKEUP is its own intent — the *understanding* layer extracts goal + recipient, firmware decides it plays a sound):

| Intent | Required slots | Optional slots |
|---|---|---|
| `MOVE` | `LOCATION` | |
| `CLEAN` | | `LOCATION` |
| `PLAY` | `FILE` | (SD-card filename: `song.mp3`, `alarm.wav`) |
| `SHOW` | `MESSAGE` | `PERSON` |
| `GET` | `OBJECT` | |
| `GIVE` | `OBJECT`, `RECIPIENT` | |
| `STOP` | | |
| `WAIT` | `DURATION` | |
| `WAKEUP` | `RECIPIENT` | (NLU emits ONE action; C++ planner expands to MOVE-to-person + wake sound) |

Canonical action object:

```python
@dataclass(frozen=True)
class Action:
    intent: str            # one of the 8
    slots: dict[str, str]  # only keys allowed for that intent
```

A prediction is `list[Action]`, order = execution order. This object is what the generator emits, what Needle must reproduce, what the student predicts, and what firmware will consume.

Notes:
- The student extracts **raw spans** ("where I cook", "five minutes"). Mapping to room IDs and seconds happens in an entity-resolver table / duration parser at runtime — never inside the model.
- PLAY is limited to SD-card files, so `FILE` values are from a closed-ish vocabulary you control; the generator should still include unseen filenames in Test C to prove span extraction generalizes.

---

## 3. Architecture

```text
MAC (dataset factory)                      COLAB (training)                MAC (deploy)
─────────────────────                      ─────────────────               ─────────────
schema/dsl.py (Action, intents)
        │
generator: canonical → synonyms →          student/train.py (Char-CNN,
light typos → compounds w/ stored           dual head) → best_model.h5
boundaries → off-topic                              │
        │                                           ▼
Needle complete() validation                student/evaluate.py
(confidence ≥ τ AND match vs known          Test A/B/C metrics
label) ──► validated/dataset.jsonl                  │
        │                                           ▼
OpenRouter free-model paraphrases ──►       deployment/convert_tflite.py
(same validation loop)                      full-INT8 ──► benchmark
        │                                           │
Test A/B/C construction                    ◄──────────┘
hand-written Test C                        xxd -i ──► model_data.cc
```

### 3.1 Compound handling (the key design decision)

Global "bag of intents + bag of slots" cannot bind `"go to the kitchen and play music in the bedroom"` correctly. Fix:

- Runtime clause splitter (deterministic: `and`, `then`, `,`, `.`) cuts utterances into atomic clauses.
- The student sees **one clause at a time**, predicts one primary action + spans.
- Training data stores clause boundaries explicitly, so per-clause examples are generated directly — no re-splitting ambiguity.
- Safety net: the intent head stays **sigmoid multi-label** (not softmax) so an unsplit compound degrades gracefully instead of silently picking one intent.

---

## 4. Phase plan

### Phase 0 — Schema + seed set (Mac)
- `schema/dsl.py`: intents, allowed slots, `Action`, `normalize()` (casefold, strip articles, `5`↔`five`), `match()`.
- Hand-write **200–400 reviewed** `(text, list[Action])` seed pairs covering every intent, every slot, compounds, off-topic. Include your real usage sentences.
- Unit-test `Action.match()`.

**Exit:** `data/seed_commands.jsonl` reviewed by you.

### Phase 1 — Deterministic generator (Mac)
- Per-intent synonym tables, template expansion, realistic typo injection (swaps/drops, homophone-lite since input is typed), compound joins (`and`, `and then`, `then`, `,`) with stored boundaries, off-topic class (`[]`).
- Emit 2–5k labeled examples. Character-length histogram → fix `MAX_LEN` (start 64/clause).

**Exit:** `data/generated/v1.jsonl`, labels perfect by construction.

### Phase 2 — Needle teacher loop (Mac, `cactus-needle[metal]`)
- One `@needle.tool` per intent, good docstrings (they are the whole game for Needle). Map tool name → intent enum; drop empty optionals.
- `agent.complete(text)` per example, **cache every raw response** so nothing is recomputed.
- Keep an example iff: calls non-empty (or empty for off-topic seeds), `confidence ≥ CONF_MIN` (probe 200 examples to set τ, start 0.70), predicted `list[Action]` semantically matches the known label under `normalize()`.
- Everything else → `rejected.jsonl`. If Needle systematically misses a phrasing, **fix the tool docstring**, not the data.

**Exit:** ≥ 80% acceptance rate; disagreement log inspected.

### Phase 3 — OpenRouter augmentation (Mac, optional but recommended)
- Use your `.env` key with **free models** via `needle generate-data --augment seed.jsonl` (or direct OpenAI-compatible calls) to get linguistically diverse paraphrases beyond what templates produce.
- Every paraphrase passes through the same Phase-2 validation loop before entering the dataset. Target ~15–30k total validated examples.

**Exit:** `data/validated/dataset.jsonl` + `rejected.jsonl`.

### Phase 4 — Splits + Test C
- **Test A** (~500): held-out generator templates. **Test B** (~500): held-out paraphrase/noise patterns. **Test C** (≥300): sentences you write by hand, never generated — includes typos, polite forms, weird filenames, compounds. Test C is the number that matters; teacher-agreement accuracy is not product accuracy.

**Exit:** `data/tests/{a,b,c}.jsonl`.

### Phase 5 — Train the student (Colab free T4)
Upload only `dataset.jsonl` + tests. Keras Char-CNN, no recurrence:

```text
char ids [B,64] → Embedding(≈52 vocab, 16)
→ Conv1D(32,k3) → Conv1D(32,k3) → Conv1D(32,k5), padding=same
→ [ GlobalMaxPool → Dense(16) → Dense(N_intents, sigmoid) ]  # 9 intents
→ [ Conv1D(num_tags, k1) softmax per char ]              # BIO slots
```

- Vocab: `a–z 0–9 space . , ? ! ' " -` + PAD/UNK (apostrophe/hyphen mandatory: `don't`, `living-room`).
- Losses: BCE (intents), sparse CE (slots, ignore PAD). Adam 1e-3, batch 64, early stop on Test A slot-F1.
- Decoding: sigmoid ≥ 0.6 flags; BIO argmax + repair (`O I-x` → `B-x`); drop slots not allowed for the intent.
- Model is tiny (~40k params): trains in minutes even on CPU, so free Colab is plenty.

**Exit:** float model meets Test C bars (Intent F1 > 95%, Slot F1 > 90%, exact-sequence > 85%).

### Phase 6 — Quantize + export (Colab then Mac)
- Full-integer INT8 TFLite, representative set = 200 real commands.
- Re-run **all metrics on the quantized graph**; require ≤ ~2 points drop vs float.
- Record size (< 200 KB target). `xxd -i` → `model_data.cc`.
- Host demo: `python demo.py "clean the kitchen then play alarm.wav"` → printed action list.

**Exit:** `model_quantized.tflite` + `model_data.cc` + quantized-metric report.

### Phase 7 — Firmware (separate follow-up, not this phase)
TFLM on the S3, tokenizer/splitter/BIO-decode/resolver/duration-parser in C++, arena measured on hardware with camera/mic live.

---

## 5. Repository layout

```text
LILM_V1/
├── PLAN-final.md
├── schema/dsl.py                 # Action, intents, normalize(), match()
├── schema/tools.py               # Needle @tool declarations
├── generator/{canonical,synonyms,noise,compound}.py
├── teacher/{needle_client.py,validate.py}   # caching + confidence gate
├── augment/openrouter_augment.py
├── data/
│   ├── seed_commands.jsonl
│   ├── generated/  validated/  rejected.jsonl
│   └── tests/{a,b,c}.jsonl
├── student/{tokenizer.py,dataset.py,model.py,train.py,evaluate.py}
├── deployment/{convert_tflite.py,benchmark.py,export_c_array.py,demo.py}
└── notebooks/colab_train.ipynb   # thin wrapper: upload jsonl → train → download artifacts
```

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Needle rejects too many generated examples | Fix tool docstrings; loosen templates; τ tuned on a 200-example probe |
| Paraphrases lose verbatim slot spans → BIO alignment fails | Alignment: exact substring → fuzzy (Levenshtein/token-F1) → Needle `reasoning` quotes → else drop the example |
| Clause splitter fails ("show John's message and give him the cup") | v1: live with it; sigmoid head degrades gracefully; revisit only if Test C shows real failures |
| Free OpenRouter models produce low-quality paraphrases | Everything passes Needle validation regardless of source; bad sources show up in rejection rate and get dropped |
| Colab session dies mid-training | Training is minutes-scale; checkpoint to Drive; dataset versioned locally |
| INT8 accuracy drop | int8 weights/int16 activations fallback before touching architecture |
| `PLAY` FILE names overfit to seen filenames | Ensure Test C contains unseen filenames; span extraction is char-level so it should generalize |

## 7. What not to do
- Never train on Needle labels that disagree with the generator's known label.
- Never report teacher-agreement as product accuracy — Test C gates release.
- Don't ask Needle to paraphrase; it can't. Templates + OpenRouter do variation.
- Don't start with LSTM/GRU; don't put home topology in the model.
