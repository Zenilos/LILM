# Teaching a 45M-parameter language model to live inside a $8 microcontroller

*A project diary — LILM_V1, August 2026. This is the story version. The dry
version lives in [`docs/ARCHITECTURE_AND_DEPLOYMENT.md`](docs/ARCHITECTURE_AND_DEPLOYMENT.md),
the future in [`docs/V2_PLAN.md`](docs/V2_PLAN.md), and the full training saga
in [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).*

---

## Where we started

One sentence on a whiteboard: **make a real robot brain run entirely offline
on an ESP32-S3.**

Not "call an API." Not "stream from a server." A spoken-style command like
*"clean the living room and then go to kitchen"* had to become ordered tool
calls — `CLEAN(the living room)` → `MOVE(kitchen)` — generated token by token,
on a microcontroller with 512 KB of RAM, while the model itself weighs more
than thirty times the board's flash budget.

The raw material: Needle-2, a 45M-parameter Simple Attention Network — an
exotic little architecture with Hadamard-modulated MLPs, engram lookup tables,
and a confidence head — shipped as a 90 MB JAX checkpoint alongside a
closed-source native engine and a demo firmware whose entire personality was
blinking an RGB LED.

The gap between those two sentences and this repository is what this post is
about.

---

## Act I — The engine that lied

We fine-tuned the model. It got... worse. Then better, then worse again.
LoRA runs landed around 57%, then 36%, then 38% exact-match. An aggressive
QAT continuation scored 22%. Every debugging instinct pointed at the data,
the hyperparameters, the tokenizer, ourselves.

The problem was none of those. **The official evaluation engine was broken.**
When we ran the *exact same weights* through an independent C99 implementation
we built and patched ourselves, the score jumped from 38% to 100%.

> **Learning #1 — Your runtime is part of your model.** A closed eval path
> doesn't just hide bugs, it manufactures them — and every hour spent tuning
> against a lying oracle makes the model worse, not better. We now treat the
> C engine harness (`scripts/eval_c_engine.py`) as the only truth.

That single fix carried us to **199/200 = 99.5%** end-to-end. For about a day,
we thought we were done.

---

## Act II — The size wall

Then we measured the export: **23.17 MB**. The board's model partition:
**~14.6 MB**.

The byte audit explained everything. In this architecture, cost scales with
d² — the attention projections alone are 21M parameters — plus vocab-sized
tables you cannot shrink without shrinking the vocabulary itself. Squeezing
under the ceiling forced 2-bit weights exactly where the model keeps its
finest distinctions. The WAIT/WAKEUP boundary — two intents separated by
nothing but the presence of a duration — collapsed to zero. Best fitting
artifact: **76%**, with every miss being the same mistake.

So we did what everyone does: we tried QAT. Three times. Four times. Training
loss went to zero while deployed accuracy fell off a cliff — 60%, 82%, 42%.
Root cause, eventually: the training-time fake-quantization and the exporter's
real quantization are *different functions*. QAT was optimizing a world that
didn't exist.

> **Learning #2 — Quantize once, after training, through one code path.**
> If your STE simulation doesn't bit-match your exporter, QAT is not
> compression, it's sabotage wearing a lab coat.

---

## Act III — Chaos

Fine. Keep training longer, add data — we built 700 synthetic WAIT/WAKEUP
examples, merged them into the fine-tune set, relaunched. The fp32 validation
loss fell to 0.0000. Textbook convergence.

The quantized snapshots scored **8% → 24% → 14% → 0%** across five epochs.

That sequence broke our mental model completely. Identical pipeline, healthy
training, and the deployed artifact swung wildly — because at 2–3 bits,
accuracy is a *chaotic function of the weights*. We diffed checkpoints:
no parameter moved more than 5.5%, most under 1%. The fp32 models were
near-identical twins. Their quantized souls were unrecognizable strangers.
Our original checkpoint wasn't just good — it was **lucky**, a random draw
that happened to land in one of the few basins where coarse codebooks still
mean something.

> **Learning #3 — At extreme compression, reproducibility dies first.** You
> can't retrain your way back to a lucky artifact. Treat it like a rare
> mineral: preserve it, study it, build around it.

---

## Act IV — Ten lines worth more than a GPU-week

Staring at the failure log, a pattern: *every* WAIT failure emitted
`WAKEUP` with a duration filled in and **no recipient** — an output that isn't
even legal under our own schema. WAKEUP requires someone to wake.

So we wrote ten lines: *if you see WAKEUP with a duration and nobody to wake,
that's a WAIT.* No retraining. No new data. Counterfactual score on saved
outputs: 76% → 86%. Confirmed on fresh queries: **88.5%**.

> **Learning #4 — Spend your ML budget last.** A constraint violation is not
> a modeling failure; it's an invitation to write an if-statement.

---

## Act V — Deployment day

Flashing day produced its own gauntlet: our blob missed upstream's flash
layout by 35,687 bytes (fixed by shrinking an absurdly oversized app slot —
the firmware is 267 KB); the IDF component wrapper the README implies but
doesn't ship; Xtensa's compiler rejecting printf lines that x86 clang happily
ignored. And then:

```
EVT READY - type a request and press enter
Q: wait 30 minutes
ACT robot_action intent=WAIT duration=30 minutes    ← the repair rule, live
CONF 1.0000
```

A language model, reading its weights straight out of flash mmap, generating
grammar-constrained JSON, at 1.28 seconds per token, on a board that costs
less than lunch.

---

## Epilogue — Two diseases, one symptom

We stress-tested with crueler sentences, comparing the deployed compressed
model against its full-quality twin on a Mac:

| Symptom | Compressed | Full quality | Actual disease |
|---|---|---|---|
| "my daughter" garbled to "my mom" | ✗ | ✓ | Quantization |
| Five-step command chains collapsing mid-way | ✗ | ✗ | Training data never contained chains that long |

Same visible failure, two unrelated roots. That diagnosis now shapes
everything next: a two-board variant hosting the uncompressed model kills the
first disease outright; a deeper-data generator cures the second; neither
works without the other.

---

## The ledger

**Biggest risks that actually bit us:** trusting a broken oracle (Act I);
a silent config flag that enabled QAT behind our backs; irreproducibility of
our best artifact; three copies of one schema quietly drifting apart.

**What we'd tell the next crew:** the whole hard-won playbook is in
[`docs/ARCHITECTURE_AND_DEPLOYMENT.md`](docs/ARCHITECTURE_AND_DEPLOYMENT.md)
— including the pitfalls list, each line of which has a scar behind it.

**Status:** deployed and verified on hardware. 88.5% fitting the budget,
99.5% waiting for more flash or a smaller student. The plan to get there is
written, costed, and de-risked. The robot is listening.
