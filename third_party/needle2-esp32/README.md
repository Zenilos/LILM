# needle2-esp32 (vendored, patched)

Source: https://github.com/andrisgauracs/needle-2-esp32
Independent C99 reader/engine for Needle 2 `.cact` blobs (tag 0x05E12A83),
validated to run on-host and on ESP32-S3.

## Why vendored

The official native engine (`libneedle.dylib`, engine 2.0.3) reads our
fine-tuned W4 blob correctly at prefill but diverges during generation,
producing base-model dialect text instead of our tool calls. This engine,
once patched, reproduces the JAX reference bit-for-bit on the same blob and
scores 199/200 (99.5%) on the robot eval — see `scripts/eval_c_engine.py`.

## Patch applied (upstream bug)

`nd_cq_lut_build()` + `dot_group_lut2()` in `engine/src/nd_quant.c` hardcode
the **2-bit codebook** and 2-bit index packing. Any tensor with `bits != 2`
projected through the LUT path (engram k/v, attention q/k/v/gate/out) yields
garbage. The official `needle2.cact` ships CQ2 for all projection weights, so
the bug is invisible upstream; our export (`needle.model.export.write_export`,
bits=4 uniform) exposes it.

Fix in `engine/src/nd_model.c`: all LUT call sites now branch on
`t->bits == 2`; other widths take the verified generic `nd_cq_gemv()`.
Slower per-token on host (~2x), still trivially fast; re-optimize later if
needed for device.

## Layout

- `engine/` — engine sources (patched as above)
- `host/host_runner.c` — CLI: `host_runner <blob> <tool_schema.json> "<query>"`,
  primes system+tools prefix once, rewinds per query, greedy-generates.
- `tools/ref_forward.py`, `tools/check_engine.py` — upstream NumPy reference
  forward + blob validator (both agree with JAX on our blob).

## Build

    cc -O2 -std=c99 -Iengine/include -o build/host_runner \
       host/host_runner.c engine/src/*.c -lm

## Eval

    python scripts/eval_c_engine.py 200   # -> 199/200 = 99.5%
