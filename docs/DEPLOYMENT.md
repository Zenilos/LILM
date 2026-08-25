# Deployment — robot firmware on ESP32-S3 DevKitC-1 N16R8

Status: **deployed and verified on hardware** (2026-08-25)
Board: ESP32-S3 DevKitC-1, 16 MB quad flash, 8 MB octal PSRAM
Toolchain: ESP-IDF v5.5.4 (`source ~/.espressif/tools/activate_idf_v5.5.4.fish`,
or the `.sh` twin for POSIX shells)

## What runs where

| Flash region | Offset | Size | Contents |
|---|---|---|---|
| bootloader+app | `0x0`–`0x1B0000` | 1.7 MB | IDF bootloader + needle_demo.bin (267 KB) |
| `model` partition | `0x1B0000`–`0x1000000` | 15,007,744 B | `/tmp/robot_t4.cact` = 14,651,215 B (356 KB margin) |

**Layout change vs upstream:** upstream's partitions.csv gives model only
13.125 MB and leaves a full 2 MB for the app. Our t4 blob is 14.65 MB — it
does not fit even if model took every byte after a 2 MB app (short by
35,687 B). The demo app is actually 267 KB, so `factory` shrinks to
1.625 MB and `model` extends to exactly end of flash.

## Firmware changes (all in third_party/needle2-esp32)

1. `esp32/needle_demo/partitions.csv` — layout above.
2. `esp32/needle_demo/main/CMakeLists.txt` — schema source is
   `tools/robot.json` (the same canonical JSON used by training + eval;
   single source of truth preserved).
3. `esp32/needle_demo/main/main.c`:
   - `tool_robot_action` handler: parses intent + slots, prints
     `ACT robot_action ...`, maps intents to LED patterns as feedback.
   - **Repair rule on device**: WAKEUP without recipient but with
     duration_amount ⇒ rewritten to WAIT (see docs/V2_PLAN.md; 76% → 88.5%
     at 200 queries). Verified live: "wait 30 minutes" produces
     `ACT robot_action intent=WAIT duration=30 minutes`.
   - `dispatch_call` walks multi-call arrays: trained outputs pack several
     actions into one `<tool_call>[{...},{...}]</tool_call>`; upstream's
     dispatcher executed only the first call.
4. `components/needle/CMakeLists.txt` — wraps repo-root `engine/` as an IDF
   component (upstream ships no such wrapper; README implies one exists).
5. `engine/src/nd_model.c` — printf casts to `(unsigned)` for xtensa, where
   `uint32_t` is `unsigned long` and `-Werror=format` rejects `%u`.

## Build & flash

```sh
cd third_party/needle2-esp32/esp32/needle_demo
idf.py set-target esp32s3 && idf.py build
idf.py -p /dev/cu.usbmodemXXXX flash          # app + partitions
python3 -m esptool --chip esp32s3 -p /dev/cu.usbmodemXXXX --baud 921600 \
    write_flash 0x1B0000 /tmp/robot_t4.cact    # model blob (~90 s @921600)
```

## Measured behaviour on board

- Boot: prefix priming 154 tokens ≈ 168 s once (cached via snapshot);
  bench 1.18 s/token decode.
- Per request: prefill ~13 tokens, then ~1.28 s/token generation,
  typical answer ≈ 31 tokens ≈ 40 s, CONF 1.0000.
- Smoke tests all passed: CLEAN (location slot), WAIT 30 min (**repair
  fired**), GIVE object+recipient from a compound query, STOP.
- Host-side harness score of this exact artifact: **88.5% (177/200)** with
  repair; per-intent table in docs/V2_PLAN.md.

## Known gaps

- Serial line input has no echo/backspace handling (matches upstream TUI
  contract; use tools/mon.py or any terminal).
- The compound-query test above executed both calls; ACT lines stream per
  call, so hosts must treat consecutive ACT lines as one turn.
- GIVE remains the weakest intent host-side (18/27); v2 tracks address this.
