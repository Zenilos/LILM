# Requirements

Everything needed on a fresh macOS (Apple Silicon) machine to run this project.

## Automated setup
```bash
git clone https://github.com/Zenilos/LILM.git && cd LILM
./scripts/setup_mac.sh          # installs everything, ~10-20 min first run
source ~/p3.11/bin/activate     # activate the project env afterwards
```

## What setup_mac.sh installs / verifies

| Component | Version | Purpose |
|---|---|---|
| Homebrew | latest | package manager (skipped if present) |
| Python 3.11 | via brew | project runtime → `~/p3.11` venv |
| `cactus-needle[metal]` | 2.x | Needle inference, finetune (JAX/Metal), build |
| TensorFlow | 2.x (mac) | Path-B CNN student (kept for future N8R8 path) |
| Pydantic | 2.x | schema validation |
| httpx | latest | OpenRouter/Ollama HTTP calls |
| Ollama + `qwen2.5:7b-instruct` | current | local validator LLM (~4.7 GB download) |

## Runtime notes (learned the hard way — see PLAN-final.md)
- Finetune: `JAX_PLATFORMS=METAL --batch-size 4 --max-len 192` on 16 GB Macs;
  batch-size ≥ 8 OOM-kills and batch < 16 silently zeroes the loss.
  On ≥ 32 GB Macs try `--batch-size 16` and WATCH the loss prints.
- Stop Ollama (`brew services stop ollama`) before Metal finetune runs.
- Colab T4 is the preferred training machine; see `notebooks/colab_onecell.py`.

## Hardware targets
- XIAO ESP32-S3 **N16R8**: deployment board (16 MB flash / 8 MB octal PSRAM) — required
- ESP32-S3 **N8R8**: NOT able to host Needle 2 (flash too small); would need Path-B CNN
