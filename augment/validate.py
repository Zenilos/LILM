from __future__ import annotations

import httpx
import json
import time
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "z-ai/glm-5.2:free",
]

SYSTEM_PROMPT = """You verify robot command labels. Given a user command and a proposed action list, decide if the actions exactly match what the command asks.
Actions: MOVE(location), CLEAN(location?), PLAY(file), SHOW(message, person?), GET(object), GIVE(object, recipient), STOP(), WAIT(duration), WAKEUP(recipient).
Rules:
- Every action in the command must appear, in order. No extra actions. No missing actions.
- Slot values must be the words used in the command (raw spans, e.g. "five minutes" not 300).
- WAKEUP is a single action even if it implies moving; do not split it.
- Politeness, typos and filler ("could you", "hey robot") do not change the label.
Reply with ONLY one letter: Y if the labels are correct, N if not."""


def _load_key() -> str:
    env = Path(__file__).resolve().parents[1] / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("openrouter="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no openrouter key in .env")


def chat(messages: list[dict], model: str, max_retries: int = 3) -> str:
    key = _load_key()
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4,
    }).encode()
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=60) as client:
                r = client.post(
                    OPENROUTER_URL,
                    content=body,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                )
                r.raise_for_status()
                data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def check_example(text: str, actions: list[dict], model: str) -> bool:
    label = json.dumps([{"intent": a["intent"], "slots": a.get("slots", {})} for a in actions])
    out = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Command: {text!r}\nProposed actions: {label}\nCorrect?"},
    ], model)
    return out.upper().startswith("Y")
