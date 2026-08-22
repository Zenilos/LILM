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
- Slot values must be the words used in the command (raw spans), except duration which is normalized: "five seconds" -> 5/seconds, "half an hour" -> 30/minutes, "a couple minutes" -> 2/minutes.
- WAKEUP is a single action even if it implies moving; do not split it.
- Commands that are not understandable, off-topic, or impossible for a home robot (e.g. "go wash yourself") are labeled UNAVAILABLE.
- Politeness, typos and filler ("could you", "hey robot") do not change the label.
Start your reply with exactly Y or N, then optionally a short reason."""


def _load_key() -> str:
    env = Path(__file__).resolve().parents[1] / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("openrouter="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no openrouter key in .env")


def chat(messages: list[dict], model: str | None = None, max_retries: int = 6) -> str:
    key = _load_key()
    body = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 300,
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        use_model = model or FREE_MODELS[attempt % len(FREE_MODELS)]
        body["model"] = use_model
        try:
            with httpx.Client(timeout=60) as client:
                r = client.post(
                    OPENROUTER_URL,
                    content=json.dumps(body).encode(),
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                )
                r.raise_for_status()
                data = r.json()
            content = data["choices"][0]["message"]["content"]
            if content:
                return content.strip()
            last_err = RuntimeError(f"empty content from {use_model}")
        except Exception as e:
            last_err = e
        time.sleep(2 ** min(attempt, 4))
    raise RuntimeError(f"all retries failed: {last_err}")


def check_example(text: str, actions: list[dict], model: str | None = None) -> bool:
    label = json.dumps([{"intent": a["intent"], "slots": a.get("slots", {})} for a in actions])
    out = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Command: {text!r}\nProposed actions: {label}\nCorrect?"},
    ], model)
    for ch in out.upper():
        if ch in "YN":
            return ch == "Y"
    raise ValueError(f"no verdict in reply: {out!r}")
