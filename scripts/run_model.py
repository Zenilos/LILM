#!/usr/bin/env python3
"""Run the fine-tuned robot model locally on Mac (Metal/CPU).

Usage:
  python run_model.py                      # interactive REPL
  python run_model.py "go to my room" ...  # one-shot queries
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "robot.cact"
SCHEMA = ROOT / "schema/tool_schema.json"
SYSTEM = "device: domestic robot; locale: en-US"


def make_agent():
    import needle
    return needle.Needle(
        weights=str(WEIGHTS),
        tools=json.load(open(SCHEMA)),
        system=SYSTEM,
    )


def parse(agent, text: str) -> list[dict]:
    resp = agent.complete(text)
    calls = []
    for c in resp.get("function_calls") or []:
        a = dict(c.get("arguments") or {})
        calls.append({"intent": a.pop("intent", "?"),
                      "slots": {k: v for k, v in a.items()}})
    return calls


def main():
    if not WEIGHTS.exists():
        sys.exit(f"missing {WEIGHTS} — download it from Colab first")

    agent = make_agent()

    if len(sys.argv) > 1:
        for q in sys.argv[1:]:
            print(f"{q!r}\n  -> {json.dumps(parse(agent, q))}")
        return

    print("robot NLU — type a command, empty line to quit")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print(f"  -> {json.dumps(parse(agent, q), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
