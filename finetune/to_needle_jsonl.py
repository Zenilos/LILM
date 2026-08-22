from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SLOT_KEYS = ["location", "object", "recipient", "file",
             "duration_amount", "duration_unit", "message", "person"]


def example_to_answer(action: dict) -> dict:
    args = {"intent": action["intent"]}
    for k in SLOT_KEYS:
        v = (action.get("slots") or {}).get(k)
        if v not in (None, ""):
            args[k] = v
    return {"name": "robot_action", "arguments": args}


def reasoning_for(text: str, actions: list[dict]) -> str:
    parts = []
    for a in actions:
        bits = [f"{a['intent'].lower()}"]
        for k, v in (a.get("slots") or {}).items():
            needle_span = f"'{v}' -> {k}"
            parts.append(needle_span)
    return "; ".join(parts) if parts else "no actionable request"


def convert(records: list[dict], tool_schema: list[dict]) -> list[dict]:
    tools_compact = tool_schema
    out = []
    for r in records:
        actions = r["actions"]
        answers = [example_to_answer(a) for a in actions]
        rec = {
            "query": r["text"],
            "tools": tools_compact,
            "answers": answers,
        }
        if answers and answers != [{"name": "robot_action", "arguments": {"intent": "UNAVAILABLE"}}]:
            rec["reasoning"] = reasoning_for(r["text"], actions)
        out.append(rec)
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/generated/v1.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/finetune/train.jsonl"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    schema = json.load(open("schema/tool_schema.json"))
    records = [json.loads(l) for l in open(src) if l.strip()]
    converted = convert(records, schema)
    with open(dst, "w") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_multi = sum(1 for r in converted if len(r["answers"]) > 1)
    n_unavail = sum(1 for r in converted
                    if r["answers"] == [{"name": "robot_action",
                                         "arguments": {"intent": "UNAVAILABLE"}}])
    print(f"wrote {len(converted)} examples -> {dst}")
    print(f"compound: {n_multi}, unavailable: {n_unavail}")
