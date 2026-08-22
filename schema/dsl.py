from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

INTENTS = (
    "MOVE",
    "CLEAN",
    "PLAY",
    "SHOW",
    "GET",
    "GIVE",
    "STOP",
    "WAIT",
    "WAKEUP",
)

SLOTS = {
    "MOVE": {"required": ("location",), "optional": ()},
    "CLEAN": {"required": (), "optional": ("location",)},
    "PLAY": {"required": ("file",), "optional": ()},
    "SHOW": {"required": ("message",), "optional": ("person",)},
    "GET": {"required": ("object",), "optional": ()},
    "GIVE": {"required": ("object", "recipient"), "optional": ()},
    "STOP": {"required": (), "optional": ()},
    "WAIT": {"required": ("duration",), "optional": ()},
    "WAKEUP": {"required": ("recipient",), "optional": ()},
}

ALL_SLOTS = (
    "location",
    "object",
    "recipient",
    "file",
    "duration",
    "message",
    "person",
)

_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "fifteen": "15", "twenty": "20", "thirty": "30",
    "forty five": "45", "sixty": "60", "ninety": "90", "half": "half",
}


@dataclass(frozen=True)
class Action:
    intent: str
    slots: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent: {self.intent}")
        allowed = set(SLOTS[self.intent]["required"]) | set(SLOTS[self.intent]["optional"])
        bad = set(self.slots) - allowed
        if bad:
            raise ValueError(f"slots {bad} not allowed for {self.intent}")
        missing = [s for s in SLOTS[self.intent]["required"] if not self.slots.get(s)]
        if missing:
            raise ValueError(f"{self.intent} missing required slots {missing}")

    def to_dict(self) -> dict:
        return {"intent": self.intent, "slots": dict(sorted(self.slots.items()))}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(intent=d["intent"], slots={k: v for k, v in (d.get("slots") or {}).items() if v})


def normalize_value(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[.,!?;:\"']+$", "", t)
    t = re.sub(r"^(please\s+)?(the|a|an)\s+", "", t)
    t = re.sub(r"\s+", " ", t)
    words = [_NUMBERS.get(w, w) for w in t.split(" ")]
    return " ".join(words).strip()


def _values_match(a: str, b: str) -> bool:
    na, nb = normalize_value(a), normalize_value(b)
    if na == nb:
        return True
    return na.replace(" ", "") == nb.replace(" ", "")


def action_matches(pred: Action, gold: Action) -> bool:
    if pred.intent != gold.intent:
        return False
    pred_slots = {k: v for k, v in pred.slots.items() if v}
    for slot in SLOTS[gold.intent]["required"]:
        if slot not in pred_slots:
            return False
        if not _values_match(pred_slots[slot], gold.slots[slot]):
            return False
    for slot in SLOTS[gold.intent]["optional"]:
        if gold.slots.get(slot) and not pred_slots.get(slot):
            return False
        if pred_slots.get(slot) and not gold.slots.get(slot):
            return False
        if pred_slots.get(slot) and not _values_match(pred_slots[slot], gold.slots[slot]):
            return False
    return True


def actions_match(pred: Optional[list[Action]], gold: list[Action]) -> bool:
    if pred is None or len(pred) != len(gold):
        return False
    return all(action_matches(p, g) for p, g in zip(pred, gold))


def load_actions(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["actions"] = [Action.from_dict(a) for a in rec["actions"]]
                out.append(rec)
    return out


if __name__ == "__main__":
    a1 = Action("MOVE", {"location": "my room"})
    a2 = Action.from_dict({"intent": "MOVE", "slots": {"location": "My Room"}})
    assert action_matches(a2, a1), a2
    assert action_matches(Action("MOVE", {"location": "the kitchen"}), Action("MOVE", {"location": "kitchen"}))
    assert not action_matches(Action("MOVE", {"location": "The Room"}), Action("MOVE", {"location": "my room"}))
    assert actions_match([a1], [Action("MOVE", {"location": "My room"})])
    assert not actions_match([Action("WAIT", {"duration": "5 seconds"}), a1], [a1, Action("WAIT", {"duration": "five seconds"})])
    assert actions_match([Action("WAIT", {"duration": "5 seconds"}), a1],
                         [Action("WAIT", {"duration": "five seconds"}), a2])
    try:
        Action("PLAY", {})
        raise AssertionError("should have raised")
    except ValueError:
        pass
    try:
        Action("STOP", {"location": "kitchen"})
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("dsl.py OK")
