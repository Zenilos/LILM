from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PEOPLE = ["Alex", "mom", "dad", "my wife", "my husband", "the guests",
          "grandma", "my sister", "john", "the kids"]
AMOUNTS = ["1", "2", "3", "5", "10", "15", "20", "30", "45", "60", "90"]
UNITS = [("minutes", ["minutes", "minuts", "minnutes", "mins", "minute"]),
         ("seconds", ["seconds", "secnds", "secs", "second"])]
FILLERS = ["", " please", " now", " right now", " will you", ", ok?", " robot,", " hey robot,"]
WORD_NUMBERS = {"a minute": ("1", "minutes"), "an hour": ("60", "minutes"),
                "half an hour": ("30", "minutes"), "half a minute": ("30", "seconds"),
                "a sec": ("1", "seconds"), "one minute": ("1", "minutes"),
                "two minutes": ("2", "minutes"), "five minutes": ("5", "minutes")}
WAIT_VERBS = ["wait", "wait for", "hold on", "hold on for", "pause", "pause for",
              "hang on", "hang on for", "chill for", "hold still for", "stand by for"]
WAKE_TEMPLATES = ["wake {p} up", "wake up {p}", "wake {p}", "rouse {p}",
                  "go and wake {p}", "{p} needs to get up", "get {p} up",
                  "wakke {p} up", "please wake up {p}", "can you wake {p}"]
LOCATIONS = ["the kitchen", "the living room", "the bedroom", "here"]


def dur_phrase(rng):
    if rng.random() < 0.25:
        w = rng.choice(list(WORD_NUMBERS))
        return w, WORD_NUMBERS[w]
    unit_canon, unit_forms = rng.choice(UNITS)
    amt = rng.choice(AMOUNTS)
    form = rng.choice(unit_forms)
    return f"{amt} {form}", (amt, unit_canon)


def wait_sentence(rng):
    phrase, (amt, unit) = dur_phrase(rng)
    verb = rng.choice(WAIT_VERBS)
    s = f"{verb} {phrase}"
    s += rng.choice(FILLERS)
    return s.strip(), {"duration_amount": amt, "duration_unit": unit}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/generated/v1.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/generated/wait_aug.jsonl"
    n_target = int(sys.argv[3]) if len(sys.argv) > 3 else 700
    rng = random.Random(7)
    base = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    others = [r for r in base if not any(a["intent"] in ("WAIT", "WAKEUP") for a in r["actions"])]

    out = []
    for _ in range(n_target):
        roll = rng.random()
        if roll < 0.38:
            text, slots = wait_sentence(rng)
            actions = [{"intent": "WAIT", "slots": slots}]
            kind = "atomic"
        elif roll < 0.68:
            p = rng.choice(PEOPLE)
            text = rng.choice(WAKE_TEMPLATES).format(p=p) + rng.choice(FILLERS)
            actions = [{"intent": "WAKEUP", "slots": {"recipient": p}}]
            kind = "atomic"
        elif roll < 0.83:
            p = rng.choice(PEOPLE)
            text = rng.choice(WAKE_TEMPLATES).format(p=p)
            phrase, (amt, unit) = dur_phrase(rng)
            text += f", then hold on for {phrase}"
            actions = [{"intent": "WAKEUP", "slots": {"recipient": p}},
                       {"intent": "WAIT", "slots": {"duration_amount": amt, "duration_unit": unit}}]
            kind = "compound"
        elif roll < 0.94:
            phrase, (amt, unit) = dur_phrase(rng)
            oth = rng.choice(others)
            first = oth["actions"][0]
            loc = rng.choice(LOCATIONS)
            lead = {"CLEAN": f"clean {loc}", "MOVE": f"go to {loc}"}.get(first["intent"])
            if lead is None:
                text = f"{first['intent'].lower()} a bit, then wait for {phrase}"
            else:
                text = f"{lead}, then wait for {phrase}"
            actions = [dict(first),
                       {"intent": "WAIT", "slots": {"duration_amount": amt, "duration_unit": unit}}]
            kind = "compound"
        else:
            p = rng.choice(PEOPLE)
            phrase, (amt, unit) = dur_phrase(rng)
            text = f"wait for {phrase}, then wake {p} up"
            actions = [{"intent": "WAIT", "slots": {"duration_amount": amt, "duration_unit": unit}},
                       {"intent": "WAKEUP", "slots": {"recipient": p}}]
            kind = "compound"
        out.append({"text": text.lower() if rng.random() < 0.3 else text,
                    "actions": actions, "kind": kind})

    with open(dst, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} -> {dst}")


if __name__ == "__main__":
    main()
