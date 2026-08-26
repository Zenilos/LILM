from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.canonical import GENERATORS, OFF_TOPIC, give, move  # noqa: E402
from generator.deep_chains import make_deep_compound              # noqa: E402
from generator.noise import inject_typo, politize, synonym_variant  # noqa: E402
from schema.dsl import Action, actions_match  # noqa: E402

COMPOUND_JOINERS = [" and ", " and then ", " then ", ", then ", ", and "]

UNAVAILABLE = [
    "go wash yourself", "make me a coffee", "fly to the moon",
    "dance for me", "tell me a joke", "what's the weather",
    "sing a song", "cook dinner", "water the plants",
    "do my homework", "pet the dog", "jump three times",
]


def make_compound(rng: random.Random):
    gens = list(GENERATORS)
    a_gen, b_gen = rng.sample(gens, 2)
    ta, aa = a_gen()
    tb, ab = b_gen()
    joiner = rng.choice(COMPOUND_JOINERS)
    text = ta + joiner + tb
    actions = aa + ab
    return text, actions, [len(ta), len(tb)]


def corrupt(text: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:
        text = politize(text, rng)
    if r > 0.55:
        text = synonym_variant(text, rng)
    if 0.25 < r < 0.5:
        t = inject_typo(text, rng)
        text = t or text
    return text


def generate(n_atomic: int = 3000, n_compound: int = 800, n_offtopic: int = 300,
             n_deep: int = 600, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[str] = set()

    def add(text: str, actions: list[Action], meta: dict):
        key = " ".join(text.lower().split())
        if key in seen:
            return
        seen.add(key)
        records.append({"text": text, "actions": [a.to_dict() for a in actions], **meta})

    for _ in range(n_atomic):
        gen = rng.choice(list(GENERATORS.keys()))
        text, actions = gen()
        text = corrupt(text, rng)
        add(text, actions, {"kind": "atomic"})

    for _ in range(n_compound):
        text, actions, bounds = make_compound(rng)
        text = corrupt(text, rng)
        add(text, actions, {"kind": "compound", "clause_lengths": bounds})

    for _ in range(n_deep):
        text, actions, meta = make_deep_compound(rng)
        add(text, actions, meta)

    unavailable = OFF_TOPIC + UNAVAILABLE
    for t in unavailable:
        for variant in (t, politize(t, rng)):
            add(variant, [Action("UNAVAILABLE", {})], {"kind": "unavailable"})

    for _ in range(n_offtopic - 2 * len(unavailable)):
        gen = rng.choice([give, move])
        text, actions = gen()
        words = text.split()
        rng.shuffle(words)
        add(" ".join(words), [Action("UNAVAILABLE", {})], {"kind": "unavailable_scrambled"})

    bad = [r for r in records if not actions_match(
        [Action.from_dict(a) for a in r["actions"]],
        [Action.from_dict(a) for a in r["actions"]])]
    assert not bad
    return records


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/generated/v1.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    recs = generate()
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    lens = sorted(len(r["text"]) for r in recs)
    print(f"wrote {len(recs)} records -> {out_path}")
    print(f"kinds: {kinds}")
    print(f"text len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")
