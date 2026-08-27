"""Extended compound generator: depth 3–5 chains with coreference and distractors.
Produces (query, actions, meta) tuples. Plug into build_dataset or standalone."""
from __future__ import annotations
import random
from generator.canonical import (
    GENERATORS, PEOPLE, LOCATIONS, RECIPIENTS, FILES, DURATIONS, DURATION_PHRASES,
    MESSAGES
)
from generator.noise import inject_typo, politize, synonym_variant
from schema.dsl import Action

CHAIN_JOINERS = [
    " and ", " and then ", " then ", ", then ", ", and ",
    " after that, ", " afterwards, ", ", afterwards ",
]

DISTRACTORS = [
    "", "", "", "", "",          # most queries: no distractor
    " please", " also",          # 2-word phrases
    " and don't forget", " don't forget to",
    " once you are done",
]

PERSON_PRONOUNS = {
    "John": "him", "Sara": "her", "Emma": "her", "Lina": "her",
    "grandma": "her", "grandpa": "him", "mom": "her", "dad": "him",
    "Alex": "him", "Omar": "him", "grandma": "her", "grandpa": "him",
    "my wife": "her", "my husband": "him", "my son": "him",
    "my daughter": "her", "my brother": "him", "my sister": "her",
    "the kids": "them", "the guests": "them",
}

def _coref(recipient: str) -> str:
    """Return a pronoun or keep the name if unknown."""
    return PERSON_PRONOUNS.get(recipient, recipient)

def _location_coref(loc: str) -> str:
    """Pronoun only for known person-locations."""
    return PERSON_PRONOUNS.get(loc, loc)

def make_deep_chain(rng: random.Random, depth: int = None) -> tuple[str, list[Action], dict]:
    """Build a chain of 3–5 actions with coreference and distractors."""
    if depth is None:
        depth = rng.choice([3, 3, 4, 4])

    gens = list(GENERATORS.keys())
    actions: list[Action] = []
    phrases: list[str] = []
    last_recipient: str | None = None
    last_location: str | None = None

    for i in range(depth):
        gen = rng.choice(gens)
        text, acts = gen()
        act = acts[0]  # atomic always yields one
        actions.append(act)

        # Coreference in later clauses: replace explicit slot with pronoun
        if i > 0 and rng.random() < 0.45:
            if act.intent in ("GIVE", "WAKEUP") and "recipient" in act.slots:
                if rng.random() < 0.4:
                    # Use pronoun in query text, keep explicit in gold
                    for name, pron in PERSON_PRONOUNS.items():
                        if name in text.lower():
                            text = text.replace(name, pron, 1)
                            break
            elif act.intent in ("MOVE", "CLEAN") and "location" in act.slots:
                if last_recipient and rng.random() < 0.35:
                    pron = _coref(last_recipient)
                    if pron != last_recipient:
                        # "go to her room" — keep explicit in gold
                        for name, p in PERSON_PRONOUNS.items():
                            if name in text.lower():
                                text = text.replace(name, p, 1)
                                break

        # Track state for coreference
        if "recipient" in act.slots:
            last_recipient = act.slots["recipient"]
        if "location" in act.slots:
            last_location = act.slots["location"]

        # Distractor in later clauses
        distractor = ""
        if i > 0 and rng.random() < 0.25:
            distractor = rng.choice(DISTRACTORS)

        phrases.append(text + distractor)

    # Join with varied joiners
    parts = [phrases[0]]
    for p in phrases[1:]:
        joiner = rng.choice(CHAIN_JOINERS)
        parts.append(joiner + p)

    full_text = "".join(parts)
    return full_text, actions, {"depth": depth, "kind": "deep-chain"}

def make_deep_compound(rng: random.Random) -> tuple[str, list[Action], dict]:
    """Top-level entry: random depth 3–5 chain with corruption."""
    text, actions, meta = make_deep_chain(rng)
    r = rng.random()
    if r < 0.45:
        text = politize(text, rng)
    if r > 0.55:
        text = synonym_variant(text, rng)
    if 0.25 < r < 0.5:
        text = inject_typo(text, rng) or text
    return text, actions, meta

if __name__ == "__main__":
    import json
    rng = random.Random(999)
    for _ in range(10):
        text, actions, meta = make_deep_compound(rng)
        print(f"[depth={meta['depth']}] {text!r}")
        print(f"  -> {[a.to_dict() for a in actions]}")
