from __future__ import annotations

import random

VERB_SYNONYMS = {
    "go": ["head", "move", "walk", "navigate", "travel"],
    "clean": ["tidy up", "wipe down", "sweep"],
    "play": ["start playing", "put on"],
    "show": ["display", "put on screen"],
    "get": ["pick up", "grab", "fetch"],
    "give": ["hand over", "pass"],
    "stop": ["halt", "cancel that", "cease"],
    "wait": ["hold on", "pause", "hang on"],
    "wake": ["wake up", "rouse", "get up"],
}

POLITE_PREFIXES = [
    "", "", "", "", "please ", "could you ", "can you ",
    "would you ", "hey ", "hey robot, ", "robot, ",
]

POLITE_SUFFIXES = ["", "", "", "", "", " please", " now", " right now", " will you"]


def politize(text: str, rng: random.Random) -> str:
    text = rng.choice(POLITE_PREFIXES) + text + rng.choice(POLITE_SUFFIXES)
    return text.strip().strip(",").strip()


SYNONYM_REWRITES = [
    ("wake up", ["rouse", "get"]),
    ("hold on for", ["hang on for", "wait for"]),
]


def synonym_variant(text: str, rng: random.Random) -> str:
    t = text
    for canonical, alts in SYNONYM_REWRITES:
        if canonical in t and rng.random() < 0.5:
            t = t.replace(canonical, rng.choice(alts), 1)
            break
    return t


TYPO_RULES = [
    ("the", "teh"), ("and", "adn"), ("kitchen", "kitcken"),
    ("seconds", "secnds"), ("minutes", "minuts"), ("room", "rom"),
    ("please", "plese"), ("what", "waht"), ("your", "you're"),
    ("to", "too"), ("their", "there"), ("here", "hear"),
]


def inject_typo(text: str, rng: random.Random) -> str | None:
    candidates = []
    lowered = text.lower()
    for i, (src, dst) in enumerate(TYPO_RULES):
        idx = lowered.find(" " + src + " ")
        if idx >= 0:
            candidates.append((i, idx))
    if not candidates or rng.random() < 0.35:
        pos = rng.randrange(1, max(2, len(text) - 1))
        if not text[pos].isalpha():
            return None
        action = rng.random()
        if action < 0.5:
            return text[:pos] + text[pos + 1:]
        if action < 0.8:
            return text[:pos] + text[pos] + text[pos:]
        return None
    i, idx = rng.choice(candidates)
    src = TYPO_RULES[i][0]
    start = idx + 1
    return text[:start] + TYPO_RULES[i][1] + text[start + len(src):]
