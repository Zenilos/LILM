from __future__ import annotations

import random

from schema.dsl import Action

LOCATIONS = [
    "the kitchen", "kitchen", "my room", "the living room", "living room",
    "the garage", "garage", "here", "the hallway", "hallway",
    "the bedroom", "bedroom", "where I cook", "my side",
]

OBJECTS = [
    "the cup", "cup", "the red ball", "red ball", "the plant",
    "the box", "box", "the remote", "my phone", "the towel",
    "the trash", "trash", "the toy", "toy", "the keys", "keys",
    "the plate", "plate", "bottle", "the bottle",
]

RECIPIENTS = [
    "John", "my wife", "the kids", "my daughter", "Sara", "my son",
    "mom", "dad", "Alex", "grandma",
]

PEOPLE = RECIPIENTS

FILES = [
    "song.mp3", "alarm.wav", "wake.mp3", "ring.wav", "beep.wav",
    "music.mp3", "tone.wav", "chime.mp3", "alert.wav", "podcast.mp3",
]

DURATIONS = [
    ("5", "seconds"), ("5", "seconds"), ("10", "seconds"),
    ("30", "seconds"), ("2", "minutes"), ("2", "minutes"),
    ("1", "minutes"), ("15", "minutes"), ("30", "minutes"),
    ("45", "seconds"), ("3", "seconds"), ("1", "hours"),
]

DURATION_PHRASES = {
    "five seconds": ("5", "seconds"),
    "two minutes": ("2", "minutes"),
    "ten seconds": ("10", "seconds"),
    "half an hour": ("30", "minutes"),
    "a minute": ("1", "minutes"),
    "a couple minutes": ("2", "minutes"),
    "an hour": ("1", "hours"),
    "one minute": ("1", "minutes"),
    "thirty seconds": ("30", "seconds"),
    "quarter of an hour": ("15", "minutes"),
}

MESSAGES = [
    "hello", "dinner is ready", "time for bed", "welcome home",
    "good morning", "meeting in 5", "I am here", "battery low",
]


def move() -> tuple[str, list[Action]]:
    loc = random.choice(LOCATIONS)
    return (random.choice([
        f"go to {loc}",
        f"head to {loc}",
        f"move to {loc}",
        f"go {loc}",
        f"navigate to {loc}",
        f"walk to {loc}",
        f"come to {loc}" if loc != "here" else "come here",
        f"make your way to {loc}",
    ]), [Action("MOVE", {"location": loc})])


def clean() -> tuple[str, list[Action]]:
    if random.random() < 0.6:
        loc = random.choice(LOCATIONS)
        return (random.choice([
            f"clean {loc}",
            f"clean up {loc}",
            f"vacuum {loc}",
            f"mop {loc}",
            f"please clean {loc}",
        ]), [Action("CLEAN", {"location": loc})])
    return (random.choice(["clean", "start cleaning", "vacuum", "do the cleaning"]),
            [Action("CLEAN", {})])


def play() -> tuple[str, list[Action]]:
    f = random.choice(FILES)
    return (random.choice([
        f"play {f}",
        f"play the file {f}",
        f"put on {f}",
        f"play song {f}",
    ]), [Action("PLAY", {"file": f})])


def show() -> tuple[str, list[Action]]:
    m = random.choice(MESSAGES)
    if random.random() < 0.4:
        p = random.choice(PEOPLE)
        return (random.choice([
            f"show {m} to {p}",
            f"display {m} for {p}",
        ]), [Action("SHOW", {"message": m, "person": p})])
    return (random.choice([
        f"show {m}",
        f"display {m}",
        f"show message {m}",
    ]), [Action("SHOW", {"message": m})])


def get() -> tuple[str, list[Action]]:
    o = random.choice(OBJECTS)
    return (random.choice([
        f"get {o}",
        f"pick up {o}",
        f"grab {o}",
        f"bring me {o}",
        f"fetch {o}",
    ]), [Action("GET", {"object": o})])


def give() -> tuple[str, list[Action]]:
    o = random.choice(OBJECTS)
    r = random.choice(RECIPIENTS)
    return (random.choice([
        f"give {r} {o}",
        f"give {o} to {r}",
        f"hand {o} to {r}",
        f"hand {r} {o}",
        f"take {o} to {r}",
    ]), [Action("GIVE", {"object": o, "recipient": r})])


def stop() -> tuple[str, list[Action]]:
    return (random.choice([
        "stop", "halt", "stop what you're doing", "cancel",
        "stop now", "quit it", "knock it off", "abort",
    ]), [Action("STOP", {})])


def wait() -> tuple[str, list[Action]]:
    if random.random() < 0.35:
        phrase = random.choice(list(DURATION_PHRASES))
        amount, unit = DURATION_PHRASES[phrase]
        template = random.choice([
            f"wait {phrase}",
            f"wait for {phrase}",
            f"hold on for {phrase}",
            f"pause for {phrase}",
        ])
    else:
        amount, unit = random.choice(DURATIONS)
        template = random.choice([
            f"wait {amount} {unit}",
            f"wait for {amount} {unit}",
            f"hold on for {amount} {unit}",
            f"pause for {amount} {unit}",
            f"hold on {amount} {unit}",
        ])
    return (template, [Action("WAIT", {"duration_amount": amount, "duration_unit": unit})])


def wakeup() -> tuple[str, list[Action]]:
    r = random.choice(RECIPIENTS)
    return (random.choice([
        f"wake up {r}",
        f"wake {r}",
        f"go wake up {r}",
        f"go and wake {r}",
        f"wake {r} up",
    ]), [Action("WAKEUP", {"recipient": r})])


GENERATORS = {
    move: 5, clean: 4, play: 3, show: 4, get: 4,
    give: 4, stop: 2, wait: 4, wakeup: 4,
}

OFF_TOPIC = [
    "what's the weather like", "tell me a joke", "who won the game",
    "what time is it", "how are you", "sing a song for me",
    "what's your name", "do you love me", "hello there",
    "what can you do", "tell me a story", "what day is it today",
]
