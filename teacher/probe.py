from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import needle  # noqa: E402

SINGLE_TOOL_SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema/tool_schema.json").read_text())


def build_nine_tools():
    from typing import Annotated

    @needle.tool
    def move(location: Annotated[str, needle.Field(description="place: kitchen, my room, here, garage")] = ""):
        """Move the robot to a location."""
        return {"intent": "MOVE", "location": location}

    @needle.tool
    def clean(location: str = ""):
        """Clean or vacuum a place."""
        return {"intent": "CLEAN", "location": location}

    @needle.tool
    def play(file: Annotated[str, needle.Field(description="sdcard filename like song.mp3")]):
        """Play an audio file from the sdcard."""
        return {"intent": "PLAY", "file": file}

    @needle.tool
    def show(message: str, person: str = ""):
        """Display a message, optionally for a person."""
        return {"intent": "SHOW", "message": message}

    @needle.tool
    def get(object: Annotated[str, needle.Field(description="physical object: cup, red ball")]):
        """Pick up an object."""
        return {"intent": "GET", "object": object}

    @needle.tool
    def give(object: str, recipient: Annotated[str, needle.Field(description="person: John, my wife")] = ""):
        """Hand an object to someone."""
        return {"intent": "GIVE", "object": object}

    @needle.tool
    def stop():
        """Halt all current actions."""
        return {"intent": "STOP"}

    @needle.tool
    def wait(duration: Annotated[str, needle.Field(description="as spoken: 5 seconds, half an hour")]):
        """Pause for a duration."""
        return {"intent": "WAIT", "duration": duration}

    @needle.tool
    def wakeup(recipient: Annotated[str, needle.Field(description="person to wake: John, my daughter")]):
        """Go wake up a person."""
        return {"intent": "WAKEUP", "recipient": recipient}

    return [move, clean, play, show, get, give, stop, wait, wakeup]


QUERIES = [
    "go to my room",
    "go to my room and go to oven",
    "go to my room and wait there for 5 minutes and then go to oven",
    "wake up my daughter",
    "give John the cup",
    "what's the weather",
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"

    if mode == "single":
        agent = needle.Needle(tools=SINGLE_TOOL_SCHEMA,
                              system="device: domestic robot; locale: en-US")
    else:
        agent = needle.Needle(tools=build_nine_tools(),
                              system="device: domestic robot; locale: en-US")

    for q in QUERIES:
        resp = agent.complete(q)
        calls = resp.get("function_calls")
        print(f"Q: {q!r}")
        print(f"   type={resp.get('type')} conf={resp.get('confidence')}")
        print(f"   calls={json.dumps(calls)}")
        if resp.get("reasoning"):
            print(f"   reasoning={resp.get('reasoning')}")
        agent.reset()


if __name__ == "__main__":
    main()
