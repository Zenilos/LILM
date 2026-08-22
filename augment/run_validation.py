from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augment.validate import FREE_MODELS, check_example  # noqa: E402

MODEL = sys.argv[2] if len(sys.argv) > 2 else None
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/generated/v1.jsonl"
    records = [json.loads(l) for l in open(src) if l.strip()]
    model = MODEL or FREE_MODELS[0]
    sample = random.Random(11).sample(records, min(N, len(records)))

    kept, rejected, errors = [], [], []
    t0 = time.time()
    for i, rec in enumerate(sample):
        try:
            ok = check_example(rec["text"], rec["actions"], model)
        except Exception as e:
            errors.append((rec, str(e)))
            continue
        (kept if ok else rejected).append(rec)
        if (i + 1) % 20 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[{i+1}/{len(sample)}] keep={len(kept)} reject={len(rejected)} "
                  f"err={len(errors)} ({rate:.1f}/s)", flush=True)

    print(f"\nmodel={model}")
    print(f"kept={len(kept)} rejected={len(rejected)} errors={len(errors)}")
    print(f"acceptance={len(kept)/(len(kept)+len(rejected)):.1%}")
    print("\n--- rejected samples ---")
    for r in rejected[:10]:
        print(f"{r['text']!r} -> {json.dumps(r['actions'])}")


if __name__ == "__main__":
    main()
