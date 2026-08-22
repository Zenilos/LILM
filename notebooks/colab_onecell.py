# ============================================================
# Needle 2 robot-DSL fine-tune — ONE-CELL Colab script
# Runtime > Change runtime type > T4 GPU, then paste & run.
# Watch losses: must start ~2-4 and DROP. 0.0000 = broken, stop.
# ============================================================
import json, subprocess, sys, time

REPO = "https://github.com/Zenilos/LILM.git"
DATA = "data/finetune/train_v2.jsonl"
SCHEMA = "schema/tool_schema.json"

def sh(cmd):
    print(f"\n$ {cmd}\n" + "-" * 60)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED: {cmd}")
    print("-" * 60)

# ---- 1. clone ----
sh(f"test -d LILM || git clone {REPO}")
get_ipython().run_line_magic("cd", "LILM")

# ---- 2. install ----
sh('pip install -q "cactus-needle[gpu]"')
import jax
print("jax devices:", jax.devices())

# ---- 3. fine-tune (streams live output) ----
t0 = time.time()
sh(f"needle finetune {DATA} --epochs 10 --val-split 0.1 "
   f"--out checkpoints/needle_lora_v2.pkl")
print(f"train wall time: {(time.time()-t0)/60:.1f} min")

# ---- 4. build deployable model ----
sh("needle build checkpoints/needle2.pkl --lora checkpoints/needle_lora_v2.pkl "
   "--bits 2 --out robot.cact")
sh("ls -la robot.cact")

# ---- 5. empirical checks ----
import needle
schema = json.load(open(SCHEMA))
agent = needle.Needle(weights="robot.cact", tools=schema,
                      system="device: domestic robot; locale: en-US")
tests = [
    "go to my room",
    "go to my room and wait there for 5 minutes and then go to oven",
    "wait for five seconds",
    "give John the cup",
    "wake up my daughter",
    "go wash yourself",
    "play alarm.wav then stop",
    "clean the kitchen then show dinner is ready",
]
print("\n=== spot checks ===")
for q in tests:
    r = agent.complete(q)
    print(repr(q), "->", json.dumps(r["function_calls"]))

print("\n=== exact match on first 100 dataset examples ===")
records = [json.loads(l) for l in open(DATA)][:100]
ok = 0
for rec in records:
    pred = agent.complete(rec["query"])["function_calls"] or []
    gold = rec["answers"]
    if len(pred) == len(gold) and all(
            p["name"] == g["name"] and p["arguments"].items() >= g["arguments"].items()
            and set(p["arguments"]) - {"intent"} <= set(g["arguments"])
            for p, g in zip(pred, gold)):
        ok += 1
print(f"exact match: {ok}/100")

# ---- 6. download artifacts ----
from google.colab import files
files.download("robot.cact")
files.download("checkpoints/needle_lora_v2.pkl")
print("\nDone — bring robot.cact back to the Mac.")
