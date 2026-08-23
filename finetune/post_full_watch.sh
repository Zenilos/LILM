#!/bin/bash
# Waits for full-FT (pid $1) to finish, then evaluates robot.cact and logs a verdict.
PID=${1:?pid}
LOG=/tmp/train_full.log
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== training exited $(date '+%H:%M') ===" >> "$LOG"
if [ ! -f checkpoints/full_v1.pkl ]; then echo "TRAIN FAILED - no checkpoint" >> "$LOG"; exit 1; fi
CACT=robot.cact EVAL_N=50 ./finetune/eval_mac.sh >> "$LOG" 2>&1
ACC=$(grep -o 'FINAL RESULT: [0-9]*/50 = [0-9.]*%' "$LOG" | tail -1)
echo "VERDICT: $ACC" >> "$LOG"
N=$(echo "$ACC" | grep -o '[0-9]*' | head -1)
if [ "${N:-0}" -ge 45 ]; then echo "DECISION: >=90% gate reached -> proceed to ESP32 size work" >> "$LOG"
elif [ "${N:-0}" -ge 35 ]; then echo "DECISION: relaunch with --qat 4 for noise-robust weights" >> "$LOG"
else echo "DECISION: margins still thin -> revisit schema/token budget" >> "$LOG"; fi
