#!/usr/bin/env bash
P="/c/Users/Anushka Verma/OneDrive/Desktop/Sem 5/ML/Project"
cd "$P" || exit 1
PY=.venv/Scripts/python
LOG=logs/run_detector_block.log
RUN=models/runs/detector_block
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }
installed() { [ "$(cat models/detector/installed_from.txt 2>/dev/null)" = "detector_block" ]; }
epochs_done() { [ -f $RUN/results.csv ] && echo $(( $(wc -l < $RUN/results.csv) - 1 )) || echo 0; }
say "start"
for attempt in $(seq 1 12); do
  installed && break
  if [ "$(epochs_done)" -ge 20 ] && [ -f $RUN/weights/best.pt ]; then
    say "all epochs done, installing weights"
    $PY -c "from src.detection.train import install_detector; install_detector('detector_block')"
    break
  fi
  if [ -f $RUN/weights/last.pt ]; then FLAG=--resume; else FLAG=; fi
  say "attempt $attempt $FLAG (epochs done: $(epochs_done))"
  $PY -u -m src.detection.train --data block --name detector_block --epochs 20 --imgsz 960 --workers 6 $FLAG > "logs/detector_block_$attempt.log" 2>&1
  say "exit $?"
  sleep 20
done
installed || { say "FAILED"; exit 1; }
[ -f reports/detection_metrics_block.json ] || { say "test evaluation"; $PY -u -m src.detection.train --data block --eval-only > logs/detector_block_eval.log 2>&1; say "eval exit $?"; }
say "ALL DONE"
