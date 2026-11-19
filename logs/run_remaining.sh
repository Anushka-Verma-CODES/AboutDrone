#!/usr/bin/env bash
# Finish every remaining training / evaluation step. Runs detached from the
# chat session; progress goes to logs/run_remaining.log.
P="/c/Users/Anushka Verma/OneDrive/Desktop/Sem 5/ML/Project"
cd "$P" || exit 1
PY=.venv/Scripts/python
LOG=logs/run_remaining.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

say "start"
for attempt in $(seq 1 12); do
  [ -f models/detector/best.pt ] && break
  say "detector resume attempt $attempt"
  $PY -u -m src.detection.train --resume --imgsz 960 --workers 6 --cls-epochs 20 > "logs/detector_r$attempt.log" 2>&1
  say "detector process exited with $?"
  sleep 20
done
[ -f models/detector/best.pt ] || { say "FAILED: detector never finished"; exit 1; }

[ -f reports/detection_metrics.json ] || { say "detection eval"; $PY -u -m src.detection.train --eval-only --imgsz 960 > logs/detector_eval.log 2>&1; say "eval exit $?"; }
[ -f reports/classifier_metrics.json ] || { say "classifier"; $PY -u -m src.detection.train --classifier-only --cls-epochs 20 --workers 6 > logs/classifier.log 2>&1; say "classifier exit $?"; }

say "tracking evaluation"
$PY -u -m src.tracking.evaluate > logs/tracking.log 2>&1; say "tracking exit $?"

say "anomaly training"
$PY -u -m src.anomaly.train > logs/anomaly.log 2>&1; say "anomaly exit $?"

say "report"
$PY -u -m src.anomaly.report > logs/report.log 2>&1; say "report exit $?"
say "ALL DONE"
