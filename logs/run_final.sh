#!/usr/bin/env bash
P="/c/Users/Anushka Verma/OneDrive/Desktop/Sem 5/ML/Project"
cd "$P" || exit 1
PY=.venv/Scripts/python
LOG=logs/run_final.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }
say "waiting for run_remaining"
until grep -qE "ALL DONE|FAILED" logs/run_remaining.log; do sleep 30; done
say "tracking evaluation"
$PY -u -m src.tracking.evaluate > logs/tracking.log 2>&1; say "tracking exit $?"
say "classifier"
$PY -u -m src.detection.train --classifier-only --cls-epochs 20 --workers 6 > logs/classifier.log 2>&1; say "classifier exit $?"
say "report"
$PY -u -m src.anomaly.report > logs/report.log 2>&1; say "report exit $?"
say "ALL DONE"
