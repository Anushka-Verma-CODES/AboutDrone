#!/usr/bin/env bash
P="/c/Users/Anushka Verma/OneDrive/Desktop/Sem 5/ML/Project"
cd "$P" || exit 1
PY=.venv/Scripts/python
LOG=logs/run_improve.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }
say "start"
say "anomaly training"
$PY -u -m src.anomaly.train > logs/anomaly.log 2>&1; say "anomaly exit $?"
say "report"
$PY -u -m src.anomaly.report > logs/report.log 2>&1; say "report exit $?"
say "classifier evaluation"
$PY -u -m src.detection.classifier_eval > logs/classifier_eval.log 2>&1; say "classifier exit $?"
say "ALL DONE"
