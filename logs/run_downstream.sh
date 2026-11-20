#!/usr/bin/env bash
P="/c/Users/Anushka Verma/OneDrive/Desktop/Sem 5/ML/Project"
cd "$P" || exit 1
PY=.venv/Scripts/python
LOG=logs/run_downstream.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }
say "waiting for detector"
until grep -qE "ALL DONE|FAILED" logs/run_detector_block.log; do sleep 60; done
grep -q "ALL DONE" logs/run_detector_block.log || { say "detector failed, stopping"; exit 1; }
say "tracking comparison"
$PY -u -m src.tracking.evaluate > logs/tracking.log 2>&1; say "tracking exit $?"
say "anomaly training"
$PY -u -m src.anomaly.train > logs/anomaly.log 2>&1; say "anomaly exit $?"
say "no-video ablation"
$PY -u -m src.anomaly.train --ablation-no-video > logs/anomaly_ablation.log 2>&1; say "ablation exit $?"
say "report"
$PY -u -m src.anomaly.report > logs/report.log 2>&1; say "report exit $?"
say "robustness"
$PY -u -m src.anomaly.robustness > logs/robustness.log 2>&1; say "robustness exit $?"
say "ALL DONE"
