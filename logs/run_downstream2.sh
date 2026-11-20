#!/usr/bin/env bash
P="/c/Users/Anushka Verma/OneDrive/Desktop/Sem 5/ML/Project"
cd "$P" || exit 1
PY=.venv/Scripts/python
LOG=logs/run_downstream2.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }
say "rerun after track filter"
say "tracking comparison"
$PY -u -m src.tracking.evaluate --reuse > logs/tracking.log 2>&1; say "tracking exit $?"
say "anomaly training"
$PY -u -m src.anomaly.train > logs/anomaly.log 2>&1; say "anomaly exit $?"
say "no-video ablation"
$PY -u -m src.anomaly.train --ablation-no-video > logs/anomaly_ablation.log 2>&1; say "ablation exit $?"
say "report"
$PY -u -m src.anomaly.report > logs/report.log 2>&1; say "report exit $?"
say "robustness"
$PY -u -m src.anomaly.robustness > logs/robustness.log 2>&1; say "robustness exit $?"
say "ALL DONE"
