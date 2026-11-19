# Drone Detection, Tracking and Trajectory Anomaly Detection

Video in, flagged drone trajectories out:

```
Video ─► YOLO11 drone detector ─► ByteTrack ─► trajectory (cx, cy, size, t)
      ─► resample 10 Hz, 3 s windows ─► unit-free kinematic features
      ─► anomaly model (Isolation Forest / One-Class SVM / LSTM autoencoder / ensemble)
      ─► per-window score + reason ─► per-track verdict ─► Streamlit demo
```

## Why the design looks like this

The two suggested datasets do not describe the same thing. VisioDECT is still
images seen from the ground; UAV-SEAD is the drone's own PX4 telemetry. An
anomaly model trained on raw telemetry columns cannot score a pixel track.

The bridge is a **shared, unit-free trajectory representation**. Every source
is reduced to `(t, u, v, scale)`: horizontal position, vertical position (up),
and the drone's size in the same unit. Dividing by size turns positions into
*body lengths*, so a 20 px drone in a video and a 0.5 m airframe in a log file
produce comparable speeds, accelerations and turn rates.

| Source | u, v | scale | Labels |
|---|---|---|---|
| Video tracks (DUT Anti-UAV) | bbox centre in px, y flipped | sqrt(bbox area) | none (real-world check) |
| UAV-SEAD telemetry | local position projected onto a vertical plane at a random camera azimuth | 0.5 m | annotated anomaly time ranges |
| Synthetic | generated in body lengths | 1 | injected anomaly intervals, per type |

Anomaly types in the synthetic set: `zigzag`, `dive`, `speed_burst`,
`reversal`, `circling`, `jump` (tracker / estimator glitch).

### Features (per 3 s window, 10 Hz)

- Sequence channels (for the LSTM autoencoder): speed, vertical velocity,
  tangential acceleration, normal (turning) acceleration, jerk.
- 16 window statistics (for Isolation Forest / OCSVM): speed mean/std/max/p90,
  vertical speed, max tangential and normal acceleration, jerk, turn rate,
  straightness, hover ratio, heading reversals.

All values pass through a sign-preserving log so heavy tails do not dominate.

## Datasets

| Dataset | Used for | Source |
|---|---|---|
| VisioDECT (6 drone models, 3 weather scenarios) | detector training, drone-model classifier | Kaggle mirror of IEEE DataPort, CC BY-NC-SA 4.0 |
| DUT Anti-UAV detection (10k images) | detector training | Hugging Face `aegean-ai/dut-anti-uav`, Apache-2.0 |
| DUT Anti-UAV tracking (20 videos, per-frame GT) | tracking evaluation, real trajectories, demo | same |
| UAV-SEAD (PX4 flight logs) | anomaly training / evaluation | Hugging Face `aykutkabaoglu/uav-flight-anomaly-dataset`, CC BY 4.0 |

UAV-SEAD is ~30 GB in total; the download script takes every annotated
anomalous flight and a random sample of normal flights.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`data/` (~22 GB) is kept outside OneDrive and linked into the project with a
junction, so cloud sync does not touch it:

```powershell
Move-Item data C:\Users\<you>\ML_data\drone_project_data
New-Item -ItemType Junction -Path data -Target C:\Users\<you>\ML_data\drone_project_data
```

## Running everything

```bash
python -m src.data.download --what all --normal 300   # datasets
python -m src.detection.prepare                         # YOLO + crop datasets
python -m src.trajectory.synth                          # synthetic trajectories
python -m src.trajectory.telemetry                      # UAV-SEAD -> windows
python -m src.detection.train --epochs 20 --imgsz 960   # detector + classifier
python -m src.tracking.evaluate                         # tracking metrics + real tracks
python -m src.anomaly.train                             # anomaly models
python -m src.anomaly.report                            # figures on real tracks
streamlit run app/streamlit_app.py                      # demo
```

Single video from the command line:

```bash
python -m src.pipeline --source my_video.mp4 --out outputs/my_video
```

## Project layout

```
configs/      ByteTrack settings for small, fast targets
src/data/     dataset download
src/detection prepare YOLO data, train + evaluate detector and classifier
src/tracking/ detection + tracking, MOT evaluation on DUT videos
src/trajectory/ shared features, synthetic generator, telemetry conversion
src/anomaly/  models, training, evaluation
src/pipeline.py  end-to-end inference and annotated video
app/          Streamlit demo
reports/      metrics (csv/json) and figures
models/       trained weights
```

## Results

All numbers below are on held-out test splits (files in `reports/`).
Hardware: single RTX 5050 Laptop GPU (8 GB).

### Detection (YOLO11s, 20 epochs, imgsz 960)

| Test set | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| All | 0.950 | 0.934 | 0.942 | 0.962 | 0.610 |
| DUT Anti-UAV | 0.953 | 0.882 | 0.916 | 0.928 | 0.582 |
| VisioDECT | 0.972 | 0.967 | 0.970 | 0.990 | 0.632 |

DUT is the harder set: drones are often a few pixels wide against cluttered
backgrounds. Training curves and PR plots are in `models/runs/detector/`.

### Drone-model classifier (VisioDECT crops, 6 classes)

Top-1 accuracy **0.999** on 2,907 test crops. The ImageNet `yolo11n-cls`
weights could not be downloaded, so the classifier (`yolo11s-cls`) was
initialised from the trained drone detector's backbone instead. Crops of the
same drone model come from the same recording sessions, so this score is
optimistic for unseen footage.

### Tracking (ByteTrack on the 20 DUT tracking videos)

| MOTA | IDF1 | ID switches | Precision | Recall | Mostly tracked | Success@0.5 |
|---|---|---|---|---|---|---|
| 0.825 | 0.648 | 94 | 0.916 | 0.913 | 15 / 20 | 0.873 |

IDF1 is lower than MOTA because the drone is lost for a few frames when it
becomes tiny or passes clutter, and a new ID is started afterwards
(516 fragmentations). Per-video numbers: `reports/tracking_metrics.csv`.

### Trajectory anomaly detection

Thresholds are F1-optimal on the validation split; the pipeline model is
chosen by validation AUPRC (Isolation Forest). The supervised gradient
boosting model uses anomaly labels and is only a reference upper bound.

| Test set | Model | AUROC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Synthetic | Isolation Forest | 0.984 | 0.918 | 0.868 | 0.809 | 0.936 |
| Synthetic | One-Class SVM | 0.979 | 0.914 | 0.871 | 0.825 | 0.922 |
| Synthetic | LSTM autoencoder | 0.975 | 0.923 | 0.829 | 0.769 | 0.899 |
| Synthetic | Ensemble (IF + LSTM) | 0.985 | 0.934 | 0.857 | 0.797 | 0.926 |
| Synthetic | *Supervised GBM* | *0.992* | *0.972* | *0.922* | *0.897* | *0.948* |
| UAV-SEAD | Isolation Forest | 0.817 | 0.378 | 0.263 | 0.601 | 0.168 |
| UAV-SEAD | One-Class SVM | 0.430 | 0.216 | 0.242 | 0.729 | 0.145 |
| UAV-SEAD | LSTM autoencoder | 0.845 | 0.430 | 0.198 | 0.822 | 0.113 |
| UAV-SEAD | Ensemble (IF + LSTM) | 0.839 | 0.427 | 0.219 | 0.804 | 0.127 |
| UAV-SEAD | *Supervised GBM* | *0.799* | *0.363* | *0.354* | *0.400* | *0.318* |

With a threshold tuned on UAV-SEAD validation alone, recall on UAV-SEAD rises
to 0.68 (IF) / 0.78 (LSTM-AE) at F1 0.37 / 0.40 (`*_src_thr` columns of
`reports/anomaly_metrics.csv`).

Isolation Forest detection rate per anomaly type:

| Synthetic type | Rate | | UAV-SEAD class | Rate |
|---|---|---|---|---|
| dive | 0.99 | | External Position | 0.23 |
| zigzag | 0.98 | | Global Position | 0.17 |
| circling | 0.96 | | Mechanical | 0.03 |
| jump | 0.94 | | Altitude | 0.03 |
| speed_burst | 0.92 | | normal (false-positive rate) | 0.011 |
| reversal | 0.81 | | | |
| normal (false-positive rate) | 0.027 | | | |

Reading the numbers: anomalies that change *how the drone moves* (synthetic
manoeuvres, position-estimator faults) are found well from the trajectory
alone. Mechanical and altitude faults in UAV-SEAD barely change the flight
path during the labelled interval, so a camera-only view cannot see them;
this is a property of the data, not of the threshold.

On the 50 real DUT video tracks (no anomaly labels) the Isolation Forest
flags 32 % of windows, the LSTM autoencoder 19 %. Real pixel tracks carry
detector jitter and camera motion that neither training source has, so these
flags should be read as "unusual motion" rather than confirmed anomalies
(`reports/figures/real_vs_synthetic_scores.png`).

Figures: `reports/figures/` (ROC / PR curves, score histograms, LSTM loss,
example trajectories).

## Limitations

- No bird / non-drone class: neither detection dataset contains labelled
  birds, so bird false positives are not measured.
- Video trajectories are in image coordinates. Camera motion is not
  compensated; DUT videos are mostly from a static or slowly moving camera.
- Pixel-to-body-length normalisation assumes the box size tracks the drone's
  true size; very small boxes (< 10 px) make speed estimates noisy.
- Real video tracks have no anomaly labels; they are used to check the false
  alarm rate on (presumably normal) flights, not to measure recall.
- UAV-SEAD anomalies are state-estimation faults. Only the part of a fault
  that shows up in the estimated position is visible to these features.
