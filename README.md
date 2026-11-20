# Drone Detection, Tracking and Trajectory Anomaly Detection

Video in, flagged drone trajectories out:

```
Video ─► YOLO11 drone detector ─► ByteTrack ─► trajectory (cx, cy, size, t)
      ─► Kalman / RTS smoothing ─► resample 10 Hz, 3 s windows ─► unit-free kinematic features
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
| Video tracks (DUT Anti-UAV, 20 videos split 12/4/4 by video) | bbox centre in px, y flipped, Kalman-smoothed | sqrt(bbox area) | train: assumed normal; val/test: real tracks + anomalies injected into copies of them |
| UAV-SEAD telemetry | local position projected onto a vertical plane at a random camera azimuth | 0.5 m | annotated anomaly time ranges |
| Synthetic | generated in body lengths | 1 | injected anomaly intervals, per type |

Anomaly types in the synthetic set: `zigzag`, `dive`, `speed_burst`,
`reversal`, `circling`, `jump` (tracker / estimator glitch).

**Real video in training.** The unsupervised models are fitted on normal
windows from all three sources. The 12 training videos give only ~3k windows,
so they are repeated to make up ~10 % of the normal training data. This
teaches the models what real detector tracks look like (residual jitter,
size changes, camera motion).

**Labelled video test set.** The DUT videos have no anomaly labels, so the
six anomaly types are injected into copies of the real val/test tracks: the
anomaly's displacement is added on top of the real motion, keeping the
detector noise. Untouched tracks stay as normal examples.

**Kalman smoothing.** Detector boxes jitter by a few pixels per frame, which
becomes fake acceleration and jerk after differentiation. Each pixel track is
smoothed with a constant-velocity Kalman filter + RTS smoother in
body-length units (`src/tracking/smooth.py`) before features are computed.

**Operating points.** Each source gets its own threshold, picked on its own
validation set: F1-optimal for synthetic and telemetry, and a 5 % false-alarm
budget on real tracks for video (the anomaly share of the injected set is
arbitrary, so F1 there would be meaningless). The video pipeline uses the
video threshold (`--threshold-source` to change it).

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
python -m src.detection.train --epochs 20 --imgsz 960 --skip-classifier   # detector
python -m src.detection.classifier_eval                 # classifier, leakage-free splits
python -m src.tracking.evaluate                         # tracking metrics + video trajectories
python -m src.anomaly.train                             # anomaly models
python -m src.anomaly.train --ablation-no-video         # optional: without real video in training
python -m src.anomaly.report                            # figures, held-out video check
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
src/tracking/ detection + tracking, Kalman smoothing, MOT evaluation on DUT videos
src/trajectory/ shared features, synthetic generator, telemetry and video conversion
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

### Drone-model classifier (VisioDECT crops, 6 classes, YOLO11s-cls, ImageNet init)

VisioDECT images are consecutive video frames (1,200 numbered frames per
drone model and weather), so a random split puts near-identical frames in
train and test. Three protocols, same model and settings
(`src/detection/classifier_eval.py`):

| Split | Test crops | Top-1 |
|---|---|---|
| Random image split (leaky) | 2,907 | 0.999 |
| Frame blocks: 100-frame blocks, 8/2/2 per sequence, 5-frame gap at block edges | 2,990 | **0.988** |
| Cross-weather: train on 2 scenarios, test on the 3rd (mean of 3 folds) | 5,566-6,940 per fold | **0.794** |
| - test cloudy / evening / sunny | | 0.703 / 0.844 / 0.835 |

Within the same recording conditions the classifier is reliable (0.988).
On unseen weather it drops to ~0.79, so lighting and background still carry
a lot of the signal. The frame-block model is the one saved in
`models/classifier/best.pt`.

### Tracking (ByteTrack on the 20 DUT tracking videos)

| MOTA | IDF1 | ID switches | Precision | Recall | Mostly tracked | Success@0.5 |
|---|---|---|---|---|---|---|
| 0.825 | 0.648 | 94 | 0.916 | 0.913 | 15 / 20 | 0.873 |

IDF1 is lower than MOTA because the drone is lost for a few frames when it
becomes tiny or passes clutter, and a new ID is started afterwards
(516 fragmentations). Per-video numbers: `reports/tracking_metrics.csv`.

### Trajectory anomaly detection

Each source uses its own operating point picked on its validation set (see
above). The pipeline model is chosen by the mean of the per-source validation
AUPRCs: **ensemble (IF + LSTM autoencoder)**. The supervised gradient boosting
model sees anomaly labels and is only a reference.

| Test set | Model | AUROC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Synthetic | Isolation Forest | 0.982 | 0.909 | 0.871 | 0.854 | 0.889 |
| Synthetic | One-Class SVM | 0.972 | 0.896 | 0.860 | 0.854 | 0.865 |
| Synthetic | LSTM autoencoder | 0.975 | 0.926 | 0.876 | 0.916 | 0.840 |
| Synthetic | **Ensemble** | 0.984 | 0.933 | 0.885 | 0.892 | 0.878 |
| Synthetic | *Supervised GBM* | *0.992* | *0.971* | *0.927* | *0.945* | *0.909* |
| UAV-SEAD | Isolation Forest | 0.806 | 0.367 | 0.363 | 0.259 | 0.608 |
| UAV-SEAD | One-Class SVM | 0.398 | 0.195 | 0.166 | 0.091 | 1.000 |
| UAV-SEAD | LSTM autoencoder | 0.846 | 0.426 | 0.406 | 0.276 | 0.763 |
| UAV-SEAD | **Ensemble** | 0.836 | 0.425 | 0.389 | 0.273 | 0.680 |
| UAV-SEAD | *Supervised GBM* | *0.780* | *0.341* | *0.325* | *0.231* | *0.548* |
| DUT video (held-out) | Isolation Forest | 0.980 | 0.984 | 0.925 | 0.909 | 0.941 |
| DUT video (held-out) | One-Class SVM | 0.966 | 0.973 | 0.910 | 0.921 | 0.900 |
| DUT video (held-out) | LSTM autoencoder | 0.969 | 0.976 | 0.919 | 0.936 | 0.902 |
| DUT video (held-out) | **Ensemble** | 0.977 | 0.982 | 0.930 | 0.954 | 0.906 |
| DUT video (held-out) | *Supervised GBM* | *0.965* | *0.978* | *0.915* | *0.919* | *0.911* |

Notes:
- UAV-SEAD with its own threshold: the ensemble finds 68 % of anomalous
  windows (12 % with the single global threshold used before), at the cost
  of an 18 % false-alarm rate on normal flights. The OCSVM collapses to
  "flag everything" there and should not be used for telemetry.
- The video test set is ~52 % injected anomalies, which inflates AUPRC and
  F1. The meaningful video numbers are: **4.8 % false alarms on real tracks
  and 91 % recall on injected anomalies** (ensemble, 4 unseen videos).

Detection rate per anomaly type (ensemble, per-source threshold):

| Synthetic | Rate | UAV-SEAD | Rate | DUT video (injected) | Rate |
|---|---|---|---|---|---|
| zigzag | 0.94 | External Position | 0.84 | circling | 1.00 |
| dive | 0.93 | Global Position | 0.52 | zigzag | 0.99 |
| jump | 0.91 | Mechanical | 0.42 | dive | 0.97 |
| circling | 0.87 | Altitude | 0.32 | speed_burst | 0.94 |
| speed_burst | 0.82 | | | jump | 0.93 |
| reversal | 0.76 | | | reversal | 0.59 |
| normal (false alarms) | 0.013 | normal (false alarms) | 0.18 | normal (false alarms) | 0.048 |

#### What the improvements changed (held-out DUT test videos, ensemble)

| Setting | AUROC | False alarms on real tracks | Recall (injected) |
|---|---|---|---|
| No smoothing (same models, same threshold) | 0.963 | 23.3 % | 0.95 |
| Smoothing, no real video in training | 0.964 | 8.0 % | 0.88 |
| **Smoothing + real video in training** | **0.977** | **4.8 %** | **0.91** |

Before these changes the Isolation Forest flagged 32 % of all windows on the
DUT videos. Smoothing removes most of the detector-jitter false alarms;
adding real video tracks to training lifts the ranking quality on video
(AUROC 0.964 to 0.977) while synthetic and telemetry AUROC change by at most
0.01 for IF, LSTM-AE and the ensemble (0.03 for the OCSVM)
(`reports/anomaly_metrics_no_video_train.csv`,
`reports/real_video_flag_rate.json`).

Reading the numbers: anomalies that change *how the drone moves* (synthetic
manoeuvres, injected video anomalies, position-estimator faults) are found
well from the trajectory alone. Mechanical and altitude faults in UAV-SEAD
barely change the flight path, so a camera-only view finds them only at a
high false-alarm cost; this is a property of the data, not of the threshold.
Slow `reversal` manoeuvres are the hardest motion anomaly.

Figures: `reports/figures/` (ROC / PR curves for every test set, score
histograms, LSTM loss, example trajectories, real vs synthetic scores).

## Limitations

- No bird / non-drone class: neither detection dataset contains labelled
  birds, so bird false positives are not measured.
- Video trajectories are in image coordinates. Camera motion is not
  compensated; DUT videos are mostly from a static or slowly moving camera.
- Pixel-to-body-length normalisation assumes the box size tracks the drone's
  true size; very small boxes (< 10 px) make speed estimates noisy.
- Real video tracks have no anomaly labels. Video recall is measured on
  anomalies injected into real tracks, which follow the synthetic anomaly
  types; real-world video anomalies may look different. The DUT flights are
  assumed to be normal.
- Only 20 DUT videos exist, so the video test set is 4 videos (15 tracks);
  its numbers have wide error bars.
- UAV-SEAD anomalies are state-estimation faults. Only the part of a fault
  that shows up in the estimated position is visible to these features.
