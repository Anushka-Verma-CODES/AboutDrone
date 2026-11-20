# Drone Detection, Tracking and Trajectory Anomaly Detection

Video in, flagged drone trajectories out:

```
Video ─► YOLO11 drone detector ─► ByteTrack / BoT-SORT ─► offline tracklet linking
      ─► camera motion compensation ─► Kalman / RTS smoothing ─► trajectory (u, v, size, t)
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

**Camera motion compensation.** Several DUT videos are filmed with a panning
camera (median 3.5-14 px of image motion per frame), so a hovering drone
appears to move. Global image motion is estimated per frame from background
corners (Lucas-Kanade optical flow + RANSAC similarity fit) and chained, and
box centres / sizes are mapped into the first frame's coordinates before
smoothing (`src/tracking/camera.py`). Aligning frames with the estimate
reduces the frame-to-frame background difference 2.5-4x on the moving-camera
videos (`reports/camera_motion.json`). Featureless sky frames fall back to
no compensation.

**Tracklet linking.** When the drone is missed for a moment the tracker
starts a new ID. Afterwards, tracklets are joined when the new one starts
within 8 s of the old one ending, near the position predicted from the old
one's velocity, with a similar box size (`src/tracking/link.py`).

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
python -m src.detection.block_split                     # VisioDECT split by frame blocks
python -m src.detection.train --data block --epochs 20  # detector
python -m src.detection.classifier_eval                 # classifier, leakage-free splits
python -m src.tracking.camera                           # camera motion of the DUT videos
python -m src.tracking.evaluate                         # tracker comparison + video trajectories
python -m src.anomaly.train                             # anomaly models
python -m src.anomaly.train --ablation-no-video         # optional: without real video in training
python -m src.anomaly.report                            # figures, held-out video check
python -m src.anomaly.robustness                        # bootstrap CIs, 5-fold video CV, seeds
streamlit run app/streamlit_app.py                      # demo
```

Single video from the command line:

```bash
python -m src.pipeline --source my_video.mp4 --out outputs/my_video
# options: --tracker bytetrack|botsort  --no-linking  --no-camera-comp  --min-track-conf 0.75
#          --threshold-source video|telemetry|synthetic|global
```

## Project layout

```
configs/      ByteTrack / BoT-SORT settings for small, fast targets
src/data/     dataset download
src/detection prepare YOLO data, train + evaluate detector and classifier
src/tracking/ detection + tracking, linking, camera motion, Kalman smoothing, MOT evaluation
src/trajectory/ shared features, synthetic generator, telemetry and video conversion
src/anomaly/  models, training, evaluation, robustness (CIs, cross-validation)
src/pipeline.py  end-to-end inference and annotated video
app/          Streamlit demo
reports/      metrics (csv/json) and figures
models/       trained weights
```

## Results

All numbers below are on held-out test splits (files in `reports/`).
Hardware: single RTX 5050 Laptop GPU (8 GB).

### Detection (YOLO11s, 20 epochs, imgsz 960)

VisioDECT images are consecutive video frames, so the detector is trained and
tested on the frame-block split (`src/detection/block_split.py`, the same
test blocks as the classifier). DUT keeps its own split.

| Test set | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| All | 0.943 | 0.916 | 0.929 | 0.944 | 0.568 |
| DUT Anti-UAV | 0.942 | 0.884 | 0.912 | 0.929 | 0.576 |
| VisioDECT | 0.946 | 0.940 | 0.943 | 0.950 | 0.564 |

With the earlier random image split, VisioDECT reached mAP50 0.990: about
0.04 of that came from near-duplicate frames shared between train and test.
DUT, which was never affected, is unchanged (0.928 before, 0.929 now).
Earlier metrics: `reports/detection_metrics.json`; training curves:
`models/runs/detector_block/`.

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

### Tracking (20 DUT tracking videos)

Configurations were compared on the 12 anomaly-training videos only and the
best (BoT-SORT + linking + track-confidence filter) is used everywhere.

| Tracker | Post-processing | MOTA (test / all) | IDF1 (test / all) | ID switches (all) | Precision (all) | Success@0.5 (all) |
|---|---|---|---|---|---|---|
| ByteTrack | none | 0.559 / 0.671 | 0.511 / 0.585 | 99 | 0.800 | 0.879 |
| ByteTrack | linking | 0.562 / 0.674 | 0.588 / 0.701 | 33 | 0.800 | 0.879 |
| ByteTrack | linking + filter | 0.889 / 0.834 | 0.687 / 0.759 | 31 | 0.933 | 0.879 |
| BoT-SORT | none | 0.568 / 0.696 | 0.534 / 0.601 | 53 | 0.807 | 0.896 |
| BoT-SORT | linking | 0.571 / 0.697 | 0.722 / 0.778 | 10 | 0.807 | 0.896 |
| **BoT-SORT** | **linking + filter** | **0.913 / 0.862** | **0.850 / 0.844** | **8** | **0.945** | **0.895** |

"test" = the 4 anomaly-test videos, "all" = all 20. For reference, the first
version of the project (old detector, plain ByteTrack) had MOTA 0.825,
IDF1 0.648 and 94 ID switches over all 20 videos.

- BoT-SORT's global motion compensation keeps predictions aligned when the
  camera pans (fewer fragmentations, higher success rate).
- Linking joins tracklets split by short misses: ID switches 53 to 10.
- The filter drops tracks whose most confident detection stays below 0.75.
  The retrained detector produces more short, low-confidence false tracks
  on DUT video than the first one; dropping them lifts precision from 0.81 to
  0.95 at almost no recall cost. The threshold was tuned on the training
  videos (real drone tracks only start to be lost above 0.85); lower it with
  `--min-track-conf` for footage with very distant drones.

Per-video numbers: `reports/tracking_metrics.csv`; all configurations:
`reports/tracking_comparison.csv`.

### Trajectory anomaly detection

Each source uses its own operating point picked on its validation set (see
above). The pipeline model is chosen by the mean of the per-source validation
AUPRCs; the **LSTM autoencoder** and the IF + LSTM ensemble tie (0.752) and
the LSTM-AE is used. The supervised gradient boosting model sees anomaly
labels and is only a reference.

| Test set | Model | AUROC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| Synthetic | Isolation Forest | 0.982 | 0.908 | 0.870 | 0.858 | 0.883 |
| Synthetic | One-Class SVM | 0.970 | 0.889 | 0.853 | 0.859 | 0.848 |
| Synthetic | **LSTM autoencoder** | 0.975 | 0.926 | 0.874 | 0.917 | 0.835 |
| Synthetic | Ensemble (IF + LSTM) | 0.984 | 0.933 | 0.884 | 0.907 | 0.862 |
| Synthetic | *Supervised GBM* | *0.992* | *0.970* | *0.925* | *0.934* | *0.917* |
| UAV-SEAD | Isolation Forest | 0.804 | 0.368 | 0.360 | 0.256 | 0.610 |
| UAV-SEAD | One-Class SVM | 0.402 | 0.192 | 0.166 | 0.091 | 1.000 |
| UAV-SEAD | **LSTM autoencoder** | 0.843 | 0.426 | 0.400 | 0.271 | 0.763 |
| UAV-SEAD | Ensemble (IF + LSTM) | 0.834 | 0.426 | 0.391 | 0.280 | 0.650 |
| UAV-SEAD | *Supervised GBM* | *0.780* | *0.348* | *0.316* | *0.212* | *0.622* |
| DUT video (held-out) | Isolation Forest | 0.961 | 0.968 | 0.898 | 0.891 | 0.905 |
| DUT video (held-out) | One-Class SVM | 0.928 | 0.944 | 0.868 | 0.865 | 0.872 |
| DUT video (held-out) | **LSTM autoencoder** | 0.962 | 0.964 | 0.878 | 0.844 | 0.915 |
| DUT video (held-out) | Ensemble (IF + LSTM) | 0.967 | 0.972 | 0.884 | 0.853 | 0.918 |
| DUT video (held-out) | *Supervised GBM* | *0.964* | *0.975* | *0.929* | *0.940* | *0.918* |

Notes:
- UAV-SEAD with its own threshold: the LSTM-AE finds 76 % of anomalous
  windows (11 % with one global threshold), at a 20 % false-alarm rate on
  normal flights. The OCSVM collapses to "flag everything" there.
- The video test set is about half injected anomalies, which inflates AUPRC
  and F1. What matters there is the false-alarm rate on the real tracks and
  the recall on injected anomalies; both are reported with confidence
  intervals and cross-validation below.

Detection rate per anomaly type (LSTM autoencoder, per-source threshold):

| Synthetic | Rate | UAV-SEAD | Rate | DUT video (injected) | Rate |
|---|---|---|---|---|---|
| zigzag | 0.93 | External Position | 0.94 | circling | 1.00 |
| dive | 0.90 | Global Position | 0.53 | zigzag | 0.99 |
| jump | 0.89 | Mechanical | 0.43 | dive | 0.97 |
| circling | 0.78 | Altitude | 0.39 | jump | 0.95 |
| speed_burst | 0.77 | | | speed_burst | 0.86 |
| reversal | 0.74 | | | reversal | 0.72 |
| normal (false alarms) | 0.009 | normal (false alarms) | 0.20 | normal (false alarms) | 0.165 |

#### What each video improvement contributes (same 4 held-out videos, ensemble)

AUROC is the fair comparison (threshold-free); the false-alarm / recall
columns use the video threshold of each run, so they trade against each other.

| Setting | AUROC | False alarms on real tracks | Recall (injected) |
|---|---|---|---|
| Everything on | **0.967** | 15.4 % | 0.92 |
| without camera motion compensation | 0.945 | 25.2 % | 0.94 |
| without Kalman smoothing | 0.963 | 23.0 % | 0.95 |
| without real video in training | 0.947 | 11.4 % | 0.83 |

Real video in training changes synthetic / telemetry AUROC by at most 0.01
(0.03 for the OCSVM) (`reports/anomaly_metrics_no_video_train.csv`,
`reports/real_video_flag_rate.json`).

#### How reliable are these numbers? (`src/anomaly/robustness.py`)

**Bootstrap 95 % confidence intervals** (resampling whole tracks / flights,
1000 resamples, `reports/anomaly_bootstrap_ci.csv`), LSTM autoencoder:

| Test set | Independent tracks | AUROC | False-alarm rate | Recall |
|---|---|---|---|---|
| Synthetic | 1000 | 0.975 [0.967, 0.981] | 0.009 [0.008, 0.011] | 0.835 [0.814, 0.855] |
| UAV-SEAD | 131 flights | 0.843 [0.774, 0.901] | 0.204 [0.151, 0.273] | 0.763 [0.651, 0.863] |
| DUT video | 7 real tracks | 0.962 [0.917, 0.984] | 0.165 [0.029, 0.564] | 0.915 [0.867, 0.968] |

**5-fold cross-validation over the 20 DUT videos** (every video is a test
video once; `reports/anomaly_video_cv.csv`):

| Model | AUROC | False-alarm rate | Recall (injected) |
|---|---|---|---|
| Isolation Forest | 0.961 ± 0.016 | 0.070 ± 0.113 | 0.815 ± 0.107 |
| LSTM autoencoder | 0.962 ± 0.020 | 0.126 ± 0.168 | 0.836 ± 0.113 |
| Ensemble | 0.969 ± 0.014 | 0.104 ± 0.145 | 0.872 ± 0.082 |

Per fold the ensemble's false-alarm rate is 0 %, 1 %, 2 %, 15 % and 34 %.
Looking at where the alarms fall explains the spread:

- The DUT videos contain two flight styles. In videos 14-20 the drone flies
  fast with sharp turns (median peak turning acceleration 4-10 body
  lengths/s², jerk 5-18); in the others it flies smoothly (0.4-1.4 and 1-3).
  Nearly all "false" alarms are windows of the agile flights (e.g. video15:
  61 %, video16: 34-56 %, video19: 75-80 % of windows), plus video04, whose
  camera pans over featureless sky that cannot be compensated (100 %).
- Panning-camera videos are no longer a problem after compensation: video08,
  the fastest pan (median 14 px per frame), gets 0-4 % false alarms.

So the model separates smooth from aggressive flight very consistently; the
DUT videos have no anomaly labels and all of them were assumed normal. Whether
aggressive manoeuvring should be flagged is a labelling decision. The main
test split contains video15 and video16, which is why its false-alarm rate
(15-17 %) is above the 5 % target set on the validation videos.

**Training seeds** (main split, 4 seeds, `reports/anomaly_seed_runs.csv`):
spreads are tiny (AUROC ± 0.000-0.005, recall ± 0.002-0.018, false-alarm
rate ± 0.001-0.014), so the uncertainty comes from the small number of
videos and flights, not from training randomness.

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
- Camera motion compensation is a 2-D similarity model. It cannot separate
  camera rotation from parallax and fails on featureless sky (no
  compensation then, e.g. video04).
- The DUT videos are assumed to show normal flight, but about a third of
  them are fast, sharply turning flights that the models flag. Treating
  "agile" as normal would need such flights in the normal training data
  (or labels saying so).
- Pixel-to-body-length normalisation assumes the box size tracks the drone's
  true size; very small boxes (< 10 px) make speed estimates noisy.
- Real video tracks have no anomaly labels. Video recall is measured on
  anomalies injected into real tracks, which follow the synthetic anomaly
  types; real-world video anomalies may look different. The DUT flights are
  assumed to be normal.
- Only 20 DUT videos exist (4 test videos, 7 real test tracks); see the
  confidence intervals and cross-validation above.
- Every DUT video holds exactly one drone. Track linking (8 s gap) and the
  track-confidence filter (0.75) were tuned on that; multi-drone or very
  distant-drone footage may need looser settings.
- UAV-SEAD anomalies are state-estimation faults. Only the part of a fault
  that shows up in the estimated position is visible to these features.
