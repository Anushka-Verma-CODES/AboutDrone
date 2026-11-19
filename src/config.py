from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

VISIODECT_RAW = RAW / "visiodect"
DUT_RAW = RAW / "dut_anti_uav"
SEAD_RAW = RAW / "uav_sead"

YOLO_DATA = PROCESSED / "yolo"
TRAJ_DATA = PROCESSED / "trajectories"

MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"

# Trajectory settings shared by every source (telemetry, synthetic, video).
# Tracks are resampled to TRAJ_HZ and cut into windows of WINDOW samples.
TRAJ_HZ = 10
WINDOW = 30          # 3 seconds
STRIDE = 5

SEED = 42

for d in (PROCESSED, YOLO_DATA, TRAJ_DATA, MODELS, OUTPUTS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)
