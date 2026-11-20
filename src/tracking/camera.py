"""Camera motion compensation for trajectories.

Box centres are measured in image coordinates, so when the camera pans or
zooms a hovering drone appears to move. For every frame we estimate the
global image motion relative to the previous frame (similarity transform:
rotation, uniform scale, translation) from background corners tracked with
Lucas-Kanade optical flow, fitted with RANSAC. Chaining the transforms maps
every frame into the coordinates of the first frame; box centres and sizes
are mapped with it before smoothing and feature extraction.

Frames where the fit is unreliable (featureless sky, too few inliers) get
the identity transform, i.e. no compensation for that step.

    python -m src.tracking.camera      # cache motion for the DUT videos
"""
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from src.config import DUT_RAW, REPORTS, TRAJ_DATA
from src.tracking.track import iter_frames

WORK_SIDE = 640          # motion is estimated on a downscaled copy
MIN_INLIERS = 20
CAM_DIR = TRAJ_DATA / "camera"


def _gray(frame, side=WORK_SIDE):
    h, w = frame.shape[:2]
    f = side / max(h, w)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), f


def estimate_motion(source, max_frames=None):
    """Returns (N, 2, 3) affine matrices mapping frame k pixels to frame 0
    pixels, and the per-step inlier counts."""
    cum = [np.eye(3)]
    inl = [0]
    prev = None
    for i, frame in enumerate(iter_frames(source)):
        if max_frames and i >= max_frames:
            break
        g, f = _gray(frame)
        if prev is None:
            prev = g
            continue
        step = np.eye(3)
        n_in = 0
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=8)
        if pts is not None and len(pts) >= MIN_INLIERS:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= MIN_INLIERS:
                # maps current-frame points onto previous-frame points
                M, mask = cv2.estimateAffinePartial2D(nxt[ok], pts[ok], method=cv2.RANSAC,
                                                      ransacReprojThreshold=1.5)
                if M is not None and mask is not None and mask.sum() >= MIN_INLIERS:
                    n_in = int(mask.sum())
                    S = np.diag([f, f, 1.0])            # full-res <-> work-res
                    step = np.linalg.inv(S) @ np.vstack([M, [0, 0, 1]]) @ S
        cum.append(cum[-1] @ step)
        inl.append(n_in)
        prev = g
    return np.array(cum)[:, :2, :], np.array(inl)


def compensate(frames, cx, cy, size, cam):
    """Map box centres / sizes of the given frames into first-frame pixels."""
    A = cam[np.clip(frames, 0, len(cam) - 1)]
    x = A[:, 0, 0] * cx + A[:, 0, 1] * cy + A[:, 0, 2]
    y = A[:, 1, 0] * cx + A[:, 1, 1] * cy + A[:, 1, 2]
    zoom = np.sqrt(np.abs(A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]))
    return x, y, size * zoom


def dut_videos():
    root = next(p for p in (DUT_RAW / "tracking").rglob("video01") if p.is_dir()).parent
    return root, sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("video"))


def _one(args):
    root, name = args
    cam, inl = estimate_motion(root / name)
    np.savez_compressed(CAM_DIR / f"{name}.npz", cam=cam, inliers=inl)
    step = np.diff(cam[:, :, 2], axis=0)
    return name, {"frames": len(cam), "median_shift_px": float(np.median(np.hypot(*step.T))),
                  "p95_shift_px": float(np.percentile(np.hypot(*step.T), 95)),
                  "identity_steps": float((inl[1:] == 0).mean())}


def load_cam(name):
    p = CAM_DIR / f"{name}.npz"
    return np.load(p)["cam"] if p.exists() else None


def main(workers=3):
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    root, videos = dut_videos()
    stats = {}
    with ProcessPoolExecutor(workers) as ex:
        for name, st in ex.map(_one, [(root, v) for v in videos]):
            stats[name] = st
            print(name, st)
    (REPORTS / "camera_motion.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
