"""Kalman (RTS) smoothing of pixel tracks before feature extraction.

Detector boxes jitter by a few pixels from frame to frame. For a drone that
is only 15-20 px wide this is a sizeable fraction of a body length, and
taking two or three derivatives turns it into fake acceleration and jerk.
A constant-velocity model, run forward and then backward (Rauch-Tung-Striebel),
removes most of that jitter without lagging real manoeuvres.

Positions are smoothed in body-length units (pixels / median box size) so
the same noise settings work for near and far drones.
"""
import numpy as np
from scipy.ndimage import median_filter

MEAS_STD_BL = 0.15     # detector jitter, body lengths
ACC_STD_BL = 6.0       # allowed acceleration, body lengths / s^2
LOGSCALE_MEAS_STD = 0.08
LOGSCALE_ACC_STD = 0.5


def rts_cv(t, z, meas_std, acc_std):
    """1-D constant-velocity RTS smoother. t may be irregular."""
    n = len(z)
    if n < 3:
        return np.asarray(z, float).copy()
    R = meas_std ** 2
    xs = np.zeros((n, 2))
    Ps = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2))
    Pp = np.zeros((n, 2, 2))
    Fs = np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0])
    P = np.diag([R, 25.0])
    for k in range(n):
        if k:
            dt = max(t[k] - t[k - 1], 1e-3)
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = acc_std ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
            x = F @ x
            P = F @ P @ F.T + Q
            Fs[k] = F
        xp[k], Pp[k] = x, P
        K = P[:, 0] / (P[0, 0] + R)
        x = x + K * (z[k] - x[0])
        P = P - np.outer(K, P[0])
        xs[k], Ps[k] = x, P
    for k in range(n - 2, -1, -1):
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0]


def smooth_track(t, u, v, scale):
    """Smooth one track. Returns (u, v, scale) in the original units."""
    t = np.asarray(t, float)
    s_med = float(np.median(scale)) or 1.0
    log_s = median_filter(np.log(np.asarray(scale, float)), size=min(9, len(scale)), mode="nearest")
    scale_s = np.exp(rts_cv(t, log_s, LOGSCALE_MEAS_STD, LOGSCALE_ACC_STD))
    u_s = rts_cv(t, np.asarray(u, float) / s_med, MEAS_STD_BL, ACC_STD_BL) * s_med
    v_s = rts_cv(t, np.asarray(v, float) / s_med, MEAS_STD_BL, ACC_STD_BL) * s_med
    return u_s, v_s, scale_s
