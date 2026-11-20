"""Extra figures / checks after training.

1. Example synthetic trajectories (one per anomaly type) coloured by score.
2. Held-out DUT test videos: false-alarm rate on the real tracks, recall on
   anomalies injected into them, and the same with Kalman smoothing switched
   off (ablation). Uses the video operating point of each model.
3. Example telemetry flights (normal vs anomalous) with labelled ranges.

    python -m src.anomaly.report
"""
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import REPORTS, SEED, TRAJ_DATA
from src.pipeline import AnomalyScorer
from src.trajectory.features import windows_from_segment
from src.trajectory.synth import ANOMALY_TYPES, make_track

FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def synthetic_examples(scorer):
    scorer.use(source="synthetic")
    rng = np.random.default_rng(SEED + 99)
    kinds = ["none"] + ANOMALY_TYPES
    fig, axes = plt.subplots(2, len(kinds), figsize=(3.2 * len(kinds), 6.5))
    for j, k in enumerate(kinds):
        tr = make_track(rng, duration_s=25, anomaly=None if k == "none" else k, noise=0.05)
        w = windows_from_segment(tr)
        s, _ = scorer.score(w)
        # per-sample score = max over covering windows
        ps = np.full(len(tr["t"]), np.nan)
        for t0, si in zip(w["t0"], s):
            a = int(round(t0 * 10))
            seg = ps[a:a + 30]
            ps[a:a + 30] = np.fmax(seg, si)
        ax = axes[0, j]
        sc = ax.scatter(tr["u"], tr["v"], c=ps, cmap="RdYlGn_r", s=5,
                        vmin=scorer.typical_normal(),
                        vmax=scorer.threshold * 1.5 if scorer.threshold > 0 else None)
        ax.set_title(k)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        ax2 = axes[1, j]
        ax2.plot(w["t0"] + 1.5, s)
        ax2.axhline(scorer.threshold, color="r", ls="--", lw=0.8)
        lab = tr["label"].astype(int)
        for a, b in zip(np.where(np.diff(lab) == 1)[0], np.where(np.diff(lab) == -1)[0]):
            ax2.axvspan(tr["t"][a], tr["t"][b], color="r", alpha=0.15)
        ax2.set_xlabel("t (s)")
    axes[1, 0].set_ylabel(f"{scorer.model} score")
    fig.suptitle("Synthetic trajectories coloured by anomaly score (red band = injected anomaly)")
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_examples.png", dpi=120)
    plt.close(fig)


def _video_stats(scorer, d):
    s, f = scorer.score(d)
    real = d["y"] == 0
    groups = np.unique(d["group"][real])
    return {
        "real_windows": int(real.sum()),
        "false_alarm_rate": float(f[real].mean()),
        "real_tracks_flagged": int(sum(f[real & (d["group"] == g)].any() for g in groups)),
        "real_tracks": int(len(groups)),
        "injected_windows": int((~real).sum()),
        "recall_injected": float(f[~real].mean()) if (~real).any() else None,
        "recall_per_type": {c: float(f[d["cls"] == c].mean()) for c in np.unique(d["cls"][~real])},
    }


def real_tracks(scorer):
    p = TRAJ_DATA / "video_test.npz"
    if not p.exists():
        print("no video test set yet (run src.trajectory.video)")
        return None
    d = dict(np.load(p, allow_pickle=True))
    raw_p = TRAJ_DATA / "video_test_unsmoothed.npz"
    raw = dict(np.load(raw_p, allow_pickle=True)) if raw_p.exists() else None
    out = {}
    for name in scorer.cal["thresholds"]:
        scorer.use(name, "video")
        out[name] = {"threshold": float(scorer.threshold), "smoothed": _video_stats(scorer, d)}
        if raw is not None:
            out[name]["unsmoothed"] = _video_stats(scorer, raw)
    (REPORTS / "real_video_flag_rate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # selected model: held-out real video tracks vs synthetic test windows
    scorer.use(scorer.cal["selected"], "video")
    s_real, _ = scorer.score({k: v[d["y"] == 0] for k, v in d.items()})
    syn = dict(np.load(TRAJ_DATA / "synthetic_test.npz", allow_pickle=True))
    s_syn, _ = scorer.score(syn)
    plt.figure(figsize=(6.5, 3.8))
    lo, hi = np.percentile(np.r_[s_real, s_syn], [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    plt.hist(s_syn[syn["y"] == 0], bins, density=True, alpha=0.5, label="synthetic normal")
    plt.hist(s_syn[syn["y"] == 1], bins, density=True, alpha=0.5, label="synthetic anomalous")
    plt.hist(s_real, bins, density=True, histtype="step", lw=2, color="k", label="real DUT test-video tracks")
    plt.axvline(scorer.threshold, color="r", ls="--", label="video threshold")
    plt.legend(fontsize=8)
    plt.title(f"{scorer.model}: real video tracks vs synthetic")
    plt.tight_layout()
    plt.savefig(FIG / "real_vs_synthetic_scores.png", dpi=130)
    plt.close()
    return out


def telemetry_examples(scorer):
    p = TRAJ_DATA / "telemetry_test.npz"
    if not p.exists():
        return
    scorer.use(scorer.cal["selected"], "telemetry")
    d = dict(np.load(p, allow_pickle=True))
    s, _ = scorer.score(d)
    groups = np.unique(d["group"])
    rng = np.random.default_rng(SEED)
    anom = [g for g in groups if d["y"][d["group"] == g].any()]
    norm = [g for g in groups if g.endswith("#0") and d["cls"][d["group"] == g][0] == "Normal"]
    pick = list(rng.choice(anom, min(3, len(anom)), replace=False)) + \
        list(rng.choice(norm, min(3, len(norm)), replace=False))
    fig, axes = plt.subplots(len(pick), 1, figsize=(10, 2.0 * len(pick)), sharex=False)
    for ax, g in zip(np.atleast_1d(axes), pick):
        m = d["group"] == g
        t = d["t0"][m]
        ax.plot(t, s[m], lw=1)
        ax.fill_between(t, 0, 1, where=d["y"][m] == 1, color="r", alpha=0.15,
                        transform=ax.get_xaxis_transform())
        ax.axhline(scorer.threshold, color="r", ls="--", lw=0.8)
        ax.set_title(f"{g.split('#')[0]}  [{d['cls'][m][0]}]", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "telemetry_examples.png", dpi=120)
    plt.close()


def main():
    scorer = AnomalyScorer()
    synthetic_examples(scorer)
    telemetry_examples(scorer)
    real_tracks(scorer)


if __name__ == "__main__":
    main()
