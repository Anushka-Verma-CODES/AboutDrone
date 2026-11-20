"""How much should the anomaly numbers be trusted?

1. Bootstrap confidence intervals for the saved models on every test set.
   Resampling is done by whole track / flight (all windows of a track stay
   together, injected video copies stay with their source track), so the
   intervals reflect how few independent trajectories there are.
2. 5-fold cross-validation over the 20 DUT videos: every video is a test
   video exactly once (4 test / 4 val / 12 train per fold).
3. Training-seed spread: the main split retrained with 3 more seeds.

    python -m src.anomaly.robustness              # all three
    python -m src.anomaly.robustness --only ci    # bootstrap only (no retraining)
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.anomaly import train as anomaly_train
from src.config import REPORTS, SEED, TRAJ_DATA
from src.pipeline import AnomalyScorer
from src.trajectory import video as video_traj

N_BOOT = 1000
MODELS = ["isolation_forest", "lstm_autoencoder", "ensemble"]
SEEDS = [1, 2, 3]
N_FOLDS = 5


def base_group(test_set, g):
    if test_set == "video":
        return "_".join(g.split("_")[:2])          # videoNN_<track id>
    if test_set == "telemetry":
        return g.split("#")[0]                     # one flight, both projections
    return g


def _stats(y, s, thr):
    pred = s >= thr
    out = {"fpr": float(pred[y == 0].mean()), "recall": float(pred[y == 1].mean())}
    out["auroc"] = float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else np.nan
    return out


def bootstrap(seed=SEED):
    rng = np.random.default_rng(seed)
    scorer = AnomalyScorer()
    rows = []
    for test_set in ("synthetic", "telemetry", "video"):
        p = TRAJ_DATA / f"{test_set}_test.npz"
        if not p.exists():
            continue
        d = dict(np.load(p, allow_pickle=True))
        groups = np.array([base_group(test_set, g) for g in d["group"]])
        uniq = np.unique(groups)
        idx_of = {g: np.where(groups == g)[0] for g in uniq}
        for m in MODELS:
            scorer.use(m, test_set)
            s, _ = scorer.score(d)
            point = _stats(d["y"], s, scorer.threshold)
            boots = []
            for _ in range(N_BOOT):
                pick = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
                boots.append(_stats(d["y"][pick], s[pick], scorer.threshold))
            b = pd.DataFrame(boots)
            row = {"test_set": test_set, "model": m, "groups": len(uniq)}
            for k in ("auroc", "fpr", "recall"):
                lo, hi = np.nanpercentile(b[k], [2.5, 97.5])
                row.update({k: point[k], f"{k}_lo": float(lo), f"{k}_hi": float(hi)})
            rows.append(row)
            print(row)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "anomaly_bootstrap_ci.csv", index=False)
    return df


def _video_view(res, pt):
    """AUROC, false-alarm rate and recall on the video test set, per model."""
    out = {}
    for m in MODELS:
        r = res[(res.test_set == "video") & (res.model == m)].iloc[0]
        f = pt[(pt.test_set == "video") & (pt.model == m) & (pt.type == "normal (FPR)")]
        out[m] = {"auroc": float(r.auroc), "fpr": float(f.rate.iloc[0]), "recall": float(r.recall_src_thr)}
    return out


def cross_validation():
    videos = sorted(p.stem.replace("dut_tracks_", "") for p in REPORTS.glob("dut_tracks_video*.csv"))
    order = list(np.random.default_rng(SEED).permutation(videos))
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]
    rows = []
    for k in range(N_FOLDS):
        test, val = folds[k], folds[(k + 1) % N_FOLDS]
        train = [v for v in videos if v not in test and v not in val]
        data = {"train": video_traj.build(train, stride=1),
                "val": video_traj.build(val, with_injected=True),
                "test": video_traj.build(test, with_injected=True)}
        res, pt = anomaly_train.main(video=data, experiment=True)
        for m, v in _video_view(res, pt).items():
            rows.append({"fold": k, "test_videos": " ".join(test), "model": m, **v})
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "anomaly_video_cv.csv", index=False)
    return df


def seed_spread():
    rows = []
    for seed in SEEDS:
        res, pt = anomaly_train.main(seed=seed, experiment=True)
        for _, r in res[res.model.isin(MODELS) & res.test_set.isin(["synthetic", "telemetry", "video"])].iterrows():
            f = pt[(pt.test_set == r.test_set) & (pt.model == r.model) & (pt.type == "normal (FPR)")]
            rows.append({"seed": seed, "test_set": r.test_set, "model": r.model, "auroc": r.auroc,
                         "auprc": r.auprc, "f1": r.f1_src_thr, "recall": r.recall_src_thr,
                         "fpr": float(f.rate.iloc[0])})
    main_res = pd.read_csv(REPORTS / "anomaly_metrics.csv")
    main_pt = pd.read_csv(REPORTS / "anomaly_per_type.csv")
    for _, r in main_res[main_res.model.isin(MODELS) & main_res.test_set.isin(["synthetic", "telemetry", "video"])].iterrows():
        f = main_pt[(main_pt.test_set == r.test_set) & (main_pt.model == r.model) & (main_pt.type == "normal (FPR)")]
        rows.append({"seed": SEED, "test_set": r.test_set, "model": r.model, "auroc": r.auroc,
                     "auprc": r.auprc, "f1": r.f1_src_thr, "recall": r.recall_src_thr,
                     "fpr": float(f.rate.iloc[0])})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "anomaly_seed_runs.csv", index=False)
    return df


def summarise(cv, seeds):
    out = {}
    if cv is not None:
        g = cv.groupby("model")[["auroc", "fpr", "recall"]]
        out["video_cv"] = {m: {k: {"mean": float(g.mean().loc[m, k]), "std": float(g.std().loc[m, k]),
                                   "min": float(g.min().loc[m, k]), "max": float(g.max().loc[m, k])}
                               for k in ("auroc", "fpr", "recall")} for m in MODELS}
    if seeds is not None:
        g = seeds.groupby(["test_set", "model"])[["auroc", "auprc", "f1", "recall", "fpr"]]
        mean, std = g.mean(), g.std()
        out["seed_spread"] = {f"{t}/{m}": {k: {"mean": float(mean.loc[(t, m), k]), "std": float(std.loc[(t, m), k])}
                                           for k in mean.columns} for t, m in mean.index}
    (REPORTS / "anomaly_robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main(only=None):
    if only in (None, "ci"):
        bootstrap()
    cv = cross_validation() if only in (None, "cv") else None
    seeds = seed_spread() if only in (None, "seeds") else None
    if cv is not None or seeds is not None:
        summarise(cv, seeds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ci", "cv", "seeds"])
    main(ap.parse_args().only)
