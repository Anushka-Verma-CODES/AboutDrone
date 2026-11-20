"""Train and evaluate the trajectory anomaly models.

Unsupervised models are fitted on NORMAL windows only: synthetic, telemetry
and real DUT video tracks (train videos only). Thresholds are picked on the
validation splits; test splits are never touched until the end.

Operating points:
- global: F1-optimal on the pooled synthetic + telemetry validation set.
- per source (used by the pipeline): F1-optimal on that source's validation
  set for synthetic and telemetry; for video, the score that gives a fixed
  false-alarm rate (VIDEO_FPR) on the real validation-video tracks, because
  the anomaly share of the injected video set is arbitrary.

    python -m src.anomaly.train
    python -m src.anomaly.train --ablation-no-video   # same, without real video in training
"""
import argparse
import json
import pickle

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.anomaly.models import IForestDetector, LSTMAEDetector, OCSVMDetector, SupervisedGBM
from src.config import MODELS, REPORTS, SEED, TRAJ_DATA
from src.trajectory.features import SUMMARY_NAMES

OUT = MODELS / "anomaly"
FIG = REPORTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
MAX_PER_SOURCE = 80000
VIDEO_WEIGHT = 0.10     # share of the normal training windows that come from real video
VIDEO_FPR = 0.05        # false-alarm budget on real video tracks
NORMAL_CLASSES = ("none", "Normal", "real_video")


def load(name):
    p = TRAJ_DATA / f"{name}.npz"
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=True))
    d["src"] = np.array([name.split("_")[0]] * len(d["y"]))
    return d


def cat(*ds):
    ds = [d for d in ds if d is not None]
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def take(d, idx):
    return {k: v[idx] for k, v in d.items()}


def cap(d, n, rng):
    if len(d["y"]) <= n:
        return d
    return take(d, np.sort(rng.choice(len(d["y"]), n, replace=False)))


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    i = int(np.nanargmax(f[:-1]))
    return float(t[i])


def metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    return {
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "anomaly_rate": float(y.mean()),
        "n": int(len(y)),
    }


class Percentile:
    """Maps raw scores to the percentile of the validation-normal score
    distribution, so different models can be averaged."""

    def __init__(self, ref):
        self.ref = np.sort(ref)

    def __call__(self, s):
        return np.searchsorted(self.ref, s) / len(self.ref)


def main(video_in_training=True, seed=SEED, video=None, experiment=False):
    """Train, evaluate and (by default) save the anomaly models.

    video_in_training=False  ablation: real video tracks left out of training;
                             results go to anomaly_metrics_no_video_train.csv.
    seed                     random seed for sampling and all models.
    video                    {"train","val","test"} window dicts that replace
                             the saved video_*.npz (cross-validation folds).
    experiment=True          return (metrics, per_type) without saving any
                             model, report or figure.
    """
    rng = np.random.default_rng(seed)
    syn = {k: load(f"synthetic_{k}") for k in ("train", "val", "test")}
    tel = {k: load(f"telemetry_{k}") for k in ("train", "val", "test")}
    vid = {k: load(f"video_{k}") for k in ("train", "val", "test")}
    if video is not None:
        vid = {k: dict(v, src=np.array(["video"] * len(v["y"]))) for k, v in video.items()}
    have_tel = tel["train"] is not None
    have_vid = vid["train"] is not None
    print("telemetry available:", have_tel, " real video available:", have_vid)

    train = cat(cap(syn["train"], MAX_PER_SOURCE, rng), cap(tel["train"], MAX_PER_SOURCE, rng) if have_tel else None)
    video_reps = 0
    if have_vid and video_in_training:
        # few real windows next to ~145k others: repeat them so they make up
        # about VIDEO_WEIGHT of the normal training data
        n_norm = int((train["y"] == 0).sum())
        video_reps = max(1, int(round(VIDEO_WEIGHT * n_norm / len(vid["train"]["y"]))))
        train = cat(train, *[vid["train"]] * video_reps)
    val = cat(syn["val"], tel["val"] if have_tel else None)
    tests = {"synthetic": syn["test"]}
    if have_tel:
        tests["telemetry"] = tel["test"]
        tests["combined"] = cat(syn["test"], tel["test"])
    if have_vid:
        tests["video"] = vid["test"]

    normal = take(train, np.where(train["y"] == 0)[0])
    print(f"train windows {len(train['y'])}  (normal {len(normal['y'])}, real video x{video_reps})  val {len(val['y'])}")

    val_normal = take(val, np.where(val["y"] == 0)[0])
    models = {}

    print("fitting isolation forest")
    models["isolation_forest"] = IForestDetector(seed).fit(normal["summary"])
    print("fitting one-class svm")
    models["one_class_svm"] = OCSVMDetector(seed=seed).fit(normal["summary"])
    print("fitting lstm autoencoder on", "cuda" if torch.cuda.is_available() else "cpu")
    lstm = LSTMAEDetector(seed=seed).fit(normal["seq"], val_normal["seq"])
    models["lstm_autoencoder"] = lstm
    print("fitting supervised reference")
    models["supervised_gbm"] = SupervisedGBM(seed).fit(train["summary"], train["y"])

    def raw_score(name, d):
        m = models[name]
        return m.score(d["seq"]) if name == "lstm_autoencoder" else m.score(d["summary"])

    # calibration on validation
    calib, thresholds = {}, {}
    for name in models:
        calib[name] = Percentile(raw_score(name, val_normal))

    def score(name, d):
        if name == "ensemble":
            return 0.5 * (calib["isolation_forest"](raw_score("isolation_forest", d)) +
                          calib["lstm_autoencoder"](raw_score("lstm_autoencoder", d)))
        return raw_score(name, d)

    names = list(models) + ["ensemble"]
    for name in names:
        thresholds[name] = best_threshold(val["y"], score(name, val))

    # Per-source thresholds, also picked on validation only. Telemetry
    # anomalies are much weaker in the kinematics than the synthetic ones, so
    # one shared threshold under-reports them.
    src_val = {"synthetic": syn["val"], "telemetry": tel["val"] if have_tel else None}
    src_thr = {(src, n): best_threshold(v["y"], score(n, v))
               for src, v in src_val.items() if v is not None for n in names}
    if have_vid:
        src_val["video"] = vid["val"]
        real = vid["val"]["y"] == 0
        for n in names:
            src_thr[("video", n)] = float(np.quantile(score(n, vid["val"])[real], 1 - VIDEO_FPR))

    # evaluation
    rows, per_type = [], []
    curves = {}
    for tname, d in tests.items():
        for name in names:
            s = score(name, d)
            m = metrics(d["y"], s, thresholds[name])
            if (tname, name) in src_thr:
                ms = metrics(d["y"], s, src_thr[(tname, name)])
                m.update({"f1_src_thr": ms["f1"], "precision_src_thr": ms["precision"],
                          "recall_src_thr": ms["recall"]})
            rows.append({"test_set": tname, "model": name, **m})
            curves[(tname, name)] = (d["y"], s)
            if tname in ("synthetic", "telemetry", "video"):
                pred = s >= src_thr.get((tname, name), thresholds[name])   # per-source operating point
                for c in np.unique(d["cls"]):
                    if c in NORMAL_CLASSES:
                        mask = d["y"] == 0
                        per_type.append({"test_set": tname, "model": name, "type": "normal (FPR)",
                                         "rate": float(pred[mask].mean()), "n": int(mask.sum())})
                        continue
                    mask = (d["cls"] == c) & (d["y"] == 1)
                    if mask.sum():
                        per_type.append({"test_set": tname, "model": name, "type": c,
                                         "rate": float(pred[mask].mean()), "n": int(mask.sum())})

    res = pd.DataFrame(rows)
    pt = pd.DataFrame(per_type).drop_duplicates(["test_set", "model", "type"])
    if experiment:
        return res, pt
    if not video_in_training:
        res.to_csv(REPORTS / "anomaly_metrics_no_video_train.csv", index=False)
        pt.to_csv(REPORTS / "anomaly_per_type_no_video_train.csv", index=False)
        print(res.drop(columns=["n"]).round(3).to_string())
        return
    res.to_csv(REPORTS / "anomaly_metrics.csv", index=False)
    pt.to_csv(REPORTS / "anomaly_per_type.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(res.drop(columns=["n"]))
        print(pt.pivot_table(index=["test_set", "type"], columns="model", values="rate"))

    # model used by the pipeline is chosen on validation, not test: mean of
    # the per-source validation AUPRCs, so the large sources do not dominate
    val_ap_src = {n: {src: float(average_precision_score(v["y"], score(n, v)))
                      for src, v in src_val.items() if v is not None}
                  for n in names if n != "supervised_gbm"}
    val_ap = {n: float(np.mean(list(d.values()))) for n, d in val_ap_src.items()}
    best = max(val_ap, key=val_ap.get)
    print("selected model (mean per-source val AUPRC):", best, {k: round(v, 3) for k, v in val_ap.items()})

    # ---------------------------------------------------------- save
    joblib.dump(models["isolation_forest"], OUT / "isolation_forest.joblib")
    joblib.dump(models["one_class_svm"], OUT / "one_class_svm.joblib")
    joblib.dump(models["supervised_gbm"], OUT / "supervised_gbm.joblib")
    torch.save(lstm.state(), OUT / "lstm_autoencoder.pt")
    ref = {
        "thresholds": thresholds,
        "source_thresholds": {f"{k[0]}/{k[1]}": v for k, v in src_thr.items()},
        "calib_ref": {k: v.ref for k, v in calib.items()},
        "selected": best,
        "feature_names": SUMMARY_NAMES,
        "normal_median": np.median(normal["summary"], 0),
        "normal_iqr": np.subtract(*np.percentile(normal["summary"], [75, 25], 0)) + 1e-6,
    }
    with open(OUT / "calibration.pkl", "wb") as f:
        pickle.dump(ref, f)
    (REPORTS / "anomaly_summary.json").write_text(json.dumps({
        "selected_model": best, "val_auprc_mean": val_ap, "val_auprc_per_source": val_ap_src,
        "thresholds": thresholds,
        "source_thresholds": {f"{k[0]}/{k[1]}": v for k, v in src_thr.items()},
        "video_fpr_budget": VIDEO_FPR, "video_train_repeats": video_reps,
        "train_windows": int(len(train["y"])), "train_normal_windows": int(len(normal["y"])),
    }, indent=2, default=float))

    # ---------------------------------------------------------- figures
    for tname in tests:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for name in names:
            y, s = curves[(tname, name)]
            fpr, tpr, _ = roc_curve(y, s)
            p, r, _ = precision_recall_curve(y, s)
            ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, s):.3f})")
            ax[1].plot(r, p, label=f"{name} ({average_precision_score(y, s):.3f})")
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC - {tname}")
        ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall - {tname}")
        ax[0].legend(fontsize=8)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"roc_pr_{tname}.png", dpi=130)
        plt.close(fig)

    h = np.array(lstm.history)
    plt.figure(figsize=(5, 3.5))
    plt.plot(h[:, 0], h[:, 1], label="train")
    plt.plot(h[:, 0], h[:, 2], label="val (normal)")
    plt.xlabel("epoch"); plt.ylabel("reconstruction MSE"); plt.legend(); plt.title("LSTM autoencoder")
    plt.tight_layout(); plt.savefig(FIG / "lstm_ae_loss.png", dpi=130); plt.close()

    for tname in tests:
        y, s = curves[(tname, best)]
        plt.figure(figsize=(6, 3.5))
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 60)
        plt.hist(s[y == 0], bins, alpha=0.6, density=True, label="normal")
        plt.hist(s[y == 1], bins, alpha=0.6, density=True, label="anomalous")
        plt.axvline(src_thr.get((tname, best), thresholds[best]), color="k", ls="--", label="threshold")
        plt.legend(); plt.title(f"{best} scores - {tname}")
        plt.tight_layout(); plt.savefig(FIG / f"score_hist_{tname}.png", dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-no-video", action="store_true")
    main(video_in_training=not ap.parse_args().ablation_no_video)
