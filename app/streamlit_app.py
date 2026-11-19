"""Streamlit demo.

    streamlit run app/streamlit_app.py
"""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS, OUTPUTS, REPORTS, TRAJ_DATA  # noqa: E402

st.set_page_config(page_title="Drone Trajectory Anomaly Detection", layout="wide")
st.title("Drone Detection, Tracking & Trajectory Anomaly Detection")

tab_run, tab_synth, tab_results = st.tabs(["Analyse a video", "Synthetic trajectories", "Model results"])


def ffmpeg_h264(src: Path) -> Path:
    """OpenCV writes mp4v which browsers will not play; re-encode if ffmpeg exists."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", str(dst)], check=False)
    return dst if dst.exists() else src


def plot_tracks(tracks, windows, threshold, selected=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tid, g in tracks.groupby("track_id"):
        if selected and tid not in selected:
            continue
        cx, cy = (g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2
        w = windows[windows.track_id == tid]
        anom = bool(len(w) and w.anomalous.any())
        ax.plot(cx, cy, "-", lw=1.8, color="tab:red" if anom else "tab:green", alpha=0.85)
        ax.text(cx.iloc[-1], cy.iloc[-1], str(tid), fontsize=8)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Trajectories (red = contains anomalous windows)")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    return fig


with tab_run:
    have_det = (MODELS / "detector" / "best.pt").exists()
    have_anom = (MODELS / "anomaly" / "calibration.pkl").exists()
    if not (have_det and have_anom):
        st.warning("Trained models not found. Run the training steps in the README first.")
    else:
        with open(MODELS / "anomaly" / "calibration.pkl", "rb") as f:
            cal = pickle.load(f)
        c1, c2, c3, c4 = st.columns(4)
        model_name = c1.selectbox("Anomaly model", list(cal["thresholds"]),
                                  index=list(cal["thresholds"]).index(cal["selected"]))
        conf = c2.slider("Detection confidence", 0.05, 0.8, 0.15, 0.05)
        imgsz = c3.selectbox("Inference size", [640, 960, 1280], index=2)
        max_frames = c4.number_input("Max frames (0 = all)", 0, 100000, 0, 100)

        src_mode = st.radio("Source", ["Upload video", "DUT sample video"], horizontal=True)
        source = None
        if src_mode == "Upload video":
            up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
            if up:
                tmp = Path(tempfile.mkdtemp()) / up.name
                tmp.write_bytes(up.read())
                source = tmp
        else:
            frames_root = list((ROOT / "data" / "raw" / "dut_anti_uav" / "tracking").rglob("video01"))
            if frames_root:
                root = frames_root[0].parent
                vids = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("video"))
                source = root / st.selectbox("Sequence", vids)
            else:
                st.info("DUT tracking videos not downloaded.")

        if source is not None and st.button("Run pipeline", type="primary"):
            from src.pipeline import run
            out = OUTPUTS / "streamlit" / Path(source).stem
            with st.spinner("Detecting, tracking and scoring..."):
                tracks, windows, summary = run(source, out, conf=conf, imgsz=imgsz, model=model_name,
                                               max_frames=max_frames or None)
            st.session_state["result"] = (str(out), tracks, windows, summary, model_name)

        if "result" in st.session_state:
            out, tracks, windows, summary, mname = st.session_state["result"]
            out = Path(out)
            thr = cal["thresholds"][mname]
            left, right = st.columns([3, 2])
            with left:
                vid = out / "annotated.mp4"
                if vid.exists():
                    st.video(str(ffmpeg_h264(vid)))
            with right:
                st.subheader("Tracks")
                st.dataframe(summary, use_container_width=True, hide_index=True)
                if len(tracks):
                    st.pyplot(plot_tracks(tracks, windows, thr))

            if len(windows):
                st.subheader("Anomaly score over time")
                tid = st.selectbox("Track", sorted(windows.track_id.unique()))
                w = windows[windows.track_id == tid]
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(w.t_start, w.score, "-o", ms=3)
                ax.axhline(thr, color="r", ls="--", label="threshold")
                ax.set_xlabel("time (s)"); ax.set_ylabel("score"); ax.legend()
                st.pyplot(fig)
                flagged = w[w.anomalous]
                if len(flagged):
                    st.markdown("**Why windows were flagged**")
                    st.dataframe(flagged[["t_start", "t_end", "score", "reason"]], hide_index=True)


with tab_synth:
    st.markdown("Labelled synthetic trajectories from the test set, scored by the trained model. "
                "Shaded regions are the injected anomalies.")
    pk = TRAJ_DATA / "synthetic_test_tracks.pkl"
    if pk.exists() and (MODELS / "anomaly" / "calibration.pkl").exists():
        from src.pipeline import AnomalyScorer
        from src.trajectory.features import windows_from_segment

        with open(pk, "rb") as f:
            tracks = pickle.load(f)
        scorer = AnomalyScorer()
        labels = [f"#{i}: {t['behaviour']} / {t['anomaly'] if isinstance(t['anomaly'], str) else '+'.join(t['anomaly'])}"
                  for i, t in enumerate(tracks)]
        i = st.selectbox("Trajectory", range(len(tracks)), format_func=lambda k: labels[k])
        tr = tracks[i]
        w = windows_from_segment(tr)
        s, flag = scorer.score(w)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(tr["u"], tr["v"], color="0.6", lw=1)
        ax[0].scatter(tr["u"][tr["label"]], tr["v"][tr["label"]], s=6, c="r", label="true anomaly")
        ax[0].set_aspect("equal", adjustable="datalim"); ax[0].legend(); ax[0].set_title("path (body lengths)")
        ax[1].plot(w["t0"] + 1.5, s, "-o", ms=3)
        ax[1].axhline(scorer.threshold, color="r", ls="--")
        t = tr["t"]
        lab = tr["label"].astype(int)
        for a, b in zip(np.where(np.diff(lab) == 1)[0], np.where(np.diff(lab) == -1)[0]):
            ax[1].axvspan(t[a], t[b], color="r", alpha=0.15)
        ax[1].set_xlabel("time (s)"); ax[1].set_title(f"{scorer.model} score")
        st.pyplot(fig)
    else:
        st.info("Run `python -m src.trajectory.synth` and the anomaly training first.")


with tab_results:
    det = REPORTS / "detection_metrics.json"
    if det.exists():
        st.subheader("Detection (test split)")
        st.dataframe(pd.DataFrame(json.loads(det.read_text())).T.round(3))
    cls = REPORTS / "classifier_metrics.json"
    if cls.exists():
        st.subheader("Drone-model classifier (VisioDECT test crops)")
        st.json(json.loads(cls.read_text()))
    trk = REPORTS / "tracking_metrics.csv"
    if trk.exists():
        st.subheader("Tracking (DUT Anti-UAV videos)")
        st.dataframe(pd.read_csv(trk).round(3), hide_index=True)
    an = REPORTS / "anomaly_metrics.csv"
    if an.exists():
        st.subheader("Trajectory anomaly detection (test splits)")
        st.dataframe(pd.read_csv(an).round(3), hide_index=True)
        pt = REPORTS / "anomaly_per_type.csv"
        if pt.exists():
            st.markdown("Detection rate per anomaly type (normal row = false positive rate)")
            st.dataframe(pd.read_csv(pt).pivot_table(index=["test_set", "type"], columns="model",
                                                     values="rate").round(3))
    figs = sorted((REPORTS / "figures").glob("*.png")) if (REPORTS / "figures").exists() else []
    cols = st.columns(2)
    for k, f in enumerate(figs):
        cols[k % 2].image(str(f), caption=f.stem)
