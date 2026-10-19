"""Local, single-user Streamlit interface for CrickVision video processing.

Do not expose this app to untrusted users: uploaded files consume local resources.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CrickVision", page_icon="🏏", layout="wide")
st.title("🏏 CrickVision")
st.caption("Experimental local cricket-video analysis · person detections + heuristic shot cuts")
st.warning(
    "Research prototype only. Track IDs can switch, shot cuts can be false positives, "
    "and the detector is trained on generic COCO images—not cricket footage. "
    "Run locally for a single trusted user; do not expose this dashboard publicly."
)

uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv", "webm"])
col1, col2, col3 = st.columns(3)
with col1:
    threshold = st.slider("Person confidence", 0.05, 0.95, 0.60, 0.05)
with col2:
    sample_every = st.number_input("Analyse every N frames", min_value=1, max_value=120, value=2)
with col3:
    shot_threshold = st.slider("Shot-cut sensitivity", 0.05, 0.95, 0.42, 0.01)
device = st.selectbox("Inference device", ["auto", "cpu", "cuda"], index=0)
tracking = st.checkbox("Enable experimental IoU tracking", value=True)

if uploaded is not None:
    st.video(uploaded)
    if st.button("Analyse video", type="primary"):
        with tempfile.TemporaryDirectory(prefix="crickvision-") as temp_dir:
            root = Path(temp_dir)
            input_path = root / Path(uploaded.name).name
            output_path = root / "annotated.mp4"
            jsonl_path = root / "detections.jsonl"
            input_path.write_bytes(uploaded.getvalue())
            command = [
                sys.executable, "-m", "crickvision.cli",
                "--input", str(input_path),
                "--output", str(output_path),
                "--detections", str(jsonl_path),
                "--threshold", str(threshold),
                "--every-n-frames", str(sample_every),
                "--device", device,
                "--shot-threshold", str(shot_threshold),
            ]
            if not tracking:
                command.append("--no-tracking")
            with st.spinner("Processing video locally. CPU inference can be slow…"):
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=60 * 60, check=False
                )
            if result.returncode != 0:
                st.error("Video processing failed.")
                st.code((result.stderr or result.stdout or "No diagnostic output")[-12000:])
            elif not output_path.is_file() or not jsonl_path.is_file():
                st.error("The process exited without producing both expected output files.")
                st.code(result.stdout[-6000:])
            else:
                st.success("Processing finished.")
                st.code(result.stdout[-6000:])
                st.subheader("Annotated video")
                st.video(output_path.read_bytes())
                st.download_button(
                    "Download annotated MP4", output_path.read_bytes(),
                    file_name="crickvision_annotated.mp4", mime="video/mp4",
                )
                st.download_button(
                    "Download detection JSONL", jsonl_path.read_bytes(),
                    file_name="crickvision_detections.jsonl", mime="application/x-ndjson",
                )
else:
    st.info("Upload a local video to begin. No file is uploaded to a hosted service by this app.")
