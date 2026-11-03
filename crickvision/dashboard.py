"""Local, single-user Streamlit interface for CrickVision video processing and research."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import cv2

MAX_UPLOAD_BYTES = 150 * 1024 * 1024

st.set_page_config(page_title="CrickVision", page_icon="🏏", layout="wide")
st.title("🏏 CrickVision")
st.caption("Local cricket-video analysis and open-source resource discovery")
st.warning(
    "Research prototype only. Track IDs can switch, shot cuts can be false positives, "
    "and the default detector is trained on generic COCO images—not cricket footage. "
    "Run locally for a single trusted user; do not expose this dashboard publicly."
)

with st.expander("Discover open-source datasets, models and reference projects", expanded=False):
    st.write(
        "Search public metadata APIs without API keys. This does not download or execute "
        "third-party assets; check the licence, annotations and source-footage rights yourself."
    )
    resource_limit = st.slider("Maximum results per source", min_value=1, max_value=30, value=10)
    if st.button("Search public resources"):
        with st.spinner("Querying public dataset, model and repository metadata…"):
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "crickvision.research", "--limit", str(resource_limit), "--json"],
                    capture_output=True, text=True, timeout=35, check=False,
                )
            except subprocess.TimeoutExpired:
                st.error("Resource search timed out. Try again later.")
            else:
                if result.returncode not in (0, 2):
                    st.error("Resource search could not produce a report.")
                    st.code((result.stderr or result.stdout or "No diagnostic output")[-6000:])
                else:
                    try:
                        report = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        st.error("Resource search returned invalid JSON.")
                        st.code(result.stdout[-6000:])
                    else:
                        st.caption(
                            f"Source status: {report.get('status')} · "
                            f"{report.get('healthy_sources', 0)} of "
                            f"{len(report.get('sources', {}))} sources responded"
                        )
                        st.info(report.get("safety_note", "Verify permissions before use."))
                        for source_name, source in report.get("sources", {}).items():
                            with st.expander(f"{source_name.title()} · {source.get('count', 0)} results"):
                                if source.get("error"):
                                    st.warning(source["error"])
                                for item in source.get("items", []):
                                    st.markdown(f"**[{item.get('id', 'Unnamed resource')}]({item.get('url', '')})**")
                                    licence = item.get("license_tag") or "Licence not declared in API metadata"
                                    st.caption(f"Licence metadata: {licence}")
                                    if item.get("description"):
                                        st.write(item["description"])
                                    st.caption(item.get("warning", "Verify provenance and terms before use."))

st.divider()
st.subheader("Analyse a video")
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

if uploaded is not None and uploaded.size > MAX_UPLOAD_BYTES:
    st.error("This file exceeds the 150 MB limit. Trim the clip locally and try again.")
elif uploaded is not None:
    st.video(uploaded)
    st.caption(f"Selected file: {Path(uploaded.name).name} · {uploaded.size / (1024 * 1024):.1f} MB")
    if st.button("Analyse video", type="primary"):
        with tempfile.TemporaryDirectory(prefix="crickvision-") as temp_dir:
            root = Path(temp_dir)
            input_path = root / Path(uploaded.name).name
            output_path = root / "annotated.mp4"
            jsonl_path = root / "detections.jsonl"
            input_path.write_bytes(uploaded.getvalue())
            # Validate actual media decoding; a supported filename extension alone is insufficient.
            capture = cv2.VideoCapture(str(input_path))
            is_open = capture.isOpened()
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_open else 0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if is_open else 0
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if is_open else 0
            fps = float(capture.get(cv2.CAP_PROP_FPS)) if is_open else 0.0
            capture.release()
            if not is_open or frame_count < 1 or width < 1 or height < 1:
                st.error("This file could not be decoded as video by the installed OpenCV build.")
                st.info("Try exporting a short H.264 MP4 locally. The file has not been sent to a hosted service.")
                st.stop()
            info_cols = st.columns(4)
            info_cols[0].metric("Resolution", f"{width} × {height}")
            info_cols[1].metric("Frames", f"{frame_count:,}")
            info_cols[2].metric("Frame rate", f"{fps:.2f} fps" if fps > 0 else "Unknown")
            info_cols[3].metric("Duration", f"{frame_count / fps:.1f} s" if fps > 0 else "Unknown")
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
            try:
                with st.spinner("Processing video locally. CPU inference can be slow…"):
                    result = subprocess.run(
                        command, capture_output=True, text=True,
                        timeout=60 * 60, check=False,
                    )
            except subprocess.TimeoutExpired:
                st.error("Processing exceeded the one-hour dashboard limit.")
                st.info("Try a shorter clip or analyse fewer frames by increasing N.")
                st.stop()
            if result.returncode != 0:
                st.error("Video processing failed.")
                st.code((result.stderr or result.stdout or "No diagnostic output")[-12000:])
            elif not output_path.is_file() or not jsonl_path.is_file():
                st.error("The process exited without producing both expected output files.")
                st.code(result.stdout[-6000:])
            else:
                records = []
                malformed_lines = 0
                with jsonl_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            malformed_lines += 1
                st.success("Processing finished. Review the outputs before using them for analysis.")
                result_cols = st.columns(3)
                result_cols[0].metric("Output video", f"{output_path.stat().st_size / (1024 * 1024):.1f} MB")
                result_cols[1].metric("JSONL records", f"{len(records):,}")
                result_cols[2].metric("Malformed JSONL lines", str(malformed_lines))
                if malformed_lines:
                    st.warning("Some JSONL lines could not be parsed. Do not treat this export as valid until investigated.")
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
    st.info("Upload a local video to begin. No video is uploaded to a hosted service by this app.")
