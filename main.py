import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Logo Tracker", layout="wide", page_icon="🎭")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .step-badge {
    display: inline-block;
    background: #6C63FF;
    color: white;
    border-radius: 50%;
    width: 32px; height: 32px;
    line-height: 32px;
    text-align: center;
    font-weight: 700;
    font-size: 16px;
    margin-right: 10px;
  }
  .step-title {
    font-size: 22px;
    font-weight: 700;
    display: flex;
    align-items: center;
    margin-bottom: 4px;
  }
  .step-box {
    border: 1.5px solid #e0e0e0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 28px;
    background: #fafafa;
  }
  .done-badge {
    background: #22c55e;
    color: white;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 13px;
    margin-left: 10px;
  }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Face Logo Tracker")
st.caption("Crop your video into a short clip, then overlay motion-tracked logos on every face.")

mp_face_detection = mp.solutions.face_detection

# ── Session state ─────────────────────────────────────────────────────────────
if "cropped_path" not in st.session_state:
    st.session_state.cropped_path = None

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Crop Video
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="step-box">
  <div class="step-title">
    <span class="step-badge">1</span> Crop Your Video
  </div>
  <p style="color:#666;margin-top:4px;">Upload a video and trim it to a short clip before processing.</p>
</div>
""", unsafe_allow_html=True)

video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="video_upload")

if video_file:
    # Save upload to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(video_file.read())
    tfile.flush()
    src_path = tfile.name

    cap_info = cv2.VideoCapture(src_path)
    fps = cap_info.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap_info.release()

    st.video(src_path)
    st.caption(f"Duration: **{duration:.1f}s** · FPS: **{fps:.0f}** · Frames: **{total_frames}**")

    col1, col2 = st.columns(2)
    with col1:
        start_sec = st.number_input("Start time (seconds)", min_value=0.0,
                                    max_value=max(0.0, duration - 0.5),
                                    value=0.0, step=0.5)
    with col2:
        end_sec = st.number_input("End time (seconds)", min_value=0.5,
                                  max_value=duration,
                                  value=min(30.0, duration), step=0.5)

    if st.button("✂️ Crop Video", use_container_width=True):
        if end_sec <= start_sec:
            st.error("End time must be after start time.")
        else:
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            cap = cv2.VideoCapture(src_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_file.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            progress = st.progress(0, text="Cropping…")
            total = end_frame - start_frame
            for i in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                progress.progress((i + 1) / total, text=f"Cropping… {i+1}/{total} frames")

            cap.release()
            writer.release()
            progress.empty()

            st.session_state.cropped_path = out_path
            st.success(f"✅ Cropped! Clip is {end_sec - start_sec:.1f}s long.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Add Logos
# ══════════════════════════════════════════════════════════════════════════════
step2_done = st.session_state.cropped_path is not None
badge_suffix = '<span class="done-badge">✓ Ready</span>' if step2_done else ""

st.markdown(f"""
<div class="step-box">
  <div class="step-title">
    <span class="step-badge">2</span> Overlay Logos on Faces {badge_suffix}
  </div>
  <p style="color:#666;margin-top:4px;">Upload up to 5 logos. Each detected face gets the next logo in your list (cycles if there are more faces than logos).</p>
</div>
""", unsafe_allow_html=True)

if not step2_done:
    st.info("Complete Step 1 first to unlock this step.")
else:
    logo_files = st.file_uploader(
        "Upload logos (PNG with transparency recommended) — up to 5",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="logo_upload"
    )

    if logo_files:
        if len(logo_files) > 5:
            st.warning("Only the first 5 logos will be used.")
            logo_files = logo_files[:5]

        # Preview logos
        cols = st.columns(len(logo_files))
        for i, lf in enumerate(logo_files):
            with cols[i]:
                st.image(lf, caption=f"Logo {i+1}", width=80)

    confidence = st.slider("Face detection confidence", 0.0, 1.0, 0.5, 0.05)

    if logo_files and st.button("🚀 Process Video", use_container_width=True, type="primary"):

        # Prepare logos as numpy arrays
        logos_cv = []
        for lf in logo_files:
            pil_img = Image.open(lf).convert("RGBA")
            arr = np.array(pil_img)
            # RGBA → BGRA
            bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            logos_cv.append(bgra)

        def overlay_logo(frame, logo, bbox):
            fh, fw = frame.shape[:2]
            xmin, ymin, width, height = bbox
            x = int(xmin * fw)
            y = int(ymin * fh)
            w = int(width * fw)
            h = int(height * fh)
            if x < 0 or y < 0 or x + w > fw or y + h > fh or w <= 0 or h <= 0:
                return frame
            resized = cv2.resize(logo, (w, h), interpolation=cv2.INTER_AREA)
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3:4] / 255.0
                logo_rgb = resized[:, :, :3]
                roi = frame[y:y+h, x:x+w].astype(float)
                blended = logo_rgb * alpha + roi * (1 - alpha)
                frame[y:y+h, x:x+w] = blended.astype(np.uint8)
            else:
                frame[y:y+h, x:x+w] = resized[:, :, :3]
            return frame

        cap = cv2.VideoCapture(st.session_state.cropped_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        progress = st.progress(0, text="Processing frames…")
        preview = st.empty()

        with mp_face_detection.FaceDetection(model_selection=1,
                                              min_detection_confidence=confidence) as detector:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if results.detections:
                    for face_i, detection in enumerate(results.detections):
                        logo = logos_cv[face_i % len(logos_cv)]
                        bb = detection.location_data.relative_bounding_box
                        frame = overlay_logo(frame, logo,
                                             (bb.xmin, bb.ymin, bb.width, bb.height))

                writer.write(frame)

                frame_idx += 1
                progress.progress(frame_idx / total_frames,
                                   text=f"Processing… {frame_idx}/{total_frames} frames")

                # Preview every 15 frames
                if frame_idx % 15 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_container_width=True)

        cap.release()
        writer.release()
        progress.empty()
        preview.empty()

        st.success("✅ Done! Download your video below.")
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name="face_logo_overlay.mp4",
                mime="video/mp4",
                use_container_width=True
            )
