import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Logo Tracker", layout="wide", page_icon="🎭")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.step-wrap {
    border: 1.5px solid #E5E7EB;
    border-radius: 14px;
    padding: 22px 26px 18px;
    margin-bottom: 28px;
    background: #FAFAFA;
}
.step-wrap.active { border-color: #6C63FF; background: #F5F4FF; }
.step-wrap.done   { border-color: #22C55E; background: #F0FDF4; }

.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}
.step-num {
    background: #6C63FF;
    color: white;
    border-radius: 50%;
    width: 30px; height: 30px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px; flex-shrink: 0;
}
.step-num.done-num { background: #22C55E; }
.step-title { font-size: 18px; font-weight: 700; color: #111; }
.step-sub   { font-size: 13px; color: #666; margin-top: 2px; margin-left: 42px; }

.face-card {
    border: 1.5px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    background: white;
}
.face-card img { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎭 Face Logo Tracker")
st.caption("Three steps: crop → reformat for Shorts → assign logos to each face.")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("src_path", None),
    ("cropped_path", None),
    ("shorts_path", None),
    ("face_snapshots", []),   # list of (face_index, jpeg_bytes, bbox_norm)
    ("fps", 30.0),
    ("src_w", 0),
    ("src_h", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

mp_face_detection = mp.solutions.face_detection

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def overlay_logo_on_frame(frame, logo_bgra, bbox):
    fh, fw = frame.shape[:2]
    xmin, ymin, bw, bh = bbox
    x = int(xmin * fw);  y = int(ymin * fh)
    w = int(bw * fw);    h = int(bh * fh)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > fw or y + h > fh:
        return frame
    resized = cv2.resize(logo_bgra, (w, h), interpolation=cv2.INTER_AREA)
    if resized.shape[2] == 4:
        alpha  = resized[:, :, 3:4] / 255.0
        logo_b = resized[:, :, :3].astype(float)
        roi    = frame[y:y+h, x:x+w].astype(float)
        frame[y:y+h, x:x+w] = (logo_b * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        frame[y:y+h, x:x+w] = resized[:, :, :3]
    return frame


def reformat_to_shorts(frame, target_w=1080, target_h=1920):
    """Letterbox / crop a landscape frame into 9:16 (1080×1920)."""
    h, w = frame.shape[:2]
    # scale so height fills target_h
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, target_h))
    # centre-crop width
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if new_w >= target_w:
        x_off = (new_w - target_w) // 2
        canvas = resized[:, x_off:x_off + target_w]
    else:
        x_off = (target_w - new_w) // 2
        canvas[:, x_off:x_off + new_w] = resized
    return canvas


def detect_faces_in_frame(frame, detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    faces = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            faces.append((bb.xmin, bb.ymin, bb.width, bb.height))
    return faces


def grab_midframe(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def frame_to_jpeg_bytes(frame_rgb):
    pil = Image.fromarray(frame_rgb)
    import io
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload & Crop
# ══════════════════════════════════════════════════════════════════════════════
step1_done = st.session_state.cropped_path is not None
s1_class = "done" if step1_done else "active"
st.markdown(f"""
<div class="step-wrap {s1_class}">
  <div class="step-header">
    <div class="step-num {'done-num' if step1_done else ''}">{'✓' if step1_done else '1'}</div>
    <div class="step-title">Upload & Crop</div>
  </div>
  <div class="step-sub">Trim your video to the clip you want to use.</div>
</div>
""", unsafe_allow_html=True)

video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="vu")

if video_file:
    if st.session_state.src_path is None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()
        st.session_state.src_path = tfile.name
        cap_i = cv2.VideoCapture(tfile.name)
        st.session_state.fps   = cap_i.get(cv2.CAP_PROP_FPS) or 30
        st.session_state.src_w = int(cap_i.get(cv2.CAP_PROP_FRAME_WIDTH))
        st.session_state.src_h = int(cap_i.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap_i.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.src_duration = total_f / st.session_state.fps
        cap_i.release()

    duration = st.session_state.src_duration
    st.video(st.session_state.src_path)
    st.caption(f"Duration: **{duration:.1f}s** · {st.session_state.src_w}×{st.session_state.src_h} · {st.session_state.fps:.0f} fps")

    c1, c2 = st.columns(2)
    with c1:
        start_s = st.number_input("Start (seconds)", 0.0, max(0.0, duration - 0.5), 0.0, 0.5)
    with c2:
        end_s   = st.number_input("End (seconds)",   0.5, duration, min(60.0, duration), 0.5)

    if st.button("✂️ Crop clip", use_container_width=True):
        if end_s <= start_s:
            st.error("End time must be after start time.")
        else:
            fps = st.session_state.fps
            sf  = int(start_s * fps)
            ef  = int(end_s   * fps)
            cap = cv2.VideoCapture(st.session_state.src_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
            w = st.session_state.src_w;  h = st.session_state.src_h
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer = cv2.VideoWriter(out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            prog = st.progress(0, text="Cropping…")
            total = ef - sf
            for i in range(total):
                ret, frm = cap.read()
                if not ret: break
                writer.write(frm)
                prog.progress((i+1)/total, text=f"Cropping… {i+1}/{total}")
            cap.release(); writer.release(); prog.empty()
            st.session_state.cropped_path = out.name
            # reset downstream
            st.session_state.shorts_path    = None
            st.session_state.face_snapshots = []
            st.success(f"✅ Cropped to {end_s - start_s:.1f}s")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Reformat to Shorts (9:16)
# ══════════════════════════════════════════════════════════════════════════════
step2_done = st.session_state.shorts_path is not None
if st.session_state.cropped_path:
    s2_class = "done" if step2_done else "active"
    st.markdown(f"""
    <div class="step-wrap {s2_class}">
      <div class="step-header">
        <div class="step-num {'done-num' if step2_done else ''}">{'✓' if step2_done else '2'}</div>
        <div class="step-title">Reformat for TikTok / YouTube Shorts</div>
      </div>
      <div class="step-sub">Converts your clip to 1080×1920 vertical (9:16). Wide videos are centre-cropped; tall videos are padded.</div>
    </div>
    """, unsafe_allow_html=True)

    tw = st.number_input("Output width",  value=1080, step=2)
    th = st.number_input("Output height", value=1920, step=2)

    if st.button("📱 Reformat to Shorts", use_container_width=True):
        fps = st.session_state.fps
        cap = cv2.VideoCapture(st.session_state.cropped_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        writer = cv2.VideoWriter(out.name, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (int(tw), int(th)))
        prog = st.progress(0, text="Reformatting…")
        for i in range(total):
            ret, frm = cap.read()
            if not ret: break
            writer.write(reformat_to_shorts(frm, int(tw), int(th)))
            prog.progress((i+1)/total, text=f"Reformatting… {i+1}/{total}")
        cap.release(); writer.release(); prog.empty()
        st.session_state.shorts_path    = out.name
        st.session_state.face_snapshots = []   # reset downstream
        st.success("✅ Reformatted to vertical!")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Assign logos to faces & render
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.shorts_path:
    st.markdown("""
    <div class="step-wrap active">
      <div class="step-header">
        <div class="step-num">3</div>
        <div class="step-title">Assign Logos to Faces</div>
      </div>
      <div class="step-sub">We scan the mid-frame of your clip, show each detected face, and let you pick which logo goes on each one.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Logo upload ───────────────────────────────────────────────────────────
    logo_files = st.file_uploader(
        "Upload logos (up to 5 — PNG with transparency recommended)",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True,
        key="logos"
    )
    if logo_files and len(logo_files) > 5:
        st.warning("Only first 5 logos used.")
        logo_files = logo_files[:5]

    if logo_files:
        st.markdown("**Your logos:**")
        lcols = st.columns(len(logo_files))
        for i, lf in enumerate(logo_files):
            with lcols[i]:
                st.image(lf, caption=f"Logo {i+1}", width=70)

    # ── Face detection on mid-frame ───────────────────────────────────────────
    if st.button("🔍 Detect faces in clip", use_container_width=True):
        mid = grab_midframe(st.session_state.shorts_path)
        if mid is None:
            st.error("Could not read a frame from the video.")
        else:
            with mp_face_detection.FaceDetection(model_selection=1,
                                                  min_detection_confidence=0.4) as det:
                faces = detect_faces_in_frame(mid, det)

            snapshots = []
            fh, fw = mid.shape[:2]
            for idx, (xmin, ymin, bw, bh) in enumerate(faces):
                x = max(0, int(xmin * fw));  y = max(0, int(ymin * fh))
                w = min(int(bw * fw), fw - x);  h = min(int(bh * fh), fh - y)
                crop_bgr  = mid[y:y+h, x:x+w]
                crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                jbytes    = frame_to_jpeg_bytes(crop_rgb)
                snapshots.append({"idx": idx, "jpeg": jbytes,
                                   "bbox": (xmin, ymin, bw, bh)})
            st.session_state.face_snapshots = snapshots
            if not snapshots:
                st.warning("No faces found in the mid-frame. Try adjusting your clip or detection confidence.")
            st.rerun()

    # ── Per-face logo assignment UI ───────────────────────────────────────────
    face_logo_map = {}   # face_idx → logo_index (0-based) or None

    if st.session_state.face_snapshots:
        n_faces = len(st.session_state.face_snapshots)
        n_logos = len(logo_files) if logo_files else 0
        logo_options = (["— no logo —"] +
                        [f"Logo {i+1}" for i in range(n_logos)])

        st.markdown(f"**{n_faces} face(s) detected.** Assign a logo to each:")

        cols = st.columns(min(n_faces, 4))
        for snap in st.session_state.face_snapshots:
            fi = snap["idx"]
            col = cols[fi % len(cols)]
            with col:
                st.markdown('<div class="face-card">', unsafe_allow_html=True)
                st.image(snap["jpeg"], caption=f"Face {fi+1}", use_container_width=True)
                choice = st.selectbox(
                    f"Logo for face {fi+1}",
                    options=logo_options,
                    key=f"logo_choice_{fi}"
                )
                if choice != "— no logo —":
                    logo_idx = int(choice.split()[-1]) - 1
                    face_logo_map[fi] = logo_idx
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Confidence slider & render button ─────────────────────────────────────
    confidence = st.slider("Face detection confidence (processing)", 0.0, 1.0, 0.45, 0.05)

    if (st.session_state.face_snapshots and logo_files and
            st.button("🚀 Render final video", use_container_width=True, type="primary")):

        if not face_logo_map:
            st.warning("Assign at least one logo to a face before rendering.")
        else:
            # Prepare logo arrays
            logos_bgra = []
            for lf in logo_files:
                pil = Image.open(lf).convert("RGBA")
                arr = np.array(pil)
                logos_bgra.append(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))

            cap   = cv2.VideoCapture(st.session_state.shorts_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30
            fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out   = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer = cv2.VideoWriter(out.name, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (fw, fh))

            prog    = st.progress(0, text="Rendering…")
            preview = st.empty()

            with mp_face_detection.FaceDetection(model_selection=1,
                                                  min_detection_confidence=confidence) as det:
                for fi in range(total):
                    ret, frame = cap.read()
                    if not ret: break

                    faces = detect_faces_in_frame(frame, det)

                    # Match detected faces to assignment by proximity to snapshot bbox
                    snap_bboxes = [s["bbox"] for s in st.session_state.face_snapshots]

                    for det_bbox in faces:
                        # find closest snapshot face by bbox centre distance
                        dx = det_bbox[0] + det_bbox[2]/2
                        dy = det_bbox[1] + det_bbox[3]/2
                        best_snap = min(
                            range(len(snap_bboxes)),
                            key=lambda i: abs(snap_bboxes[i][0]+snap_bboxes[i][2]/2 - dx)
                                        + abs(snap_bboxes[i][1]+snap_bboxes[i][3]/2 - dy)
                        )
                        if best_snap in face_logo_map:
                            logo_idx = face_logo_map[best_snap]
                            frame = overlay_logo_on_frame(frame, logos_bgra[logo_idx], det_bbox)

                    writer.write(frame)
                    prog.progress((fi+1)/total, text=f"Rendering… {fi+1}/{total}")
                    if fi % 20 == 0:
                        preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                      channels="RGB", use_container_width=True)

            cap.release(); writer.release()
            prog.empty(); preview.empty()

            st.success("✅ Done! Your video is ready.")
            with open(out.name, "rb") as f:
                st.download_button(
                    "⬇️ Download final video",
                    data=f,
                    file_name="shorts_face_logos.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
