import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Logo Tracker", layout="wide", page_icon="🎭")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.step-wrap {
    border: 1.5px solid #E5E7EB; border-radius: 14px;
    padding: 22px 26px 18px; margin-bottom: 28px; background: #FAFAFA;
}
.step-wrap.active { border-color: #6C63FF; background: #F5F4FF; }
.step-wrap.done   { border-color: #22C55E; background: #F0FDF4; }
.step-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.step-num {
    background: #6C63FF; color: white; border-radius: 50%;
    width: 30px; height: 30px; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 15px; flex-shrink: 0;
}
.step-num.done-num { background: #22C55E; }
.step-title { font-size: 18px; font-weight: 700; color: #111; }
.step-sub   { font-size: 13px; color: #666; margin-top: 2px; margin-left: 42px; }
.face-card  { border: 1.5px solid #E5E7EB; border-radius: 10px; padding: 12px; text-align: center; background: white; }
.manual-card { border: 1.5px dashed #F59E0B; border-radius: 10px; padding: 12px; background: #FFFBEB; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎭 Face Logo Tracker")
st.caption("Crop → Shorts format → assign logos to faces (with manual override for CGI/hidden faces) → render with original audio.")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("src_path", None), ("cropped_path", None), ("shorts_path", None),
    ("face_snapshots", []), ("fps", 30.0), ("src_w", 0), ("src_h", 0), ("src_duration", 0.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

mp_face_det = mp.solutions.face_detection

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def overlay_logo_on_frame(frame, logo_bgra, bbox, scale=1.0):
    fh, fw = frame.shape[:2]
    xmin, ymin, bw, bh = bbox
    # apply scale multiplier (expand around centre)
    cx = xmin + bw / 2;  cy = ymin + bh / 2
    bw *= scale;          bh *= scale
    xmin = cx - bw / 2;  ymin = cy - bh / 2
    x = int(xmin * fw);  y = int(ymin * fh)
    w = int(bw * fw);    h = int(bh * fh)
    # clamp
    x = max(0, x);  y = max(0, y)
    w = min(w, fw - x);  h = min(h, fh - y)
    if w <= 0 or h <= 0:
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
    h, w = frame.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, target_h))
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


def grab_frame_at(path, frame_idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def grab_midframe(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return grab_frame_at(path, total // 2)


def frame_to_jpeg_bytes(frame_rgb):
    buf = io.BytesIO()
    Image.fromarray(frame_rgb).save(buf, format="JPEG")
    return buf.getvalue()


def mux_audio(video_no_audio: str, original_with_audio: str, out_path: str,
              start_sec: float = 0.0, end_sec: float = None):
    """Copy audio track from original into the processed video using ffmpeg."""
    duration_args = []
    if end_sec is not None:
        duration_args = ["-t", str(end_sec - start_sec)]

    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio,
        "-ss", str(start_sec), "-i", original_with_audio,
    ] + duration_args + [
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def run_ffmpeg_crop(src: str, start: float, end: float, out: str):
    """Fast ffmpeg-based video+audio crop (no re-encode for video)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", src,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        out
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_ffmpeg_shorts(src: str, tw: int, th: int, out: str):
    """Reformat to vertical using ffmpeg (keeps audio, fast)."""
    # scale to fill height, then centre-crop width
    vf = (
        f"scale=-2:{th},"
        f"crop={tw}:{th}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        out
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


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
  <div class="step-sub">Trim your video. Audio is preserved automatically.</div>
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
        end_s   = st.number_input("End (seconds)", 0.5, duration, min(60.0, duration), 0.5)

    if st.button("✂️ Crop clip", use_container_width=True):
        if end_s <= start_s:
            st.error("End time must be after start time.")
        else:
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with st.spinner("Cropping (with audio)…"):
                ok = run_ffmpeg_crop(st.session_state.src_path, start_s, end_s, out.name)
            if ok:
                st.session_state.cropped_path   = out.name
                st.session_state.crop_start_sec = start_s
                st.session_state.shorts_path    = None
                st.session_state.face_snapshots = []
                st.success(f"✅ Cropped to {end_s - start_s:.1f}s (audio included)")
                st.rerun()
            else:
                st.error("ffmpeg crop failed — check your video format.")

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
        <div class="step-title">Reformat for TikTok / YouTube Shorts (9:16)</div>
      </div>
      <div class="step-sub">Centre-crops to vertical. Audio is kept.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        tw = st.number_input("Width",  value=1080, step=2)
    with c2:
        th = st.number_input("Height", value=1920, step=2)

    if st.button("📱 Reformat to Shorts", use_container_width=True):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with st.spinner("Reformatting (with audio)…"):
            ok = run_ffmpeg_shorts(st.session_state.cropped_path, int(tw), int(th), out.name)
        if ok:
            st.session_state.shorts_path    = out.name
            st.session_state.face_snapshots = []
            st.success("✅ Reformatted to vertical!")
            st.rerun()
        else:
            st.error("ffmpeg reformat failed.")

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
      <div class="step-sub">
        Auto-detected faces show as cards. For CGI or missed faces, use Manual Override to draw a bbox on any frame.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Logo upload ───────────────────────────────────────────────────────────
    logo_files = st.file_uploader(
        "Upload logos (up to 5, PNG with transparency recommended)",
        type=["png","jpg","jpeg"], accept_multiple_files=True, key="logos"
    )
    if logo_files and len(logo_files) > 5:
        st.warning("Only first 5 logos used.")
        logo_files = logo_files[:5]

    if logo_files:
        lcols = st.columns(len(logo_files))
        for i, lf in enumerate(logo_files):
            with lcols[i]:
                st.image(lf, caption=f"Logo {i+1}", width=70)

    # ── Logo scale & confidence ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        logo_scale = st.slider("Logo size multiplier", 0.5, 3.0, 1.3, 0.1,
                               help="1.0 = exact face bbox. 1.3 covers a bit more of the head.")
    with col_b:
        confidence = st.slider("Face detection confidence", 0.0, 1.0, 0.4, 0.05)

    # ── Detect faces ──────────────────────────────────────────────────────────
    cap_info = cv2.VideoCapture(st.session_state.shorts_path)
    total_frames_s3 = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_info.release()

    scan_frame_idx = st.slider("Scan frame for face detection",
                               0, max(0, total_frames_s3 - 1),
                               total_frames_s3 // 2, 1,
                               help="Scrub to a frame where all faces are visible")

    if st.button("🔍 Detect faces in this frame", use_container_width=True):
        mid = grab_frame_at(st.session_state.shorts_path, scan_frame_idx)
        if mid is None:
            st.error("Could not read frame.")
        else:
            with mp_face_det.FaceDetection(model_selection=1,
                                           min_detection_confidence=confidence) as det:
                faces = detect_faces_in_frame(mid, det)
            snapshots = []
            fh, fw = mid.shape[:2]
            for idx, (xmin, ymin, bw, bh) in enumerate(faces):
                x = max(0, int(xmin * fw));  y = max(0, int(ymin * fh))
                w = min(int(bw * fw), fw - x);  h = min(int(bh * fh), fh - y)
                crop_rgb = cv2.cvtColor(mid[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                snapshots.append({"idx": idx, "jpeg": frame_to_jpeg_bytes(crop_rgb),
                                   "bbox": (xmin, ymin, bw, bh), "manual": False})
            st.session_state.face_snapshots = snapshots
            if not snapshots:
                st.warning("No faces detected. Lower the confidence or use Manual Override below.")
            st.rerun()

    # ── Per-face assignment ───────────────────────────────────────────────────
    face_logo_map = {}   # snap_idx → logo_index (0-based)

    if st.session_state.face_snapshots:
        n_logos = len(logo_files) if logo_files else 0
        logo_options = ["— no logo —"] + [f"Logo {i+1}" for i in range(n_logos)]
        auto_snaps = [s for s in st.session_state.face_snapshots if not s["manual"]]
        manual_snaps = [s for s in st.session_state.face_snapshots if s["manual"]]

        if auto_snaps:
            st.markdown(f"**{len(auto_snaps)} auto-detected face(s):**")
            cols = st.columns(min(len(auto_snaps), 4))
            for snap in auto_snaps:
                fi = snap["idx"]
                with cols[fi % len(cols)]:
                    st.markdown('<div class="face-card">', unsafe_allow_html=True)
                    st.image(snap["jpeg"], caption=f"Face {fi+1}", use_container_width=True)
                    choice = st.selectbox(f"Logo for face {fi+1}", logo_options, key=f"lc_{fi}")
                    if choice != "— no logo —":
                        face_logo_map[fi] = int(choice.split()[-1]) - 1
                    st.markdown('</div>', unsafe_allow_html=True)

        if manual_snaps:
            st.markdown("**Manual overrides:**")
            mcols = st.columns(min(len(manual_snaps), 4))
            for snap in manual_snaps:
                fi = snap["idx"]
                with mcols[(fi - len(auto_snaps)) % len(mcols)]:
                    st.markdown('<div class="manual-card">', unsafe_allow_html=True)
                    st.caption(f"Manual #{fi+1} — bbox {tuple(round(v,2) for v in snap['bbox'])}")
                    choice = st.selectbox(f"Logo for manual {fi+1}", logo_options, key=f"lc_{fi}")
                    if choice != "— no logo —":
                        face_logo_map[fi] = int(choice.split()[-1]) - 1
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Manual Override (for CGI / undetected faces) ──────────────────────────
    with st.expander("➕ Manual Override — add a face bbox for CGI or missed faces"):
        st.caption("Enter normalised coordinates (0.0–1.0 relative to frame size). "
                   "Use the frame preview above to estimate position.")
        mo_col = st.columns(4)
        with mo_col[0]: mo_x = st.number_input("X (left edge)", 0.0, 1.0, 0.1, 0.01, key="mo_x")
        with mo_col[1]: mo_y = st.number_input("Y (top edge)",  0.0, 1.0, 0.1, 0.01, key="mo_y")
        with mo_col[2]: mo_w = st.number_input("Width",         0.01, 1.0, 0.2, 0.01, key="mo_w")
        with mo_col[3]: mo_h = st.number_input("Height",        0.01, 1.0, 0.2, 0.01, key="mo_h")

        # live preview with rectangle drawn
        preview_frame = grab_frame_at(st.session_state.shorts_path, scan_frame_idx)
        if preview_frame is not None:
            pf = preview_frame.copy()
            pfh, pfw = pf.shape[:2]
            rx = int(mo_x * pfw);  ry = int(mo_y * pfh)
            rw = int(mo_w * pfw);  rh = int(mo_h * pfh)
            cv2.rectangle(pf, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 3)
            # also draw existing auto detections
            for s in st.session_state.face_snapshots:
                bx,by,bw2,bh2 = s["bbox"]
                cv2.rectangle(pf,
                    (int(bx*pfw), int(by*pfh)),
                    (int((bx+bw2)*pfw), int((by+bh2)*pfh)),
                    (0,255,0) if not s["manual"] else (255,165,0), 2)
            # downscale for display
            disp_h = 400
            disp_w = int(pfw * disp_h / pfh)
            st.image(cv2.cvtColor(cv2.resize(pf,(disp_w,disp_h)), cv2.COLOR_BGR2RGB),
                     caption="Yellow = your manual bbox | Green = auto | Orange = previous manual")

        if st.button("Add this manual bbox", use_container_width=True):
            existing = st.session_state.face_snapshots
            new_idx = len(existing)
            existing.append({
                "idx": new_idx,
                "jpeg": b"",
                "bbox": (mo_x, mo_y, mo_w, mo_h),
                "manual": True
            })
            st.session_state.face_snapshots = existing
            st.success(f"Manual bbox #{new_idx+1} added.")
            st.rerun()

    # ── Render ────────────────────────────────────────────────────────────────
    if (st.session_state.face_snapshots and logo_files and
            st.button("🚀 Render final video", use_container_width=True, type="primary")):

        if not face_logo_map:
            st.warning("Assign at least one logo to a face before rendering.")
        else:
            logos_bgra = []
            for lf in logo_files:
                pil = Image.open(lf).convert("RGBA")
                logos_bgra.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA))

            cap   = cv2.VideoCapture(st.session_state.shorts_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30
            fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Write video-only first, then mux audio with ffmpeg
            vid_only = tempfile.NamedTemporaryFile(delete=False, suffix="_noaudio.mp4")
            writer   = cv2.VideoWriter(vid_only.name, cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (fw, fh))

            prog    = st.progress(0, text="Rendering frames…")
            preview = st.empty()

            snap_bboxes = [s["bbox"] for s in st.session_state.face_snapshots]
            last_positions = {}   # snap_idx → last known bbox (for fallback)

            with mp_face_det.FaceDetection(model_selection=1,
                                           min_detection_confidence=confidence) as det:
                for fi in range(total):
                    ret, frame = cap.read()
                    if not ret: break

                    # Auto-detected faces
                    auto_faces = detect_faces_in_frame(frame, det)

                    assigned_snaps = set()
                    for det_bbox in auto_faces:
                        dx = det_bbox[0] + det_bbox[2] / 2
                        dy = det_bbox[1] + det_bbox[3] / 2
                        # find closest auto snap
                        auto_snap_indices = [s["idx"] for s in st.session_state.face_snapshots
                                             if not s["manual"]]
                        if not auto_snap_indices:
                            continue
                        best = min(
                            auto_snap_indices,
                            key=lambda i: abs(snap_bboxes[i][0]+snap_bboxes[i][2]/2 - dx)
                                        + abs(snap_bboxes[i][1]+snap_bboxes[i][3]/2 - dy)
                        )
                        if best in face_logo_map and best not in assigned_snaps:
                            last_positions[best] = det_bbox
                            assigned_snaps.add(best)
                            frame = overlay_logo_on_frame(
                                frame, logos_bgra[face_logo_map[best]], det_bbox, logo_scale)

                    # Manual overrides — always draw at fixed bbox (no tracking needed)
                    for snap in st.session_state.face_snapshots:
                        if snap["manual"] and snap["idx"] in face_logo_map:
                            frame = overlay_logo_on_frame(
                                frame, logos_bgra[face_logo_map[snap["idx"]]],
                                snap["bbox"], logo_scale)

                    writer.write(frame)
                    prog.progress((fi+1)/total, text=f"Rendering… {fi+1}/{total}")
                    if fi % 20 == 0:
                        preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                      channels="RGB", use_container_width=True)

            cap.release(); writer.release()
            prog.empty(); preview.empty()

            # Mux original audio back in
            final_out = tempfile.NamedTemporaryFile(delete=False, suffix="_final.mp4")
            with st.spinner("Adding original audio…"):
                ok, err = mux_audio(vid_only.name, st.session_state.shorts_path,
                                    final_out.name)

            if not ok:
                st.warning(f"Audio mux had an issue (video still works): {err[:200]}")
                final_path = vid_only.name
            else:
                final_path = final_out.name

            st.success("✅ Done! Download your video with audio below.")
            with open(final_path, "rb") as f:
                st.download_button(
                    "⬇️ Download final video (with audio)",
                    data=f,
                    file_name="shorts_face_logos.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
