import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess

# Try to import canvas, handle error if missing
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Missing library: Run 'pip install streamlit-drawable-canvas'")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pro Face/Object Tracker", layout="wide", page_icon="🎭")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.step-wrap {
    border: 1.5px solid #E5E7EB; border-radius: 14px;
    padding: 22px 26px 18px; margin-bottom: 28px; background: #FAFAFA;
}
.step-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.step-num {
    background: #6C63FF; color: white; border-radius: 50%;
    width: 30px; height: 30px; display: flex; align-items: center;
    justify-content: center; font-weight: 700; font-size: 15px; flex-shrink: 0;
}
.step-title { font-size: 18px; font-weight: 700; color: #111; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎭 Pro Object & Face Tracker")
st.caption("Draw boxes on Hulk, Loki, or CGI characters → Auto-track → Render with Audio.")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "src_path" not in st.session_state:
    st.session_state.update({
        "src_path": None, "shorts_path": None, "fps": 30.0, "width": 0, "height": 0
    })

# ── HELPERS ───────────────────────────────────────────────────────────────────
def mux_audio(video_no_audio, original_with_audio, out_path):
    # -c:v copy avoids re-encoding the video again
    cmd = [
        "ffmpeg", "-y", "-i", video_no_audio, "-i", original_with_audio,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", out_path
    ]
    subprocess.run(cmd, capture_output=True)

def overlay_logo(frame, logo_bgra, bbox):
    x, y, w, h = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, fw - x), min(h, fh - y)
    if w <= 10 or h <= 10: return frame
    
    try:
        resized = cv2.resize(logo_bgra, (w, h), interpolation=cv2.INTER_AREA)
        if resized.shape[2] == 4:
            alpha = resized[:, :, 3:4] / 255.0
            frame[y:y+h, x:x+w] = (resized[:,:,:3] * alpha + frame[y:y+h, x:x+w] * (1 - alpha)).astype(np.uint8)
        else:
            frame[y:y+h, x:x+w] = resized[:,:,:3]
    except: pass
    return frame

# ── STEP 1: UPLOAD ───────────────────────────────────────────────────────────
st.markdown('<div class="step-wrap"><div class="step-header"><div class="step-num">1</div><div class="step-title">Upload & Format</div></div></div>', unsafe_allow_html=True)

v_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
if v_file:
    if st.session_state.src_path is None:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t.write(v_file.read())
        st.session_state.src_path = t.name
        cap = cv2.VideoCapture(t.name)
        st.session_state.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    if st.button("📱 Convert to Shorts (9:16 Center Crop)"):
        with st.spinner("Reformatting..."):
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            vf = "scale=-1:1920,crop=1080:1920"
            subprocess.run(["ffmpeg", "-y", "-i", st.session_state.src_path, "-vf", vf, "-c:a", "aac", out.name])
            st.session_state.shorts_path = out.name
            st.success("Shorts Ready!")

# ── STEP 2: DRAWING ──────────────────────────────────────────────────────────
if st.session_state.shorts_path:
    st.markdown('<div class="step-wrap"><div class="step-header"><div class="step-num">2</div><div class="step-title">Draw on Hulk/Loki/CGI</div></div></div>', unsafe_allow_html=True)
    
    logo_files = st.file_uploader("Upload Logos (PNG)", accept_multiple_files=True)
    logos = []
    for lf in logo_files:
        img = Image.open(lf).convert("RGBA")
        logos.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA))

    cap = cv2.VideoCapture(st.session_state.shorts_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        h, w = frame.shape[:2]
        # FIX: Ensure image is RGB and converted to a PIL object to avoid AttributeError
        bg_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        display_width = 700
        display_height = int(display_width * (h/w))
        
        st.write("Drag your mouse to draw boxes over characters:")
        canvas_result = st_canvas(
            fill_color="rgba(108, 99, 255, 0.3)",
            stroke_width=2,
            stroke_color="#6C63FF",
            background_image=bg_img, # Pass PIL object directly
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="canvas_main",
        )

        if canvas_result.json_data is not None:
            objs = canvas_result.json_data["objects"]
            if len(objs) > 0 and logos:
                st.info(f"Targets detected: {len(objs)}. Assign logos:")
                mapping = {}
                cols = st.columns(len(objs))
                for i, obj in enumerate(objs):
                    with cols[i]:
                        mapping[i] = st.selectbox(f"Target {i+1}", range(len(logos)), format_func=lambda x: f"Logo {x+1}", key=f"map_{i}")

                if st.button("🚀 TRACK & RENDER", type="primary"):
                    # Initialize Trackers
                    trackers = []
                    for obj in objs:
                        # Try CSRT (High Quality), Fallback to KCF (Faster) if fails
                        try:
                            tracker = cv2.TrackerCSRT_create()
                        except AttributeError:
                            tracker = cv2.TrackerKCF_create()
                            
                        scale_x = w / display_width
                        scale_y = h / display_height
                        bbox = (
                            int(obj['left'] * scale_x),
                            int(obj['top'] * scale_y),
                            int(obj['width'] * scale_x),
                            int(obj['height'] * scale_y)
                        )
                        tracker.init(frame, bbox)
                        trackers.append(tracker)

                    # Process Video
                    cap = cv2.VideoCapture(st.session_state.shorts_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    writer = cv2.VideoWriter(temp_vid.name, cv2.VideoWriter_fourcc(*'mp4v'), st.session_state.fps, (w, h))
                    
                    prog = st.progress(0)
                    curr = 0
                    while cap.isOpened():
                        ret, f = cap.read()
                        if not ret: break
                        
                        for i, tracker in enumerate(trackers):
                            ok, bbox = tracker.update(f)
                            if ok:
                                f = overlay_logo(f, logos[mapping[i]], bbox)
                        
                        writer.write(f)
                        curr += 1
                        if curr % 15 == 0:
                            prog.progress(curr/total_frames)
                    
                    cap.release()
                    writer.release()
                    
                    # Merge Audio
                    final_path = "final_output.mp4"
                    mux_audio(temp_vid.name, st.session_state.shorts_path, final_path)
                    
                    st.success("✅ Tracking Finished!")
                    with open(final_path, "rb") as file:
                        st.download_button("⬇️ Download Result", file, "puny_god_tracked.mp4", use_container_width=True)
            elif len(objs) > 0 and not logos:
                st.warning("Please upload at least one logo image.")
