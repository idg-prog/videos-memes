import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess
import base64
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Keyframe Logo Tracker", layout="wide", page_icon="🎭")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #6C63FF; color: white; font-weight: bold; }
    .step-header { background: #1E1E1E; color: #6C63FF; padding: 15px; border-radius: 10px; font-weight: bold; font-size: 20px; margin: 10px 0; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "src_path" not in st.session_state:
    st.session_state.update({
        "src_path": None, "shorts_path": None, "fps": 30.0,
        "keyframes": [], # Stores: {frame_idx: [{bbox, logo_idx}]}
        "logos": []
    })

# ── HELPERS ───────────────────────────────────────────────────────────────────
def frame_to_base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def mux_audio(video_no_audio, original_with_audio, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_no_audio, "-i", original_with_audio,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-c:a", "aac", "-shortest", out_path
    ], capture_output=True)

# ── STEP 1: UPLOAD & FORMAT ──────────────────────────────────────────────────
st.markdown('<div class="step-header">Step 1: Video & Logos</div>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])

with c1:
    v_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if v_file and st.session_state.src_path is None:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t.write(v_file.read())
        st.session_state.src_path = t.name
        cap = cv2.VideoCapture(t.name)
        st.session_state.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

with c2:
    logo_files = st.file_uploader("Upload Logos (PNG)", accept_multiple_files=True)
    if logo_files:
        st.session_state.logos = []
        for lf in logo_files:
            img = Image.open(lf).convert("RGBA")
            st.session_state.logos.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA))

if st.session_state.src_path and st.button("📱 Convert to Shorts (9:16 Vertical)"):
    with st.spinner("Preparing Vertical Video..."):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        subprocess.run(["ffmpeg", "-y", "-i", st.session_state.src_path, "-vf", "scale=-1:1920,crop=1080:1920", "-c:a", "aac", out.name])
        st.session_state.shorts_path = out.name
        st.rerun()

# ── STEP 2: TIMELINE & DRAWING ───────────────────────────────────────────────
if st.session_state.shorts_path:
    st.markdown('<div class="step-header">Step 2: Frame-by-Frame Logo Placement</div>', unsafe_allow_html=True)
    
    cap = cv2.VideoCapture(st.session_state.shorts_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Timeline scrubbing
    frame_idx = st.slider("Scrub Video (Frame)", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        h, w = frame.shape[:2]
        display_width = 700
        display_height = int(display_width * (h/w))
        
        st.write(f"Showing Frame: **{frame_idx}**. Draw boxes on all faces visible here.")
        
        # DRAWING CANVAS
        canvas_result = st_canvas(
            fill_color="rgba(108, 99, 255, 0.3)",
            stroke_width=2,
            stroke_color="#6C63FF",
            background_image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key=f"canvas_{frame_idx}", # Unique key per frame to refresh properly
        )

        if canvas_result.json_data is not None:
            objs = canvas_result.json_data["objects"]
            if len(objs) > 0 and st.session_state.logos:
                st.write(f"Link these {len(objs)} boxes to logos for this frame:")
                
                current_frame_data = []
                cols = st.columns(len(objs))
                for i, obj in enumerate(objs):
                    with cols[i]:
                        l_idx = st.selectbox(f"Box {i+1} Logo", range(len(st.session_state.logos)), format_func=lambda x: f"Logo {x+1}", key=f"sel_{frame_idx}_{i}")
                        
                        # Store normalized coordinates
                        current_frame_data.append({
                            "bbox": [obj['left']/display_width, obj['top']/display_height, obj['width']/display_width, obj['height']/display_height],
                            "logo_idx": l_idx
                        })
                
                if st.button("💾 Save Placements for this Frame"):
                    # Remove existing keyframe for this index if it exists
                    st.session_state.keyframes = [k for k in st.session_state.keyframes if k['frame'] != frame_idx]
                    st.session_state.keyframes.append({"frame": frame_idx, "data": current_frame_data})
                    st.success(f"Saved boxes for frame {frame_idx}!")

    # ── STEP 3: RENDER ────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Step 3: Final Render</div>', unsafe_allow_html=True)
    st.write(f"Keyframes saved at: {[k['frame'] for k in st.session_state.keyframes]}")

    if len(st.session_state.keyframes) > 0 and st.button("🚀 TRACK & RENDER ALL"):
        # We use simple interpolation between keyframes for perfect control
        cap = cv2.VideoCapture(st.session_state.shorts_path)
        out_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        writer = cv2.VideoWriter(out_v.name, cv2.VideoWriter_fourcc(*'mp4v'), st.session_state.fps, (w, h))
        
        prog = st.progress(0)
        
        # Sorting keyframes
        sorted_keys = sorted(st.session_state.keyframes, key=lambda x: x['frame'])
        
        for f_num in range(total_frames):
            ret, f = cap.read()
            if not ret: break
            
            # Find the active keyframe (the last one drawn)
            active_data = None
            for k in sorted_keys:
                if k['frame'] <= f_num:
                    active_data = k['data']
            
            if active_data:
                for item in active_data:
                    # Scale back to video size
                    bx, by, bw, bh = item['bbox']
                    real_bbox = [int(bx*w), int(by*h), int(bw*w), int(bh*h)]
                    
                    # Overlay
                    logo = st.session_state.logos[item['logo_idx']]
                    lx, ly, lw, lh = real_bbox
                    if lw > 0 and lh > 0:
                        try:
                            res_logo = cv2.resize(logo, (lw, lh))
                            if res_logo.shape[2] == 4:
                                alpha = res_logo[:,:,3:4]/255.0
                                f[ly:ly+lh, lx:lx+lw] = (res_logo[:,:,:3]*alpha + f[ly:ly+lh, lx:lx+lw]*(1-alpha)).astype(np.uint8)
                            else:
                                f[ly:ly+lh, lx:lx+lw] = res_logo[:,:,:3]
                        except: pass
            
            writer.write(f)
            if f_num % 10 == 0: prog.progress(f_num/total_frames)

        cap.release()
        writer.release()
        
        final_file = "puny_god_final.mp4"
        mux_audio(out_v.name, st.session_state.shorts_path, final_file)
        st.success("Done!")
        with open(final_file, "rb") as f:
            st.download_button("📥 Download Result", f, "final_video.mp4")
