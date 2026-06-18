import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess
from streamlit_drawable_canvas import st_canvas

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Keyframe Logo Tracker", layout="wide", page_icon="🎭")

# ── ROBUST SESSION STATE INITIALIZATION ──────────────────────────────────────
# This ensures keys exist even if the app reruns or crashes
if "keyframes" not in st.session_state:
    st.session_state.keyframes = []
if "src_path" not in st.session_state:
    st.session_state.src_path = None
if "shorts_path" not in st.session_state:
    st.session_state.shorts_path = None
if "logos" not in st.session_state:
    st.session_state.logos = []
if "fps" not in st.session_state:
    st.session_state.fps = 30.0

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #6C63FF; color: white; font-weight: bold; height: 3em;}
    .step-header { background: #1E1E1E; color: #6C63FF; padding: 15px; border-radius: 10px; font-weight: bold; font-size: 20px; margin: 10px 0; border: 1px solid #333; }
    .css-10trblm {color: #6C63FF !important;}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Pro Keyframe Logo Tracker")
st.caption("Instructions: 1. Upload Video & Logos. 2. Scrub to a frame and draw boxes. 3. Save that frame. 4. Move to the next frame where characters move and draw again.")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def mux_audio(video_no_audio, original_with_audio, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_no_audio, "-i", original_with_audio,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-c:a", "aac", "-shortest", out_path
    ], capture_output=True)

# ── STEP 1: UPLOAD ───────────────────────────────────────────────────────────
st.markdown('<div class="step-header">Step 1: Video & Logos</div>', unsafe_allow_html=True)
c1, c2 = st.columns([2, 1])

with c1:
    v_file = st.file_uploader("Upload Video (Hulk vs Loki)", type=["mp4", "mov", "avi"])
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
        temp_logos = []
        for lf in logo_files:
            img = Image.open(lf).convert("RGBA")
            temp_logos.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA))
        st.session_state.logos = temp_logos

if st.session_state.src_path and st.session_state.shorts_path is None:
    if st.button("📱 Prepare Vertical Video (Shorts)"):
        with st.spinner("Creating 9:16 vertical version..."):
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            # This crops the center of the video for TikTok style
            subprocess.run(["ffmpeg", "-y", "-i", st.session_state.src_path, "-vf", "scale=-1:1920,crop=1080:1920", "-c:a", "aac", out.name])
            st.session_state.shorts_path = out.name
            st.rerun()

# ── STEP 2: TIMELINE & DRAWING ───────────────────────────────────────────────
if st.session_state.shorts_path:
    st.markdown('<div class="step-header">Step 2: Draw Faces & Link Logos</div>', unsafe_allow_html=True)
    
    cap = cv2.VideoCapture(st.session_state.shorts_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Scrubbing bar
    frame_idx = st.slider("Select Frame to Edit", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        h, w, _ = frame.shape
        # Ensure image is in RGB for PIL/Canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bg_img = Image.fromarray(frame_rgb)
        
        display_width = 700
        display_height = int(display_width * (h/w))
        
        st.write(f"📍 Editing Frame: {frame_idx}")
        
        # KEY= is vital here. It forces the canvas to update when frame_idx changes.
        canvas_result = st_canvas(
            fill_color="rgba(108, 99, 255, 0.3)",
            stroke_width=2,
            stroke_color="#6C63FF",
            background_image=bg_img,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key=f"canvas_f{frame_idx}",
        )

        if canvas_result.json_data is not None:
            objs = canvas_result.json_data["objects"]
            if len(objs) > 0:
                if not st.session_state.logos:
                    st.error("⚠️ Please upload logos in Step 1 first!")
                else:
                    st.write(f"Found {len(objs)} boxes. Link them:")
                    current_data = []
                    cols = st.columns(len(objs))
                    for i, obj in enumerate(objs):
                        with cols[i]:
                            l_idx = st.selectbox(f"Box {i+1} Logo", range(len(st.session_state.logos)), format_func=lambda x: f"Logo {x+1}", key=f"sel_{frame_idx}_{i}")
                            current_data.append({
                                "bbox": [obj['left']/display_width, obj['top']/display_height, obj['width']/display_width, obj['height']/display_height],
                                "logo_idx": l_idx
                            })
                    
                    if st.button("💾 Save Logo Positions for this Frame"):
                        # Save keyframe
                        st.session_state.keyframes = [k for k in st.session_state.keyframes if k['frame'] != frame_idx]
                        st.session_state.keyframes.append({"frame": frame_idx, "data": current_data})
                        st.success(f"Frame {frame_idx} saved!")

    # ── STEP 3: RENDER ────────────────────────────────────────────────────────
    st.markdown('<div class="step-header">Step 3: Render Final Video</div>', unsafe_allow_html=True)
    
    saved_frames = sorted([k['frame'] for k in st.session_state.keyframes])
    st.write(f"Current saved keyframes: {saved_frames}")

    if len(st.session_state.keyframes) > 0:
        if st.button("🚀 START FULL RENDER", type="primary"):
            cap = cv2.VideoCapture(st.session_state.shorts_path)
            out_v = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer = cv2.VideoWriter(out_v.name, cv2.VideoWriter_fourcc(*'mp4v'), st.session_state.fps, (w, h))
            
            prog = st.progress(0)
            sorted_keys = sorted(st.session_state.keyframes, key=lambda x: x['frame'])
            
            for f_num in range(total_frames):
                ret, f = cap.read()
                if not ret: break
                
                # Logic: Use the logos from the most recent keyframe
                active_boxes = None
                for k in sorted_keys:
                    if k['frame'] <= f_num:
                        active_boxes = k['data']
                
                if active_boxes:
                    for box in active_boxes:
                        bx, by, bw, bh = box['bbox']
                        lx, ly, lw, lh = int(bx*w), int(by*h), int(bw*w), int(bh*h)
                        
                        logo = st.session_state.logos[box['logo_idx']]
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
                if f_num % 20 == 0:
                    prog.progress(f_num/total_frames)

            cap.release()
            writer.release()
            
            final_out = "hulk_loki_logo.mp4"
            mux_audio(out_v.name, st.session_state.shorts_path, final_out)
            st.success("Rendering Complete!")
            with open(final_out, "rb") as f:
                st.download_button("📥 Download Result", f, "tracked_video.mp4")
