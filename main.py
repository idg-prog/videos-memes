import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="AI Face Logo Overlay", layout="wide")
st.title("🎭 Motion-Tracked Logo Overlay")
st.subheader("Upload a video and a logo to automatically track and pin the logo to faces")

# 1. Sidebar Inputs
st.sidebar.header("Upload Assets")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
logo_file = st.sidebar.file_uploader("Upload Logo (PNG with transparency recommended)", type=["png", "jpg", "jpeg"])

# Detection confidence configuration
min_detection_confidence = st.sidebar.slider("Face Detection Confidence", 0.0, 1.0, 0.5, 0.05)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

def overlay_logo(frame, logo, bbox):
    """Overlays the logo on the frame at the specified bounding box coordinates."""
    fh, fw, _ = frame.shape
    xmin, ymin, width, height = bbox
    
    # Convert normalized coordinates to pixel values
    x = int(xmin * fw)
    y = int(ymin * fh)
    w = int(width * fw)
    h = int(height * fh)
    
    # Boundary check to prevent crashing if the face is partially off-screen
    if x < 0 or y < 0 or x + w > fw or y + h > fh or w <= 0 or h <= 0:
        return frame

    # Resize logo to match the bounding box size of the face
    resized_logo = cv2.resize(logo, (w, h), interpolation=cv2.INTER_AREA)
    
    # Separate channels if PNG has transparency (Alpha channel)
    if resized_logo.shape[2] == 4:
        alpha = resized_logo[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha
        
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha * resized_logo[:, :, c] +
                                      alpha_background * frame[y:y+h, x:x+w, c])
    else:
        # If no alpha channel, just paste it over (hard boundary)
        frame[y:y+h, x:x+w] = resized_logo[:, :, :3]
        
    return frame

# 2. Main Processing Pipeline
if video_file and logo_file:
    # Read and prepare the logo
    pil_logo = Image.open(logo_file)
    logo_np = np.array(pil_logo)
    # Convert RGB/RGBA to BGR/BGRA for OpenCV processing
    logo_cv = cv2.cvtColor(logo_np, cv2.COLOR_RGBA2BGRA if logo_np.shape[-1] == 4 else cv2.COLOR_RGB2BGR)

    # Save uploaded video to a temporary local file so OpenCV can read it frame-by-frame
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Setup placeholder UI containers in Streamlit for dynamic streaming
    status_text = st.empty()
    video_placeholder = st.empty()
    
    status_text.text("Processing video frames...")
    
    # Run the tracking frame-by-frame
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_detection_confidence) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # MediaPipe expects RGB images, OpenCV defaults to BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            # If faces are detected, loop through them and apply the logo overlay
            if results.detections:
                for detection in results.detections:
                    bbox_data = detection.location_data.relative_bounding_box
                    bbox = (bbox_data.xmin, bbox_data.ymin, bbox_data.width, bbox_data.height)
                    frame = overlay_logo(frame, logo_cv, bbox)
            
            # Convert back to RGB for displaying correctly within Streamlit
            final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(final_frame, channels="RGB", use_container_width=True)
            
    cap.release()
    status_text.text("Processing Complete! 🎉")

else:
    st.info("Please upload both a video file and a logo image in the sidebar to get started.")
