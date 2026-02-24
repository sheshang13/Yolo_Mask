import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="YOLO Face Mask Detection", layout="wide")
st.title("😷 Face Mask Detection using YOLO")
st.write("Upload an image or use webcam for detection")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    model = YOLO("best.pt")  # Make sure best.pt is in same folder
    return model

model = load_model()

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

source = st.sidebar.radio("Select Source", ["Image Upload", "Webcam"])

# -----------------------------
# IMAGE UPLOAD MODE
# -----------------------------
if source == "Image Upload":

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")   # ✅ FORCE 3 CHANNELS
        image_np = np.array(image)
    
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
        results = model(image_np, conf=confidence)
    
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detection Result", use_column_width=True)

        # Detection Details
        st.subheader("📋 Detection Details")
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = model.names[cls_id]
            st.write(f"Class: {class_name} | Confidence: {conf_score:.2f}")

# -----------------------------
# WEBCAM MODE
# -----------------------------
elif source == "Webcam":

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")


        cap.release()
