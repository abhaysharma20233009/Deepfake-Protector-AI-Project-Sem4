import streamlit as st
import os
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(page_title="DFDetect", layout="wide", initial_sidebar_state="expanded")

# Load Deepfake Detection Model
@st.cache_resource
def load_deepfake_model():
    model_path = "/content/drive/My Drive/myDataset/checkpoints/best_model.h5"  # Ensure this path is correct
    return load_model(model_path)

model = load_deepfake_model()

# Title and Description
st.title("🔍 DFDetect: Deepfake Detection")
st.markdown(
    """
    Welcome to **DFDetect**, an AI-powered tool for detecting deepfake content in images and videos.  
    Upload a file, and our model will analyze it to determine its authenticity. 🚀  
    """
)

# Sidebar Navigation
st.sidebar.header("Navigation")
st.sidebar.write("Use this panel to explore different features.")
st.sidebar.info("🔹 Supports **JPG, PNG, MP4** files.")

# File Upload
uploaded_file = st.file_uploader("📤 Upload an image or video for deepfake detection", type=["jpg", "png", "mp4"])

def preprocess_image(image, target_size=(128, 128)):
    """Resize and normalize image for model prediction."""
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    return image

if uploaded_file:
    file_type = uploaded_file.type

    if file_type in ["image/jpeg", "image/png"]:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Convert image for model prediction
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = preprocess_image(img)

        # Model Prediction
        try:
            prediction = model.predict(img)[0][0]
            label = "Real" if prediction > 0.5 else "Deepfake"
            
            # Display Results
            st.subheader("🔎 Detection Result")
            st.write(f"**Prediction:** {label} ({prediction * 100:.2f}%)")
            st.progress(float(prediction))  # Visualize confidence
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")

    elif file_type == "video/mp4":
        st.video(uploaded_file, format="video/mp4")
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        # Extract a frame for deepfake detection
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if success:
            frame = preprocess_image(frame)

            # Model Prediction
            try:
                prediction = model.predict(frame)[0][0]
                label = "Real" if prediction < 0.5 else "Deepfake"

                # Display Results
                st.subheader("🔎 Detection Result")
                st.write(f"**Prediction:** {label} ({prediction * 100:.2f}%)")
                st.progress(float(prediction))
            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
        else:
            st.error("❌ Unable to process video frame.")

        os.remove(video_path)  # Clean up temp file

st.sidebar.markdown("---")
st.sidebar.caption("🚀 Built with Streamlit | DFDetect 2025")