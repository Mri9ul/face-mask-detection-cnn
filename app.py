import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Face Mask Detector", page_icon="😷")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_detector.keras")


model = load_model()

st.title("😷 Face Mask Detector")
st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("### Analysis Results")

    # Convert PIL Image to numpy array (RGB format natively)
    img_array = np.array(image)

    # Preprocess
    img_resized = cv2.resize(img_array, (128, 128))
    img_scaled = img_resized / 255.0
    img_reshaped = np.reshape(img_scaled, [1, 128, 128, 3])

    # Predict
    prediction = model.predict(img_reshaped)
    pred_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    if pred_label == 1:
        st.success(f"✅ **Result: Wearing a Mask** (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"❌ **Result: NOT Wearing a Mask** (Confidence: {confidence:.2f}%)")