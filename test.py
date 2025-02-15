# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 00:41:57 2024

@author: Amany
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os

# Load Models
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        # Load the YOLO model (update repository as needed)
        model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=True)
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

    model.eval()  # Set the model to evaluation mode
    return model

models = {
    "Hieroglyph": "https://github.com/MohamedGmee/Project-graduation/blob/main/Keywords.pt",
    "Attractions": "https://github.com/MohamedGmee/Project-graduation/blob/main/Egypt%20Attractions.pt",
    "Landmarks": "https://github.com/MohamedGmee/Project-graduation/blob/main/Landmark%20Object%20detection.pt",
    "Hieroglyph Net": "https://github.com/MohamedGmee/Project-graduation/blob/main/best_tourgiude.pt"
}

# Streamlit UI
st.title("Image Classification and Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Initialize results array
    detected_classes = []

    # Process the image with all models
    for model_name, model in models.items():
        st.write(f"Processing with {model_name}...")
        try:
            # Perform inference
            results = model(image_cv)  # Use the model's predict method

            for result in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = result
                class_name = results.names[int(cls)]
                detected_classes.append(class_name)

                # Draw bounding box and label
                cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image_cv, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            st.error(f"Error processing with {model_name}: {e}")

    # Display results
    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Detected Classes", use_column_width=True)
    st.write("Detected Classes:", detected_classes)
