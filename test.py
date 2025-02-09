import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import tempfile
import base64
import zlib
from io import BytesIO

# Load YOLO models
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model path does not exist: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        st.success(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model from {model_path}: {e}")
        return None

# Model paths and loading
model_paths = {
    "Hieroglyph": "Keywords.pt",
    "Attractions": "Egypt_Attractions.pt",
    "Landmarks": "Landmark_Object_detection.pt",
    "Hieroglyph Net": "best_tourguide.pt"
}

models = {name: load_yolo_model(path) for name, path in model_paths.items()}

# Inference function
def run_inference(model, image, draw, processed_classes):
    results = model.predict(source=image, save=False)
    detected_classes = []

    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if hasattr(box, 'cls') and hasattr(box, 'xyxy'):
                    cls_id = int(box.cls)
                    cls_name = model.names.get(cls_id, f"Class_{cls_id}")
                    if cls_name not in processed_classes:
                        detected_classes.append(cls_name)
                        processed_classes.add(cls_name)
                        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                        draw.text((x1, y1 - 10), cls_name, fill="white")

    return detected_classes

# Streamlit UI
st.title("YOLO Object Detection with Streamlit")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        image_path = temp_file.name

    try:
        # Open the image and prepare for annotation
        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)

        processed_classes = set()
        all_results = []

        # Run inference on all models
        for name, model in models.items():
            if model:
                all_results.extend(run_inference(model, original_image, draw, processed_classes))

        # Resize image for display
        original_image.thumbnail((400, 400))  # Resize to 400x400 pixels
        st.image(original_image, caption="Annotated Image", use_column_width=True)

        # Compress and encode the image
        img_byte_arr = BytesIO()
        original_image.save(img_byte_arr, format="WebP", quality=50)
        compressed_image = zlib.compress(img_byte_arr.getvalue())
        img_base64 = base64.b64encode(compressed_image).decode('utf-8')

        st.write("Detected Classes:", list(set(all_results)))
        st.download_button("Download Annotated Image", img_byte_arr.getvalue(), "annotated_image.webp", "image/webp")

    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        os.remove(image_path)
