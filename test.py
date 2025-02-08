from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import tempfile
from io import BytesIO
import base64
import zlib
from typing import List

app = FastAPI()

# Load YOLO models
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        return None

# Model paths and loading
model_paths = {
    "Hieroglyph": r"https://github.com/MohamedGmee/Project-graduation/blob/main/Keywords.pt",
    "Attractions": rhttps://github.com/MohamedGmee/Project-graduation/blob/main/Egypt%20Attractions.pt",
    "Landmarks": r"https://github.com/MohamedGmee/Project-graduation/blob/main/Landmark%20Object%20detection.pt",
    "Hieroglyph Net": r"https://github.com/MohamedGmee/Project-graduation/blob/main/best_tourgiude.pt"
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


@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Only images are allowed.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(await file.read())
    temp_file.close()
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

        # Reduce image size and quality before converting to WebP
        original_image.thumbnail((200, 200))  # Resize to 200x200 pixels
        img_byte_arr = BytesIO()
        original_image.save(img_byte_arr, format="WebP", quality=50)  # Save as WebP with reduced quality
        img_byte_arr.seek(0)

        # Compress the image before converting to base64
        compressed_image = zlib.compress(img_byte_arr.read())

        # Convert the image to base64
        img_base64 = base64.b64encode(compressed_image).decode('utf-8')

        # Return results
        return {
            "detected_classes": list(set(all_results)),
            "annotated_image_base64": img_base64
        }

    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary image file
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"Error cleaning up temporary file {image_path}: {e}")
