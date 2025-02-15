from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import tempfile
from io import BytesIO
import base64
from typing import List

app = FastAPI()

MODEL_DIR = "models" 

model_paths = {
    "Hieroglyph": os.path.join(MODEL_DIR, "https://github.com/MohamedGmee/Project-graduation/blob/main/Keywords.pt"),
    "Attractions": os.path.join(MODEL_DIR,  "https://github.com/MohamedGmee/Project-graduation/blob/main/Egypt%20Attractions.pt"),
    "Landmarks": os.path.join(MODEL_DIR, "https://github.com/MohamedGmee/Project-graduation/blob/main/Landmark%20Object%20detection.pt"),
    "Hieroglyph Net": os.path.join(MODEL_DIR, "https://github.com/MohamedGmee/Project-graduation/blob/main/best_tourgiude.pt"),
}
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return None

models = {name: load_yolo_model(path) for name, path in model_paths.items()}

@app.get("/")
def home():
    return {"message": "FastAPI Server is running! Use /docs to test the API."}

def run_inference(model, image, draw, processed_classes):
    results = model.predict(source=image, save=False)
    detected_classes = []

    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if hasattr(box, 'cls') and hasattr(box, 'xyxy'):
                    cls_id = int(box.cls)
                    cls_name = model.names.get(cls_id, f"Class_{cls_id}") if model.names else f"Class_{cls_id}"
                    if cls_name not in processed_classes:
                        detected_classes.append(cls_name)
                        processed_classes.add(cls_name)
                        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                        draw.text((x1, y1 - 10), cls_name, fill="white")

    return detected_classes

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only images are allowed.")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        temp_file.flush()
        image_path = temp_file.name

        try:
            original_image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(original_image)
            processed_classes = set()
            all_results = []

            for name, model in models.items():
                if model:
                    all_results.extend(run_inference(model, original_image, draw, processed_classes))

            # ✅ تحسين الصورة وتحويلها إلى Base64
            original_image.thumbnail((500, 500))  # تحديد حجم مناسب
            img_byte_arr = BytesIO()
            original_image.save(img_byte_arr, format="WebP", quality=80)
            img_byte_arr.seek(0)

            img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')

            return {
                "detected_classes": list(set(all_results)),
                "annotated_image_base64": img_base64
            }

        except Exception as e:
            print(f"❌ Error processing file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
