from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import tempfile
import requests

app = Flask(__name__)

# دالة لتنزيل الملفات من GitHub
def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved to: {save_path}")
    else:
        print(f"Failed to download file from: {url}")

# تنزيل النماذج من GitHub
model_urls = {
    "Hieroglyph": "https://github.com/MohamedGmee/Project-graduation/raw/main/Keywords.pt",
    "Attractions": "https://github.com/MohamedGmee/Project-graduation/raw/main/Egypt%20Attractions.pt",
    "Landmarks": "https://github.com/MohamedGmee/Project-graduation/raw/main/Landmark%20Object%20detection.pt",
    "Hieroglyph Net": "https://github.com/MohamedGmee/Project-graduation/raw/main/best_tourgiude.pt"
}

model_paths = {}
for name, url in model_urls.items():
    save_path = f"{name}.pt"
    download_file(url, save_path)
    model_paths[name] = save_path

# تحميل نماذج YOLO
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model from {model_path}: {e}")
        return None

models = {name: load_yolo_model(path) for name, path in model_paths.items()}

# دالة الاستدلال
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
                        draw.text((x1, y1 - 10), cls_name, fill="white")  # تمت إزالة font=font

    return detected_classes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    file.save(temp_file.name)
    image_path = temp_file.name

    try:
        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)
        processed_classes = set()
        all_results = []

        for name, model in models.items():
            if model:
                all_results.extend(run_inference(model, original_image, draw, processed_classes))

        final_annotated_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        original_image.save(final_annotated_path)

        original_image.close()  # إغلاق الصورة

        try:
            os.remove(image_path)  # حذف الملف المؤقت
        except PermissionError:
            print(f"Could not delete file: {image_path} because it is in use.")

        return jsonify({"detected_classes": list(set(all_results)), "annotated_image": final_annotated_path})

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
