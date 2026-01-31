from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
from io import BytesIO
from PIL import Image
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)


app = Flask(__name__)

# Load your trained model
model = YOLO("best.pt")
model.fuse()

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return "YOLOv8 Inference API is running!"


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image
    image = Image.open(file.stream).convert("RGB")
    img_np = np.array(image)

    # Run detection
    results = model(img_np, conf=0.4, device="cpu", half=False)

    # Draw bounding boxes
    annotated_img = results[0].plot()

    # Convert back to PIL
    annotated_pil = Image.fromarray(annotated_img)

    # Save to memory buffer
    buffer = BytesIO()
    annotated_pil.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype="image/png")
    
#pip install flask ultralytics opencv-python pillow numpy
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)