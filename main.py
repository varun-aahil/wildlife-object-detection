import io
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from ultralytics import YOLO

model = YOLO("runs/detect/wildlife_n/weights/best.pt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/info")
def info():
    return {"message": "wildlife detector up", "classes": list(model.names.values())}


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")
    if not 0 < conf < 1:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1")

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    start = time.time()
    result = model.predict(img, conf=conf, verbose=False)[0]
    elapsed = round((time.time() - start) * 1000, 1)

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "box": [round(x1), round(y1), round(x2), round(y2)],
        })

    return {
        "image_width": img.width,
        "image_height": img.height,
        "inference_ms": elapsed,
        "detections": detections,
    }