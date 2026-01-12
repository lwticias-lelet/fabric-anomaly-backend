import io
import os
import base64
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ==============================
# CONFIGURAÇÃO DO YOLO
# ==============================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

DEVICE = "cpu"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

print("🔹 Carregando modelo YOLO...")
model = YOLO(MODEL_PATH)

# ==============================
# FASTAPI
# ==============================

app = FastAPI(title="Fabric Anomaly Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Vercel
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# ==============================
# PRÉ-PROCESSAMENTO
# ==============================

def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    """Leve realce sem pesar CPU"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# ==============================
# HEURÍSTICA (FALLBACK)
# ==============================

def heuristic_boxes(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w

    boxes = []
    for c in contours:
        if cv2.contourArea(c) > 0.02 * img_area:
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append((x, y, x + cw, y + ch))

    return boxes

# ==============================
# INFERÊNCIA
# ==============================

def run_inference(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    img_bgr = enhance_image(img_bgr)

    has_defect = False
    used_fallback = False

    # --------------------------
    # YOLO
    # --------------------------
    results = model(
        img_np,
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        has_defect = True
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                img_bgr,
                f"{conf:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # --------------------------
    # HEURÍSTICA
    # --------------------------
    if not has_defect:
        fallback_boxes = heuristic_boxes(img_bgr)
        if fallback_boxes:
            has_defect = True
            used_fallback = True

            for x1, y1, x2, y2 in fallback_boxes:
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # --------------------------
    # OUTPUT
    # --------------------------
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(img_rgb)

    buffer = io.BytesIO()
    out_img.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return has_defect, img_b64, used_fallback

# ==============================
# ENDPOINT
# ==============================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    try:
        content = await file.read()
        has_defect, img_b64, fallback = run_inference(content)

        if has_defect:
            msg = "Defeito detectado no tecido."
            if fallback:
                msg += " (análise heurística)"
        else:
            msg = "Nenhum defeito visível detectado."

        return {
            "has_defect": has_defect,
            "message": msg,
            "image_base64": img_b64
        }

    except Exception as e:
        print("❌ ERRO:", e)
        return {
            "has_defect": False,
            "message": "Erro interno no processamento.",
            "image_base64": None
        }
