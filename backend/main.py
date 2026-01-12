from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import os

app = FastAPI()

# =======================
# CORS – libera pro frontend
# =======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois você pode limitar pro domínio da Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# HEALTH CHECK (teste rápido)
# =======================
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Backend vivo e respondendo."}

# =======================
# CARREGAR MODELO UMA VEZ
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Carregando modelo em {device} a partir de {MODEL_PATH}...")

model = YOLO(MODEL_PATH)  # YOLO v8
# vamos passar `device` na inferência, não precisa .to()

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def cv2_to_base64(img_bgr: np.ndarray) -> str:
    """OpenCV BGR -> JPEG base64."""
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("Falha ao codificar imagem em JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    try:
        # 1) Ler imagem enviada
        contents = await file.read()
        pil_img = Image.open(BytesIO(contents)).convert("RGB")
        img_bgr = pil_to_cv2(pil_img)

        # 2) Rodar YOLO
        results = model(
            img_bgr,
            imgsz=640,
            conf=0.35,
            iou=0.45,
            device=device,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes
        has_defect = len(boxes) > 0

        overlay = img_bgr.copy()

        if has_defect:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                # retângulo vermelho
                cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), 3)

                # rótulo opcional
                label = f"Defeito ({conf:.2f})"
                cv2.putText(
                    overlay,
                    label,
                    (pt1[0], max(pt1[1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            message = "Defeito detectado no tecido."
        else:
            message = "Nenhum defeito detectado."
            # opcional: moldura verde indicando ok
            h, w = overlay.shape[:2]
            cv2.rectangle(overlay, (10, 10), (w - 10, h - 10), (0, 255, 0), 3)
            cv2.putText(
                overlay,
                "SEM DEFEITO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # 3) Converter pra base64
        img_b64 = cv2_to_base64(overlay)

        return JSONResponse(
            {
                "has_defect": has_defect,
                "message": message,
                "image_base64": img_b64,
            }
        )

    except Exception as e:
        print("Erro em /scan:", e)
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )
