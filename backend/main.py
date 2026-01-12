

import io
import os
import base64
from typing import Tuple, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ================================
# CONFIGURAÇÃO DO MODELO YOLO
# ================================

# Caminho relativo ao backend/
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best.pt"
)

DEVICE = "cpu"

print(f"[BOOT] Carregando modelo YOLO em {DEVICE} a partir de {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo de modelo não encontrado em: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
model.to(DEVICE)

# Limiar de decisão (bem sensível)
CONF_THRESHOLD = 0.08   # antes estava ~0.35
IOU_THRESHOLD = 0.3


# ================================
# FASTAPI APP
# ================================

app = FastAPI(title="Fabric Anomaly Scanner - YOLO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # em produção pode restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Backend Fabric Anomaly Scanner (YOLO) rodando. Use /health ou POST /scan."
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ================================
# FUNÇÃO AUXILIAR DE INFERÊNCIA
# ================================

def run_inference(image_bytes: bytes) -> Tuple[bool, str, List[dict]]:
    """
    Roda o modelo YOLO na imagem enviada e devolve:
      - has_defect: bool
      - image_b64: imagem anotada (com bounding boxes)
      - detections_info: lista com info das detecções (p/ debug se quiser)
    """

    # Abrir imagem com PIL (RGB)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)  # RGB

    # Copiar para desenhar overlays usando OpenCV (BGR)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ------------------------------
    # YOLO PREDICT (mais sensível)
    # ------------------------------
    results = model.predict(
        source=img_np,
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    has_defect = False
    detections_info = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detections_info.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                }
            )

        # Se pelo menos UMA detecção com confiança >= CONF_THRESHOLD existir, consideramos defeito
        if any(det["conf"] >= CONF_THRESHOLD for det in detections_info):
            has_defect = True

            # Desenhar retângulos para todas as detecções válidas
            for det in detections_info:
                x1 = int(det["x1"])
                y1 = int(det["y1"])
                x2 = int(det["x2"])
                y2 = int(det["y2"])

                # Retângulo vermelho
                cv2.rectangle(
                    img_bgr,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),  # BGR (vermelho)
                    3
                )

                # Opcional: confiança
                label = f"{det['conf']:.2f}"
                cv2.putText(
                    img_bgr,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

    # Converter de volta para RGB antes de salvar
    img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb_out)

    buffer = io.BytesIO()
    pil_out.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    img_bytes_out = buffer.read()

    img_b64 = base64.b64encode(img_bytes_out).decode("utf-8")

    return has_defect, img_b64, detections_info


# ================================
# ENDPOINT PRINCIPAL /scan
# ================================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    Recebe uma imagem (frame da câmera), roda YOLO e devolve:
      - has_defect: bool
      - message: texto em PT-BR
      - image_base64: imagem anotada
      - debug (opcional): quantas detecções e confianças (ajuda pra testar)
    """
    try:
        content = await file.read()
        has_defect, img_b64, detections_info = run_inference(content)

        if has_defect:
            msg = "Defeito detectado no tecido."
        else:
            msg = "Nenhum defeito visível detectado."

        return {
            "has_defect": has_defect,
            "message": msg,
            "image_base64": img_b64,
            "detections": detections_info  # útil pra debugar no Insomnia/Postman
        }

    except Exception as e:
        return {
            "has_defect": False,
            "message": f"Erro ao processar imagem: {str(e)}",
            "image_base64": None,
            "detections": []
        }
