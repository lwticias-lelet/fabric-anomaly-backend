# backend/main.py

import io
import os
import base64
from typing import Tuple, List, Literal

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

print(f"Carregando modelo YOLO em {DEVICE} a partir de {MODEL_PATH}...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo de modelo não encontrado em: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
model.to(DEVICE)

# limiares do YOLO (deixamos mais sensível)
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.5

# ================================
# FASTAPI APP
# ================================

app = FastAPI(title="Fabric Anomaly Scanner - YOLO + Heurística")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # em produção você pode restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """
    Rota raiz, usada pelo Render/Vercel para testar se o serviço está no ar.
    """
    return {
        "status": "ok",
        "message": "Backend Fabric Anomaly Scanner (YOLO) rodando. Use /health ou POST /scan."
    }


@app.get("/health")
async def health():
    """
    Endpoint simples de saúde da aplicação.
    """
    return {"status": "ok"}


# ================================
# PRÉ-PROCESSAMENTO E HEURÍSTICA
# ================================

def enhance_for_yolo(img_bgr: np.ndarray) -> np.ndarray:
    """
    Realce leve de contraste + redução de ruído
    para facilitar a detecção do YOLO.
    """
    # Reduz ruído preservando bordas
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Equalização adaptativa no canal de luminosidade
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced


def heuristic_large_defect_boxes(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Fallback de backup:
    detecta contornos grandes (rasgos/manchas marcantes)
    e devolve bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecção de bordas
    edges = cv2.Canny(gray, 40, 120)

    # Dilatar bordas para unir áreas
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        # considera apenas contornos que ocupam pelo menos 1% da imagem
        if area > 0.01 * img_area:
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append((x, y, cw, ch))

    return boxes


# ================================
# FUNÇÃO AUXILIAR DE INFERÊNCIA
# ================================

def run_inference(image_bytes: bytes) -> Tuple[bool, str, List[dict], Literal["yolo", "heuristic", "none"]]:
    """
    Roda o modelo YOLO + heurística de backup e devolve:
      - has_defect: bool
      - image_b64: imagem anotada (com bounding boxes) em base64
      - detections: lista com boxes e origem
      - source: "yolo" | "heuristic" | "none"
    """

    # Abrir imagem com PIL (RGB)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)  # RGB

    # Converter para BGR para usar no OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Pré-processar imagem para ajudar o YOLO
    img_bgr_enh = enhance_for_yolo(img_bgr)
    img_rgb_enh = cv2.cvtColor(img_bgr_enh, cv2.COLOR_BGR2RGB)

    # ------------------------------
    # 1) YOLO
    # ------------------------------
    results = model.predict(
        source=img_rgb_enh,
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        agnostic_nms=True,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    has_defect = False
    detections_info: List[dict] = []
    source: Literal["yolo", "heuristic", "none"] = "none"

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
                    "src": "yolo",
                }
            )

        # Se qualquer box tiver confiança >= limiar, consideramos defeito
        if any(det["conf"] >= CONF_THRESHOLD for det in detections_info):
            has_defect = True
            source = "yolo"

            # Desenhar boxes do YOLO
            for det in detections_info:
                x1 = int(det["x1"])
                y1 = int(det["y1"])
                x2 = int(det["x2"])
                y2 = int(det["y2"])

                cv2.rectangle(
                    img_bgr_enh,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),  # vermelho BGR
                    3
                )

                cv2.putText(
                    img_bgr_enh,
                    f"{det['conf']:.2f}",
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

    # ------------------------------
    # 2) FALLBACK HEURÍSTICO
    # ------------------------------
    if not has_defect:
        big_boxes = heuristic_large_defect_boxes(img_bgr_enh)

        if big_boxes:
            has_defect = True
            source = "heuristic"

            for (x, y, cw, ch) in big_boxes:
                x1, y1, x2, y2 = x, y, x + cw, y + ch
                detections_info.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": None,
                        "src": "heuristic",
                    }
                )

                cv2.rectangle(
                    img_bgr_enh,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

    # Converter de volta para RGB antes de salvar
    img_rgb_out = cv2.cvtColor(img_bgr_enh, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb_out)

    # Converter para JPEG em memória
    buffer = io.BytesIO()
    pil_out.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    img_bytes = buffer.read()

    # Converter para base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return has_defect, img_b64, detections_info, source


# ================================
# ENDPOINT PRINCIPAL /scan
# ================================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    Recebe uma imagem (frame da câmera), roda YOLO + heurística e devolve:
      - has_defect: bool
      - message: texto em PT-BR
      - image_base64: imagem anotada (com retângulos vermelhos)
      - source: "yolo" | "heuristic" | "none"
      - detections: lista de boxes (opcional para debug)
    """
    try:
        content = await file.read()
        has_defect, img_b64, detections, source = run_inference(content)

        if has_defect:
            if source == "yolo":
                msg = "Defeito detectado no tecido (YOLO)."
            elif source == "heuristic":
                msg = "Possível defeito detectado no tecido (análise de bordas)."
            else:
                msg = "Defeito detectado no tecido."
        else:
            msg = "Nenhum defeito visível detectado."

        return {
            "has_defect": has_defect,
            "message": msg,
            "image_base64": img_b64,
            "source": source,
            "detections": detections,
        }

    except Exception as e:
        return {
            "has_defect": False,
            "message": f"Erro ao processar imagem: {str(e)}",
            "image_base64": None,
            "source": "error",
            "detections": [],
        }
