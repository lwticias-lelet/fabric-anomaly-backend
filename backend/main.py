# backend/main.py

import io
import os
import base64
from typing import Tuple

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

# ================================
# FASTAPI APP
# ================================

app = FastAPI(title="Fabric Anomaly Scanner - YOLO")

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
# FUNÇÃO AUXILIAR DE INFERÊNCIA
# ================================

def run_inference(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Roda o modelo YOLO na imagem enviada e devolve:
      - has_defect: bool
      - image_b64: imagem anotada (com bounding boxes) em base64
    """

    # Abrir imagem com PIL
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)  # RGB

    # Copiar para desenhar overlays usando OpenCV (BGR)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Rodar YOLO
    results = model(
        img_np,
        imgsz=640,
        conf=0.35,       # limiar de confiança
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    has_defect = False

    if boxes is not None and len(boxes) > 0:
        has_defect = True

        for box in boxes:
            # Coordenadas da caixa
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # Desenhar retângulo vermelho
            cv2.rectangle(
                img_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),    # BGR (vermelho)
                3               # espessura
            )

            # Opcional: escrever confiança acima da caixa
            label = f"{conf:.2f}"
            cv2.putText(
                img_bgr,
                label,
                (int(x1), max(int(y1) - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    # Converter de volta para RGB antes de salvar
    img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb_out)

    # Converter para JPEG em memória
    buffer = io.BytesIO()
    pil_out.save(buffer, format="JPEG")
    buffer.seek(0)
    img_bytes = buffer.read()

    # Converter para base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return has_defect, img_b64


# ================================
# ENDPOINT PRINCIPAL /scan
# ================================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    Recebe uma imagem (frame da câmera), roda YOLO e devolve:
      - has_defect: bool
      - message: texto em PT-BR
      - image_base64: imagem anotada (com retângulos vermelhos)
    """
    try:
        content = await file.read()
        has_defect, img_b64 = run_inference(content)

        if has_defect:
            msg = "Defeito detectado no tecido."
        else:
            msg = "Nenhum defeito visível detectado."

        return {
            "has_defect": has_defect,
            "message": msg,
            "image_base64": img_b64
        }

    except Exception as e:
        # Em produção você pode logar o erro
        return {
            "has_defect": False,
            "message": f"Erro ao processar imagem: {str(e)}",
            "image_base64": None
        }
