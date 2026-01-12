# backend/main.py

import io
import os
import base64
from typing import Tuple, List, Optional

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

# limiares do YOLO (mais sensível que o padrão)
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5


# ================================
# FASTAPI APP
# ================================

app = FastAPI(title="Fabric Anomaly Scanner - YOLO + Filtros")

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
# PRÉ-PROCESSAMENTO / HEURÍSTICA
# ================================

def enhance_for_yolo(img_bgr: np.ndarray) -> np.ndarray:
    """
    Realça o tecido para facilitar a detecção:
    - Redução de ruído preservando bordas (bilateral)
    - Equalização adaptativa de contraste (CLAHE) no canal de luminosidade
    """
    # (1) reduzir ruído
    denoised = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=60, sigmaSpace=60)

    # (2) equalizar contraste
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced


def heuristic_large_defect_boxes(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Fallback simples:
    - Detecta bordas fortes
    - Dilata
    - Pega contornos muito grandes (rasgos/manchas bem visíveis)
    Retorna uma lista de caixas (x, y, w, h).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecção de bordas
    edges = cv2.Canny(gray, 40, 120)

    # Dilatar bordas para unir regiões
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w
    boxes: List[Tuple[int, int, int, int]] = []

    for c in contours:
        area = cv2.contourArea(c)
        # só consideramos contornos bem grandes (>= 2% da área da imagem)
        if area > 0.02 * img_area:
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append((x, y, cw, ch))

    return boxes


# ================================
# FUNÇÃO AUXILIAR DE INFERÊNCIA
# ================================

def run_inference(image_bytes: bytes) -> Tuple[bool, str, Optional[float], str]:
    """
    Roda YOLO com filtros + heurística de backup e devolve:

      - has_defect: bool
      - image_b64: imagem anotada (com bounding boxes vermelhos) em base64
      - max_conf: maior confiança encontrada (0–1) ou None
      - source: "yolo", "heuristic" ou "none"
    """
    # Abrir imagem com PIL (RGB)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)  # RGB

    # Converter para BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Reduzir tamanho se estiver muito grande (ajuda no Render)
    h, w = img_bgr.shape[:2]
    max_side = 900
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Realce para o YOLO
    img_bgr_enh = enhance_for_yolo(img_bgr)
    img_rgb_enh = cv2.cvtColor(img_bgr_enh, cv2.COLOR_BGR2RGB)

    # ----------------------------------
    # 1) YOLO
    # ----------------------------------
    results = model(
        img_rgb_enh,
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    has_defect = False
    max_conf: Optional[float] = None
    source = "none"

    # Desenhar detecções do YOLO, se existirem
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # Atualiza maior confiança
            if max_conf is None or conf > max_conf:
                max_conf = conf

            # Só consideramos como defeito se conf >= CONF_THRESHOLD
            if conf >= CONF_THRESHOLD:
                has_defect = True
                source = "yolo"

                # Desenhar retângulo vermelho
                cv2.rectangle(
                    img_bgr_enh,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),  # BGR (vermelho)
                    3
                )

                # Confiança em cima da caixa
                cv2.putText(
                    img_bgr_enh,
                    f"{conf:.2f}",
                    (int(x1), max(int(y1) - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

    # ----------------------------------
    # 2) FALLBACK HEURÍSTICO
    # ----------------------------------
    if not has_defect:
        big_boxes = heuristic_large_defect_boxes(img_bgr_enh)
        if big_boxes:
            has_defect = True
            source = "heuristic"
            # heurística não tem confiança numérica
            for (x, y, cw, ch) in big_boxes:
                x1, y1, x2, y2 = x, y, x + cw, y + ch
                cv2.rectangle(
                    img_bgr_enh,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

    # Converter de volta para RGB
    img_rgb_out = cv2.cvtColor(img_bgr_enh, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(img_rgb_out)

    # JPEG em memória
    buffer = io.BytesIO()
    pil_out.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    img_bytes = buffer.read()

    # base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return has_defect, img_b64, max_conf, source


# ================================
# ENDPOINT PRINCIPAL /scan
# ================================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    Recebe uma imagem (frame da câmera), roda YOLO + filtros + heurística e devolve:

      - has_defect: bool
      - message: texto em PT-BR
      - image_base64: imagem anotada (com retângulos vermelhos)
      - confidence: maior confiança (0–1) ou null
      - source: "yolo" | "heuristic" | "none"
    """
    try:
        content = await file.read()
        has_defect, img_b64, max_conf, source = run_inference(content)

        if has_defect:
            if source == "yolo":
                msg = "Defeito detectado no tecido."
            elif source == "heuristic":
                msg = "Possível defeito detectado no tecido (padrão de bordas)."
            else:
                msg = "Defeito detectado no tecido."
        else:
            msg = "Nenhum defeito visível detectado."

        return {
            "has_defect": has_defect,
            "message": msg,
            "image_base64": img_b64,
            "confidence": max_conf,
            "source": source,
        }

    except Exception as e:
        # Em produção: logar o erro
        return {
            "has_defect": False,
            "message": f"Erro ao processar imagem: {str(e)}",
            "image_base64": None,
            "confidence": None,
            "source": "error",
        }
