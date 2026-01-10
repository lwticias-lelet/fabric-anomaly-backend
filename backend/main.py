# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import numpy as np
import cv2
import base64
import os

app = FastAPI()

# Libera o CORS (Vercel no front, Render no back)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se quiser, depois restringe ao domínio da Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho do modelo YOLO treinado
MODEL_PATH = os.path.join("models", "best.pt")

# Carrega o modelo YOLO na memória
try:
    print(f"Carregando modelo YOLO em: {MODEL_PATH}")
    yolo_model = YOLO(MODEL_PATH)
    print("Modelo YOLO carregado com sucesso.")
except Exception as e:
    print("Erro ao carregar modelo YOLO:", e)
    yolo_model = None


@app.get("/")
def root():
    return {"status": "Backend rodando com YOLO. Use POST /scan para analisar."}


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    """
    Recebe uma imagem (arquivo), roda YOLO em cima,
    desenha bounding boxes vermelhos nos defeitos
    e devolve a imagem final em base64 + flag has_defect.
    """
    if yolo_model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo YOLO não carregado no servidor."
        )

    # Lê bytes da imagem enviada
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)

    # Decodifica para BGR (colorida)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Não foi possível ler a imagem enviada.")

    # Faz uma cópia para desenhar por cima
    overlay = img.copy()

    # Roda inferência com YOLO
    # conf pode ser ajustado (0.25–0.4) para mais/menos sensibilidade
    results = yolo_model(overlay, conf=0.35, iou=0.45, verbose=False)
    r = results[0]
    boxes = r.boxes

    has_defect = boxes is not None and len(boxes) > 0

    # Se houver detecções, desenha retângulos vermelhos
    if has_defect:
        for box in boxes:
            # coordenadas do bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # confiança
            conf = float(box.conf[0].cpu().numpy())

            # Desenha retângulo vermelho
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label "DEFEITO" + confiança
            label = f"DEFEITO {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Caixa sólida por trás do texto
            cv2.rectangle(
                overlay,
                (x1, max(0, y1 - th - baseline)),
                (x1 + tw, y1),
                (0, 0, 255),
                thickness=-1,
            )

            # Texto em branco
            cv2.putText(
                overlay,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status_text = "DEFEITO DETECTADO"
        status_color = (0, 0, 255)  # vermelho
    else:
        status_text = "SEM DEFEITO"
        status_color = (0, 255, 0)  # verde

    # Escreve o status na parte superior da imagem
    cv2.putText(
        overlay,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Codifica overlay em PNG -> base64
    success, buffer = cv2.imencode(".png", overlay)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Erro ao converter imagem processada para PNG."
        )

    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    # Mantendo a mesma "shape" de resposta do backend antigo
    return {
        "heatmap_base64": heatmap_base64,  # agora é a imagem com bounding boxes
        "has_defect": has_defect,
        "num_boxes": int(len(boxes)) if boxes is not None else 0,
    }
