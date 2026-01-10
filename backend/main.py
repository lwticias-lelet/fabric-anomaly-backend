from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64

# =========================
# Inicialização
# =========================
app = FastAPI()

# =========================
# CORS (produção + local)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://SEU_FRONT.vercel.app",  # ⬅️ troque pelo domínio real
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Wake-up do Render
# =========================
@app.get("/")
def root():
    return {"status": "backend ativo"}

# =========================
# Carrega modelo YOLO
# =========================
model = YOLO("best.pt")  # garanta que este arquivo está no backend

# =========================
# Endpoint de detecção
# =========================
@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    try:
        # Ler imagem enviada
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Inferência com parâmetros controlados
        results = model(
            image,
            conf=0.25,   # confiança mínima do modelo
            iou=0.45
        )[0]

        has_defect = False
        valid_boxes = []

        # =========================
        # FILTRO CORRETO (ESSENCIAL)
        # =========================
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0])

                # 🔥 THRESHOLD REAL DE DECISÃO
                if confidence >= 0.45:
                    has_defect = True
                    valid_boxes.append(box)

        # Desenha SOMENTE defeitos válidos
        for box in valid_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        # Converte imagem para base64
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise HTTPException(status_code=500, detail="Erro ao gerar imagem")

        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "has_defect": has_defect,
            "num_detections": len(valid_boxes),
            "heatmap_base64": image_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
