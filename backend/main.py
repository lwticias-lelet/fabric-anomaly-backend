import os
import io
import base64

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ============================================================
# CONFIGURAÇÃO DA API
# ============================================================

app = FastAPI(title="Fabric Anomaly Scanner - YOLOv8")

# CORS liberado (pode depois restringir para o domínio do Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho do modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

DEVICE = "cpu"

print(f"Carregando modelo em {DEVICE} a partir de {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("✅ Modelo YOLO carregado com sucesso!")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converte PIL (RGB) para OpenCV (BGR)."""
    arr = np.array(pil_image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_base64(img_bgr: np.ndarray) -> str:
    """Converte imagem OpenCV (BGR) em base64 (JPEG)."""
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode("utf-8")


# ============================================================
# ROTAS
# ============================================================

@app.get("/")
async def root():
    """Rota raiz – só pra quando você abrir o link no navegador não dar 404."""
    return {
        "status": "ok",
        "message": "Backend Fabric Anomaly Scanner rodando. Use /health ou POST /scan."
    }


@app.get("/health")
async def health():
    """Health check simples (usado pelo Render/Vercel)."""
    return {"status": "ok"}


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    """
    Recebe uma imagem, roda YOLOv8 e:

    - Se encontrar qualquer bounding box -> has_defect = True
    - Se não encontrar -> has_defect = False

    Retorno:
      - has_defect: bool
      - message: str
      - image_base64: imagem com overlay (retângulo vermelho ou moldura verde)
    """
    try:
        # -------------------------
        # 1) Ler imagem enviada
        # -------------------------
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_bgr = pil_to_cv2(pil_img)

        # -------------------------
        # 2) Rodar YOLO
        # -------------------------
        results = model.predict(
            source=img_bgr,
            imgsz=640,
            conf=0.25,      # pode ajustar depois (0.1 - 0.3 se estiver muito cego)
            iou=0.45,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        overlay = img_bgr.copy()

        has_defect = False

        if boxes is not None and len(boxes) > 0:
            has_defect = True

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                # Retângulo vermelho
                cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), 3)

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
            has_defect = False
            message = "Nenhum defeito detectado no tecido."

            h, w = overlay.shape[:2]
            cv2.rectangle(overlay, (10, 10), (w - 10, h - 10), (0, 255, 0), 3)
            cv2.putText(
                overlay,
                "SEM DEFEITO",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # -------------------------
        # 3) Converter para base64
        # -------------------------
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


# ============================================================
# RODAR LOCALMENTE
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
