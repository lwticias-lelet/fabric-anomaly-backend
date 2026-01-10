from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# CORS liberado porque frontend pode estar na Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
#   Carregar modelo YOLO
# -----------------------
MODEL_PATH = "models/best.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
    print("Erro ao carregar modelo:", e)


@app.get("/")
def root():
    return {"status": "Backend rodando! Envie imagem para /scan"}


# -----------------------
#     ROTA DE SCAN
# -----------------------
@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    # Ler bytes da imagem
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Erro lendo imagem"}

    # Executar YOLO
    results = model(frame, conf=0.35)  # confiança ajustável

    # Copiar imagem para desenhar overlay
    output = frame.copy()
    has_defect = False
    boxes_output = []

    # Processar detecções
    for r in results:
        if not r.boxes:
            continue

        for box in r.boxes:
            has_defect = True

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())

            # Desenhar retângulo vermelho
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Texto "DEFEITO"
            cv2.putText(output, "DEFEITO", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Guardar box
            boxes_output.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": conf
            })

    # Texto principal no topo
    label = "DEFEITO DETECTADO" if has_defect else "SEM DEFEITO"
    color = (0, 0, 255) if has_defect else (0, 255, 0)

    cv2.putText(output, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # Converter imagem com overlay para Base64
    success, buffer = cv2.imencode(".jpg", output)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "has_defect": has_defect,
        "boxes": boxes_output,
        "image_base64": heatmap_base64
    }
