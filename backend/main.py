from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from model_loader import load_model
from heatmap import generate_heatmap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend rodando! Use /scan para análise."}

MODEL_PATH = "models/autoencoder_192.pth"
model = load_model(MODEL_PATH)

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Erro lendo imagem enviada"}

    # Agora generate_heatmap devolve: original, recon, heatmap, anomaly_score (MÁXIMO)
    original, recon, heatmap, anomaly_score = generate_heatmap(model, img)

    # 🎯 limiar com base no erro máximo (entre 0 e 1)
    THRESHOLD = 0.05  # pode ajustar depois

    has_defect = anomaly_score > THRESHOLD

    success, heatmap_buffer = cv2.imencode(".png", heatmap)
    if not success:
        return {"error": "Erro ao converter imagem para PNG"}

    heatmap_base64 = base64.b64encode(heatmap_buffer).decode("utf-8")

    return {
        "heatmap_base64": heatmap_base64,
        "has_defect": has_defect,
        "anomaly_score": anomaly_score,
        "threshold": THRESHOLD,
    }
