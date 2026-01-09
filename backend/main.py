from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from model_loader import load_model
from heatmap import generate_heatmap

app = FastAPI()

# CORS liberado
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

# Caminho do modelo treinado
MODEL_PATH = "models/autoencoder_192.pth"
model = load_model(MODEL_PATH)

# 🔥 Endpoint principal
@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    # Lê os bytes do arquivo enviado
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Erro lendo imagem enviada"}

    # Gera heatmap e score de anomalia
    original, recon, heatmap, anomaly_score = generate_heatmap(model, img)

    # 🎯 Limiar de decisão (threshold)
    # Você pode ajustar esse valor testando: quanto menor, mais sensível.
    THRESHOLD = 0.05  # 5% de erro médio

    has_defect = anomaly_score > THRESHOLD

    # Codifica o heatmap em PNG e depois em base64
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
