from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64  # 🔥 IMPORTANTE: para gerar base64
from model_loader import load_model
from heatmap import generate_heatmap
import torch

app = FastAPI()

# CORS liberado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rota principal só para teste
@app.get("/")
def root():
    return {"status": "Backend rodando! Use /scan para análise."}

# Carregar modelo treinado
MODEL_PATH = "models/autoencoder_192.pth"
model = load_model(MODEL_PATH)

# Endpoint principal: recebe imagem, processa e devolve heatmap base64
@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    # Ler bytes
    contents = await file.read()

    # Converter para imagem
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Erro lendo imagem enviada"}

    # Gerar heatmap com modelo
    original, recon, heatmap = generate_heatmap(model, img)

    # Codificar heatmap em PNG
    success, heatmap_buffer = cv2.imencode(".png", heatmap)

    if not success:
        return {"error": "Erro ao converter imagem para PNG"}

    # Converter para base64
    heatmap_base64 = base64.b64encode(heatmap_buffer).decode("utf-8")

    # Retornar no formato que o frontend espera
    return {
        "heatmap_base64": heatmap_base64
    }
