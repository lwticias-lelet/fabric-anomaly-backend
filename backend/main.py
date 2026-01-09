from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from model_loader import load_model
from heatmap import generate_heatmap
import torch

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 ROTA PRINCIPAL PARA TESTAR O BACKEND
@app.get("/")
def root():
    return {"status": "Backend rodando! Use /scan para análise."}

# 🔥 Carregar modelo
MODEL_PATH = "models/autoencoder_192.pth"
model = load_model(MODEL_PATH)

# ENDPOINT DE SCAN
@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Erro lendo imagem enviada"}

    original, recon, heatmap = generate_heatmap(model, img)

    _, heatmap_buffer = cv2.imencode(".jpg", heatmap)

    return {
        "heatmap": heatmap_buffer.tobytes()
    }
    
