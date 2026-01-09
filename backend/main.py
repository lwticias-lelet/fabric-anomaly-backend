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

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Erro lendo imagem enviada"}

    # original, recon, heatmap, max_error, area_ratio vindo do heatmap.py
    original, recon, heatmap, max_error, area_ratio = generate_heatmap(model, img)

    # 🎯 Limiares MAIS SENSÍVEIS
    # max_error  -> intensidade máxima do defeito (0–1 aprox.)
    # area_ratio -> fração da imagem marcada como suspeita (0–1)
    #
    # Se ainda marcar tecido normal como defeito:
    #   - aumente um pouco MAX_ERR_THRESHOLD ou AREA_THRESHOLD
    # Se continuar deixando passar defeito:
    #   - diminua um dos dois.
    MAX_ERR_THRESHOLD = 0.04    # estava muito alto; deixamos mais sensível
    AREA_THRESHOLD    = 0.001   # ~0,1% da imagem

    # ✅ Agora, basta UMA das condições ser verdadeira para considerar defeito
    has_defect = (max_error > MAX_ERR_THRESHOLD) or (area_ratio > AREA_THRESHOLD)

    # Codifica overlay (heatmap + retângulos) em PNG/base64
    success, heatmap_buffer = cv2.imencode(".png", heatmap)
    if not success:
        return {"error": "Erro ao converter imagem para PNG"}

    heatmap_base64 = base64.b64encode(heatmap_buffer).decode("utf-8")

    return {
        "heatmap_base64": heatmap_base64,
        "has_defect": has_defect,
        "max_error": max_error,
        "area_ratio": area_ratio,
        "max_error_threshold": MAX_ERR_THRESHOLD,
        "area_threshold": AREA_THRESHOLD,
    }
