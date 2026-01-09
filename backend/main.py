from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from model_loader import load_model
from heatmap import generate_heatmap

app = FastAPI()

# CORS liberado (front na Vercel, backend no Render)
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


# Carrega modelo treinado
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

    # overlay = heatmap + retângulos / max_error / area_ratio
    original, recon, overlay, max_error, area_ratio = generate_heatmap(model, img)

    # 🎯 Regras de decisão:
    # - max_error: intensidade da anomalia (0–1 aprox.)
    # - area_ratio: fração da imagem que passou na máscara
    #
    # Ideia:
    #   Só considerar defeito se:
    #   1) houver um erro bem alto
    #   2) e a região suspeita ocupar uma área mínima
    #
    # Ajuste fino:
    #   - se marcar tecido normal como defeito -> aumentar um pouco
    #   - se deixar passar defeito -> diminuir um pouco
    MAX_ERR_THRESHOLD = 0.10     # erro máximo mínimo pra chamar de defeito
    AREA_THRESHOLD    = 0.002    # ~0,2% da imagem

    has_defect = (max_error > MAX_ERR_THRESHOLD) and (area_ratio > AREA_THRESHOLD)

    # Codifica overlay em PNG/base64
    success, heatmap_buffer = cv2.imencode(".png", overlay)
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
