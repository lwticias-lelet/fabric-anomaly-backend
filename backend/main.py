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

    # 🎯 REGRAS DE DECISÃO OTIMIZADAS
    # Foco: Detectar defeitos REAIS (pequenos/intensos ou grandes/sutis)
    # e evitar falsos positivos em tecidos uniformes.
    MAX_ERR_THRESHOLD = 0.08      # Reduzido: captura defeitos intensos
    AREA_THRESHOLD = 0.0015       # Ajustado: captura defeitos pequenos, mas ignora ruído

    # LÓGICA PRINCIPAL CORRIGIDA: "OU" em vez de "E"
    # Um defeito PODE ser: muito intenso (alta diferença) OU cobrir uma área significativa.
    has_defect = (max_error > MAX_ERR_THRESHOLD) or (area_ratio > AREA_THRESHOLD)

    # Pós-processamento da imagem final: Adiciona um TEXTO CLARO com o resultado
    # Define a cor do texto: VERDE para "OK", VERMELHO para "DEFEITO"
    text_color = (0, 255, 0) if not has_defect else (0, 0, 255)
    result_text = "SEM DEFEITO" if not has_defect else "DEFEITO DETECTADO"

    # Adiciona o texto na parte superior da imagem de resultado (overlay)
    cv2.putText(overlay, result_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # Codifica overlay em PNG/base64
    success, heatmap_buffer = cv2.imencode(".png", overlay)
    if not success:
        return {"error": "Erro ao converter imagem para PNG"}

    heatmap_base64 = base64.b64encode(heatmap_buffer).decode("utf-8")

    return {
        "heatmap_base64": heatmap_base64,
        "has_defect": has_defect,
        "max_error": float(max_error),
        "area_ratio": float(area_ratio),
        "max_error_threshold": MAX_ERR_THRESHOLD,
        "area_threshold": AREA_THRESHOLD,
    }