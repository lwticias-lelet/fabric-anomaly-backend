import cv2
import numpy as np
import torch

IMG_SIZE = 192

def generate_heatmap(model, img_gray):
    # Redimensiona a imagem para o tamanho esperado pelo modelo
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0

    # [1, 1, H, W] para o modelo
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        recon = model(tensor).squeeze().cpu().numpy()

    # Mapa de erro (diferença entre original e reconstruído)
    diff = np.abs(img_norm - recon)

    # 🔢 Score de anomalia = média do erro (vai de 0 a ~1)
    anomaly_score = float(diff.mean())

    # Normaliza o erro para 0–255 para visualização
    diff_norm = ((diff / (diff.max() + 1e-8)) * 255).astype("uint8")

    # Aplica mapa de cores
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Sobrepõe o heatmap sobre a imagem original
    overlay = cv2.addWeighted(
        cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    # Agora também retornamos o score
    return img_resized, recon, overlay, anomaly_score
