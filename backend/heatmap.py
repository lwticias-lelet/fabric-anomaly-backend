import cv2
import numpy as np
import torch

IMG_SIZE = 192


def generate_heatmap(model, img_gray):
    """
    Recebe imagem em tons de cinza (np.ndarray),
    gera:
      - overlay com heatmap + retângulos em volta das regiões suspeitas
      - erro máximo (max_error)
      - proporção de área suspeita (area_ratio)
    """

    # 1) Redimensiona para o tamanho do modelo
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0  # 0–1

    # 2) Passa pelo modelo [1, 1, H, W]
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        recon = model(tensor).squeeze().cpu().numpy()

    # 3) Erro por pixel (0–1) - diferença entre original e reconstrução
    diff = np.abs(img_norm - recon)
    max_error = float(diff.max())  # intensidade máxima do erro na imagem

    # 4) Construir máscara BINÁRIA de pixels "suspeitos"
    #    Limiar adaptativo baseado na estatística do erro
    mean_err = float(diff.mean())
    std_err = float(diff.std())

    # limiar dinâmico: média + 1.5 * desvio padrão
    pixel_thr = mean_err + 1.5 * std_err
    # clamp para evitar extremos (nem tão baixo, nem tão alto)
    pixel_thr = float(np.clip(pixel_thr, 0.02, 0.25))

    # máscara inicial booleana -> uint8 0/255
    mask = (diff > pixel_thr).astype("uint8") * 255

    # 5) LIMPEZA DA MÁSCARA
    #    Objetivo: remover ruído fino (pontos isolados / textura) sem "engordar" demais
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # apenas Abertura: remove ruído, preserva forma dos defeitos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) Cálculo da Área Suspeita (fração da imagem marcada)
    total_pixels = float(IMG_SIZE * IMG_SIZE)
    area_pixels = cv2.countNonZero(mask)
    area_ratio = area_pixels / total_pixels if total_pixels > 0 else 0.0

    # 7) GERAÇÃO DO HEATMAP VISUAL
    diff_vis = ((diff / (diff.max() + 1e-8)) * 255).astype("uint8")
    heatmap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

    # 8) Overlay (imagem em cinza + heatmap colorido)
    base = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    # 9) ENCONTRA e DESENHA os RETÂNGULOS nas regiões suspeitas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA = 80          # ignora contornos muito pequenos (ruído)
    MAX_AREA_RATIO = 0.25  # ignora contornos gigantes (sombras/iluminação)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # filtro de tamanho
        if area < MIN_AREA:
            continue
        if area > (MAX_AREA_RATIO * total_pixels):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # filtro de proporção (evita linhas extremamente finas)
        aspect_ratio = w / float(h + 1e-6)
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue

        # 🔴 retângulo vermelho em volta da região suspeita
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # texto opcional "DEF" sobre o retângulo
        cv2.putText(
            overlay,
            "DEF",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    return img_resized, recon, overlay, max_error, area_ratio
