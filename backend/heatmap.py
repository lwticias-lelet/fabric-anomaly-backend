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

    # 3) Erro por pixel (0–1)
    diff = np.abs(img_norm - recon)

    max_error = float(diff.max())  # intensidade máxima do erro

    # 4) Construir máscara de pixels "suspeitos"
    #    Em vez de valor fixo, usamos média + 2*desvio padrão
    mean_err = float(diff.mean())
    std_err = float(diff.std())

    # limiar adaptativo por imagem
    pixel_thr = mean_err + 2.0 * std_err
    # garante que não fique nem ridiculamente baixo nem alto demais
    pixel_thr = float(np.clip(pixel_thr, 0.03, 0.5))

    # máscara binária em cima de diff (0–1)
    mask = (diff > pixel_thr).astype("uint8") * 255

    # 5) Suaviza / limpa ruído de textura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    total_pixels = float(IMG_SIZE * IMG_SIZE)
    area_pixels = cv2.countNonZero(mask_clean)
    area_ratio = area_pixels / total_pixels  # fração da imagem suspeita

    # 6) Heatmap só para visualização
    diff_norm = ((diff / (diff.max() + 1e-8)) * 255).astype("uint8")
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    base = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    # 7) Desenhar retângulos nas regiões suspeitas (a partir da máscara limpa)
    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # filtra coisas absurdamente pequenas (ruído) e gigantes (sombra geral)
        if area < 60:
            continue
        if area > 0.3 * total_pixels:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # evita contornos extremamente finos (dobras/linhas)
        aspect_ratio = w / float(h + 1e-6)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue

        # 🔴 retângulo vermelho em volta da região
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img_resized, recon, overlay, max_error, area_ratio
