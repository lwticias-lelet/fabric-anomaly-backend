import cv2
import numpy as np
import torch

IMG_SIZE = 192

def generate_heatmap(model, img_gray):
    # Redimensiona para o tamanho usado no treino
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0

    # [1, 1, H, W]
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        recon = model(tensor).squeeze().cpu().numpy()

    # Erro por pixel (0–1 aprox.)
    diff = np.abs(img_norm - recon)

    # Erro máximo (intensidade da anomalia)
    max_error = float(diff.max())

    # Normaliza para 0–255 para visualização
    diff_norm = ((diff / (diff.max() + 1e-8)) * 255).astype("uint8")

    # 🔹 Suaviza um pouco para reduzir ruído de textura fina
    diff_blur = cv2.GaussianBlur(diff_norm, (5, 5), 0)

    # 🔹 Limiar relativo: só pega os ~5% piores pixels
    flat = diff_blur.flatten().astype("float32")
    p95 = np.percentile(flat, 95)  # 95º percentil
    # garante um mínimo razoável
    threshold_val = max(p95, 180.0)

    _, mask = cv2.threshold(
        diff_blur, int(threshold_val), 255, cv2.THRESH_BINARY
    )

    # 🔹 Limpa ruído (textura repetida) com abertura morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Proporção de área suspeita
    area_pixels = cv2.countNonZero(mask_clean)
    total_pixels = float(IMG_SIZE * IMG_SIZE)
    area_ratio = area_pixels / total_pixels

    # Heatmap colorido para visualização
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    base = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    # 🔴 Desenhar retângulos apenas em blobs "relevantes"
    contours, _ = cv2.findContours(
        mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:      # muito pequeno -> provavelmente textura
            continue
        if area > 0.25 * total_pixels:  # muito grande -> provavelmente sombra/fundo
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # evita contornos extremamente finos (bordas de dobra)
        aspect_ratio = w / float(h + 1e-6)
        if aspect_ratio < 0.25 or aspect_ratio > 4.0:
            continue

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img_resized, recon, overlay, max_error, area_ratio
