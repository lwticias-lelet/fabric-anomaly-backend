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

    # 3) Erro por pixel (0–1) - Esta é a diferença crucial entre original e reconstrução
    diff = np.abs(img_norm - recon)
    max_error = float(diff.max())  # intensidade máxima do erro na imagem

    # 4) CRÍTICO: Construir uma máscara BINÁRIA limpa de pixels "suspeitos"
    #    Método: limiar adaptativo baseado na distribuição do erro
    mean_err = float(diff.mean())
    std_err = float(diff.std())

    # Limiar Dinâmico e EFICAZ: média + 1.5 * desvio padrão
    # - 1.5 é um bom equilíbrio entre sensibilidade e ruído.
    # - O clamp evita valores absurdos.
    pixel_thr = mean_err + 1.5 * std_err
    pixel_thr = float(np.clip(pixel_thr, 0.02, 0.3))  # Mais restritivo no máximo

    # Cria máscara inicial
    mask = (diff > pixel_thr).astype("uint8") * 255

    # 5) LIMPEZA DA MÁSCARA (Pós-processamento ESSENCIAL)
    #    Objetivo: remover pontos solitários (ruído) e unir regiões próximas.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Primeiro 'abre' (remove ruído), depois 'dilata' (une regiões próximas do defeito)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # 6) Cálculo da Área Suspeita (para a decisão final)
    total_pixels = float(IMG_SIZE * IMG_SIZE)
    area_pixels = cv2.countNonZero(mask)
    area_ratio = area_pixels / total_pixels

    # 7) GERAÇÃO DO HEATMAP VISUAL (apenas para overlay colorido)
    #    O heatmap é uma representação colorida do erro 'diff'
    diff_vis = np.copy(diff)
    diff_vis = ((diff_vis / (diff_vis.max() + 1e-8)) * 255).astype("uint8")
    heatmap = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

    # 8) Cria a imagem base (colorida) e sobrepõe o heatmap com transparência
    base = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    # 9) ENCONTRA e DESENHA os RETÂNGULOS (Bounding Boxes) - CORREÇÃO PRINCIPAL
    #    Busca os contornos na máscara LIMPA.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtros para os contornos:
    MIN_AREA = 80         # Contornos com menos de 80 pixels são ignorados (ruído).
    MAX_AREA_RATIO = 0.25 # Ignora contornos que cobrem mais de 25% da imagem (provavelmente sombra/iluminação).

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filtro de Tamanho
        if area < MIN_AREA:
            continue
        if area > (MAX_AREA_RATIO * total_pixels):
            continue

        # Obtém o retângulo delimitador
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtro de Proporção (ignora linhas finas que podem ser dobras)
        aspect_ratio = w / float(h + 1e-6)
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue

        # 🔴 DESENHA o RETÂNGULO VERMELHO (Borda grossa para visibilidade)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Opcional: Adiciona um pequeno texto "DEFEITO" sobre o retângulo
        cv2.putText(overlay, "DEF", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    return img_resized, recon, overlay, max_error, area_ratio