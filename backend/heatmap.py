# placeholder
import cv2
import numpy as np
import torch

IMG_SIZE = 192

def generate_heatmap(model, img_gray):
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0

    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        recon = model(tensor).squeeze().cpu().numpy()

    diff = np.abs(img_norm - recon)
    diff_norm = ((diff / (diff.max() + 1e-8)) * 255).astype("uint8")

    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )
    return img_resized, recon, overlay
