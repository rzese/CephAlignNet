import torch
import cv2
import numpy as np
import sys
import os
import torch.nn.functional as F

from src.model import CephNet

CONFIG = {
    "num_landmarks": 29,
    "img_size":      512,
    "model_path":    "modello_migliore.pth",
}

# Nomi dei 29 landmark 
NOMI_LANDMARK = [
    "A",     # 1.  A-point (Subspinale)
    "ANS",   # 2.  Anterior Nasal Spine
    "B",     # 3.  B-point (Supramentale)
    "Me",    # 4.  Menton
    "N",     # 5.  Nasion
    "Or",    # 6.  Orbitale
    "Pog",   # 7.  Pogonion
    "PNS",   # 8.  Posterior Nasal Spine
    "R",     # 9.  Ramus
    "S",     # 10. Sella
    "Ar",    # 11. Articulare
    "Co",    # 12. Condylion
    "Gn",    # 13. Gnathion
    "Go",    # 14. Gonion
    "Po",    # 15. Porion
    "LPM",   # 16. Lower 2nd PM Cusp Tip
    "LIT",   # 17. Lower Incisor Tip
    "LMT",   # 18. Lower Molar Cusp Tip
    "UPM",   # 19. Upper 2nd PM Cusp Tip
    "UIA",   # 20. Upper Incisor Apex
    "UIT",   # 21. Upper Incisor Tip
    "UMT",   # 22. Upper Molar Cusp Tip
    "LIA",   # 23. Lower Incisor Apex
    "Li",    # 24. Labrale inferius
    "Ls",    # 25. Labrale superius
    "N'",    # 26. Soft Tissue Nasion
    "Pog'",  # 27. Soft Tissue Pogonion
    "Sn",    # 28. Subnasale
    "Pn",    # 29. Pronasale
]

def carica_modello(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = CephNet(num_landmarks=CONFIG["num_landmarks"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Modello caricato (epoca {checkpoint['epoca']}, val_loss {checkpoint['val_loss']:.4f})")
    return model

def predici(model, img_path, device, img_size=512):
    # Carica immagine originale
    img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"Immagine non trovata: {img_path}")

    h_orig, w_orig = img_orig.shape

    # Preprocessing: resize + normalizza
    img_resized = cv2.resize(img_orig, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Predizione
    with torch.no_grad():
        pred = model(img_tensor)

    # Da [1, 58] a [29, 2] coordinate normalizzate
    pred = pred.view(-1, 2).cpu().numpy()

    # Riporta le coordinate ai pixel dell'immagine originale
    pred[:, 0] = pred[:, 0] * w_orig
    pred[:, 1] = pred[:, 1] * h_orig

    return img_orig, pred


def disegna_landmark(img_gray, landmarks, mostra=True):
    # Converti in BGR per disegnare a colori
    img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w = img_vis.shape[:2]

    raggio = max(6, int(min(h, w) / 150))
    font_scale = max(0.4, min(h, w) / 1500)
    spessore = max(1, int(min(h, w) / 600))

    for i, pt in enumerate(landmarks):
        x, y = int(pt[0]), int(pt[1])

        # Salta punti fuori immagine
        if not (0 <= x < w and 0 <= y < h):
            continue

        # Cerchio giallo
        cv2.circle(img_vis, (x, y), raggio, (0, 220, 255), -1)
        cv2.circle(img_vis, (x, y), raggio, (0, 0, 0), 1)

        # Nome del landmark
        nome = NOMI_LANDMARK[i] 
        cv2.putText(img_vis, nome, (x + raggio + 3, y + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255), spessore, cv2.LINE_AA)

    if mostra:
        scala = min(1.0, 1000 / max(h, w))
        img_show = cv2.resize(img_vis, (int(w * scala), int(h * scala)))
        cv2.imshow("Predizione landmark", img_show)
        print("Premi un tasto per chiudere...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_vis


def main():

    img_path = "data/Aariz/test/Cephalograms/cks2ip8fp29yl0yuf6ry9266i.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = carica_modello(CONFIG["model_path"], device)

    img_orig, landmarks = predici(model, img_path, device, CONFIG["img_size"])

    disegna_landmark(img_orig, landmarks, mostra=True)

    print(f"\nCoordinate predette (pixel):")
    for i, pt in enumerate(landmarks):
        nome = NOMI_LANDMARK[i] 
        print(f"  {nome:5s}: x={pt[0]:.1f}, y={pt[1]:.1f}")


if __name__ == "__main__":
    main()