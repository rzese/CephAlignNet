import cv2
import numpy as np
import torch
from src.dataset_ceph import CephBiometricsDataset

def test_ceph_visualizer(target_display=1000):
    # Inizializzazione del dataset con il nuovo percorso
    dataset = CephBiometricsDataset(base_path="data/CephBiometrics400")

    # Prendiamo un campione a caso (es. il primo)
    img_tensor, landmarks = dataset[0]
    
    # Tensor -> immagine uint8
    img_vis = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    if img_vis is None or img_vis.size == 0:
        print("Errore: immagine vuota")
        return

    # Se l'immagine è (1,H,W) o (H,W,1) già gestito nel dataset, qui è 2D
    # Converto in BGR per disegnare colori
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    # Auto-orientamento: prefevoglio orientamento verticale (height >= width)
    h, w = img_vis.shape[:2]
    if w > h:
        # ruota 90 gradi in senso antiorario per avere portrait
        img_vis = cv2.rotate(img_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img_vis.shape[:2]
        # ruotiamo anche i landmark: (y,x) -> new coords
        new_landmarks = []
        for pt in landmarks:
            y, x = float(pt[0]), float(pt[1])
            # dopo rotazione CCW: new_x = y, new_y = (w_old - x - 1)
            new_x = y
            new_y = (w - 1) - x
            new_landmarks.append([new_y, new_x])
        landmarks = torch.tensor(new_landmarks, dtype=torch.float32)

    # Mostra il risultato
    h, w = img_vis.shape[:2]
    radius = max(4, int(min(h, w) / 200))          # punti più grandi su immagini grandi
    font_scale = max(0.6, min(h, w) / 1200)        # scala testo proporzionale
    thickness = max(1, int(min(h, w) / 800))

    # Disegna i punti (landmarks estratti come [Y, X])
    for i, pt in enumerate(landmarks):
        y, x = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img_vis, (x, y), radius, (0, 255, 255), -1)
            # testo leggermente spostato per non coprire il punto
            cv2.putText(img_vis, str(i+1), (x + radius + 4, y - radius - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        else:
            pass

    # Resize mantenendo proporzioni: target_display è il lato maggiore desiderato
    scale = target_display / max(w, h)
    img_resized = cv2.resize(img_vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Mostra finestra più grande e leggibile
    cv2.imshow("Verifica Ceph-Biometrics-400", img_resized)
    print("Premi un tasto sull'immagine per chiudere...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # target_display: lato maggiore in pixel (es. 1000 o 1200)
    # save_annotated=True per salvare l'immagine annotata su disco
    test_ceph_visualizer(target_display=1000, save_annotated=False)