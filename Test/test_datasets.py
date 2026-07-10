import cv2
import numpy as np
import torch
from src.datasets import Datasets

# Inizializzazione corretta con i tre percorsi
dataset = Datasets(
    aariz_path="data/Aariz", 
    isbi_path="data/Cepha400",
    ceph_path="data/CephBiometrics400"
)

def visualizza_campione(idx):
    # Estrazione 4 elementi da dataset
    img_tensor, points, mask, bbox = dataset[idx]
    
    # Conversione per OpenCV
    img = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Disegno rettangolo (BBox in Verde)
    x1, y1, x2, y2 = bbox.numpy().astype(int)
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3) 
    
    # Disegno landmark (Punti in Rosso)
    for i in range(len(points)):
        if mask[i].item() == 1.0:   # .item() converte il tensore in float Python
            px, py = points[i].numpy().astype(int)
            cv2.circle(img_vis, (px, py), 5, (0, 0, 255), -1)
            
    # Capire dinamicamente da quale dataset proviene l'immagine
    if idx < dataset.len_aariz:
        origine = "AARIZ"
    elif idx < (dataset.len_aariz + dataset.len_isbi):
        origine = "ISBI"
    else:
        origine = "CEPH-BIOMETRICS"
        
    titolo = f"Test - {origine} (Idx assoluto: {idx})"
    print(f"Mostrando {titolo}...")
    
    # Visualizzazione
    cv2.imshow(titolo, cv2.resize(img_vis, (800, 1000)))
    cv2.waitKey(0)

# ----------------- ESECUZIONE -----------------
# Calcoliamo gli indici per testare la PRIMA immagine di ogni dataset
idx_aariz = 0
idx_isbi = dataset.len_aariz
idx_ceph = dataset.len_aariz + dataset.len_isbi

# Testiamo le tre immagini in sequenza
visualizza_campione(idx_aariz) 
visualizza_campione(idx_isbi) 
visualizza_campione(idx_ceph) 

cv2.destroyAllWindows()