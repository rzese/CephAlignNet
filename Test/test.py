import cv2
import torch
import numpy as np
import os
from src.dataset import AarizDataset 

def test_visualizer():
    #Configurazione percorsi
    base_path = "data/Aariz" 
    
    if not os.path.exists(base_path):
        print(f"ERRORE: La cartella {base_path} non esiste!")
        return

    #Inizializzazione dataset
    dataset = AarizDataset(base_path=base_path, split='train')
    print(f"Dataset caricato. Immagini trovate: {len(dataset)}")

    #primo elemento
    img_tensor, landmarks = dataset[0]
    
    #Conversione per visualizzare con OpenCV
    #tensor [1, H, W] a array numpy [H, W] e riportiamo i valori a 0-255
    img_vis = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    # Conversione in BGR per poter disegnare a colori
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    #Disegno tutti i punti trovati nel json
    print(f"Disegnando {len(landmarks)} punti...")
    for i, pt in enumerate(landmarks):
        x, y = int(pt[0]), int(pt[1])
        #Disegno un cerchio verde
        cv2.circle(img_vis, (x, y), 8, (0, 255, 0), -1)
        #Scrivo un numero del punto accanto
        cv2.putText(img_vis, str(i), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 6. Mostra l'immagine ridimensionata
    h, w = img_vis.shape[:2]
    img_resized = cv2.resize(img_vis, (int(w*0.5), int(h*0.5))) # Rimpicciolita al 50%
    cv2.imshow("Verifica Landmark Aariz (Tutti i punti)", img_resized)
    
    print("Visualizzazione aperta. Premi un tasto qualsiasi per chiudere.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_visualizer()