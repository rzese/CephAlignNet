import cv2
import torch
import numpy as np
import os
from src.dataset_isbi import ISBIDataset

def test_isbi_visualizer():
    #Configurazione percorsi
    base_path = "data/Cepha400"

    if not os.path.isdir(base_path):
        print(f"ERRORE: Non trovo la cartella {base_path}")
        return

    #Inizializzo il dataset
    dataset = ISBIDataset(base_path=base_path)

    #primo elemento
    img_tensor, landmarks = dataset[0]
    
    #Conversione
    img_vis = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    #punti ISBI rimappati sui 26 indici canonici ([0, 0] = non annotato)
    for i, pt in enumerate(landmarks):
        if pt[0] == 0 and pt[1] == 0:
            continue
        x, y = int(pt[0]), int(pt[1])
        #BLU per distinguere ISBI da Aariz (che era verde)
        cv2.circle(img_vis, (x, y), 8, (255, 0, 0), -1)
        cv2.putText(img_vis, str(i+1), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 6. Mostra  
    cv2.imshow("Verifica ISBI (19 punti)", cv2.resize(img_vis, (800, 1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_isbi_visualizer()