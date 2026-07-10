"""
verifica_dataset.py — controllo rapido che il dataset si carichi senza errori
prima di lanciare un training lungo.

USO (da cmd, dentro la cartella Progetto_tesi, con l'ambiente cephnet attivo):
    python verifica_dataset.py
"""
from src.datasets import Datasets

print("Caricamento dataset in corso...\n")

d = Datasets(
    aariz_path="data/Aariz",
    isbi_path="data/Cepha400",
    tracciati_path="data/tracciati_forniti",
)

print(f"\nTotale campioni nel dataset: {len(d)}")

print("\nCarico il primo campione (indice 0)...")
img, pts, mask, bbox, px = d[0]

print(f"Immagine:    shape={img.shape}  dtype={img.dtype}  range=[{img.min():.3f}, {img.max():.3f}]")
print(f"Punti:       shape={pts.shape}")
print(f"Mask:        landmark validi={int(mask.sum().item())}/{mask.shape[0]}")
print(f"Bbox:        {bbox.tolist()}")
print(f"Pixel size:  {px.item()}")

print("\nCarico anche l'ultimo campione, per controllare il dataset Tracciati...")
img2, pts2, mask2, bbox2, px2 = d[len(d) - 1]
print(f"Immagine:    shape={img2.shape}")
print(f"Mask:        landmark validi={int(mask2.sum().item())}/{mask2.shape[0]}")

print("\n>>> Se non vedi errori sopra, il dataset si carica correttamente. <<<")