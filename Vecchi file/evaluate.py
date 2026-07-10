import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, random_split
from functools import partial
import torch.nn.functional as F

from src.datasets import Datasets
from src.model import CephNet

CONFIG = {
    "aariz_path":            "data/Aariz",
    "isbi_path":             "data/Cepha400",
    "ceph_path":             "data/CephBiometrics400",
    "num_landmarks":         19,
    "batch_size":            8,
    "val_split":             0.2,
    "img_size":              512,
    "model_path":            "modello_migliore.pth",
    "csv_path":              "data/Aariz/cephalogram_machine_mappings.csv",
    "pixel_size_isbi_ceph":  0.1,
    "sdr_thresholds_mm":     [2.0, 2.5, 3.0, 4.0],
    "sdr_thresholds":        [0.05, 0.10, 0.15, 0.20],
    "larghezza_originale":   2400,
}


def carica_pixel_size_aariz(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV non trovato: {csv_path}, uso default 0.1")
        return 0.1
    df = pd.read_csv(csv_path)
    media = df['pixel_size'].mean()
    print(f"CSV Aariz: {len(df)} immagini, pixel_size media = {media:.4f} mm/px")
    return media


def collate_con_resize(batch, size=512):
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    immagini_resize = [
        F.interpolate(img.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
        for img in immagini
    ]
    max_pts = max(p.shape[0] for p in punti)
    punti_pad, masks_pad = [], []
    for img_orig, p, m in zip(immagini, punti, masks):
        h_orig = img_orig.shape[1]
        w_orig = img_orig.shape[2]
        p_scaled = p.clone()
        p_scaled[:, 0] = p[:, 0] / w_orig
        p_scaled[:, 1] = p[:, 1] / h_orig
        n = p_scaled.shape[0]
        pad_p = torch.zeros(max_pts, 2)
        pad_p[:n] = p_scaled
        pad_m = torch.zeros(max_pts)
        pad_m[:n] = m
        punti_pad.append(pad_p)
        masks_pad.append(pad_m)
    return (
        torch.stack(immagini_resize),
        torch.stack(punti_pad),
        torch.stack(masks_pad),
        torch.stack(bboxes),
        torch.stack(pixel_sizes),
    )


def calcola_mre_sdr(model, loader, device, config, pixel_size_mm):
    model.eval()
    errori_per_landmark = [[] for _ in range(config["num_landmarks"])]
    mm_per_unita_norm = config["larghezza_originale"] * pixel_size_mm
    soglie_norm = [mm / mm_per_unita_norm for mm in config["sdr_thresholds_mm"]]

    with torch.no_grad():
        for immagini, punti, mask, _, _ in loader:
            immagini = immagini.to(device)
            punti    = punti.to(device)
            mask     = mask.to(device)
            pred = model(immagini)
            B, N = mask.shape
            pred = pred.view(B, -1, 2)[:, :N, :]
            distanze = torch.sqrt(((pred - punti) ** 2).sum(dim=2))
            for b in range(B):
                for i in range(N):
                    if mask[b, i].item() == 1.0:
                        errori_per_landmark[i].append(distanze[b, i].item())

    tutti_errori = [e for lista in errori_per_landmark for e in lista]
    mre_globale  = np.mean(tutti_errori) if tutti_errori else 0.0
    mre_per_landmark = [np.mean(lista) if lista else None for lista in errori_per_landmark]

    sdr = {}
    for soglia_norm_val, soglia_mm in zip(soglie_norm, config["sdr_thresholds_mm"]):
        sdr[soglia_mm] = np.mean([e < soglia_norm_val for e in tutti_errori]) * 100 if tutti_errori else 0.0

    sdr_norm = {}
    for soglia in config["sdr_thresholds"]:
        sdr_norm[soglia] = np.mean([e < soglia for e in tutti_errori]) * 100 if tutti_errori else 0.0

    return mre_globale, mre_per_landmark, sdr, sdr_norm, mm_per_unita_norm


def stampa_risultati(titolo, mre, mre_per_lm, sdr, mm_per_unita_norm, config):
    categorie = {range(1,16): "scheletrico", range(16,24): "dentale", range(24,30): "soft tissue"}
    print(f"\n{'='*55}")
    print(titolo)
    print(f"{'='*55}")
    print(f"MRE globale: {mre * mm_per_unita_norm:.2f} mm")

    scheletrici = [mre_per_lm[i] for i in range(15)    if mre_per_lm[i] is not None]
    dentali     = [mre_per_lm[i] for i in range(15, 23) if mre_per_lm[i] is not None]
    soft        = [mre_per_lm[i] for i in range(23, 29) if mre_per_lm[i] is not None]
    if scheletrici:
        print(f"MRE scheletrici (1-15):  {np.mean(scheletrici) * mm_per_unita_norm:.2f} mm")
    if dentali:
        print(f"MRE dentali    (16-23):  {np.mean(dentali) * mm_per_unita_norm:.2f} mm")
    if soft:
        print(f"MRE soft tissue(24-29):  {np.mean(soft) * mm_per_unita_norm:.2f} mm")

    print("\nSDR per soglia (mm):")
    for soglia_mm, valore in sdr.items():
        print(f"  entro {soglia_mm}mm: {valore:.1f}%")

    print("\nMRE per landmark:")
    for i, err in enumerate(mre_per_lm):
        cat = next((v for k, v in categorie.items() if (i+1) in k), "")
        if err is not None:
            print(f"  Landmark {i+1:2d} [{cat:12s}]: {err * mm_per_unita_norm:.2f} mm  ({err * config['img_size']:.1f} px)")
        else:
            print(f"  Landmark {i+1:2d} [{cat:12s}]: nessun campione valido")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    pixel_size_aariz = carica_pixel_size_aariz(CONFIG["csv_path"])
    pixel_size_altri = CONFIG["pixel_size_isbi_ceph"]

    dataset = Datasets(
        aariz_path=CONFIG["aariz_path"],
        isbi_path=CONFIG["isbi_path"],
        ceph_path=CONFIG["ceph_path"],
    )
    n_val   = int(len(dataset) * CONFIG["val_split"])
    n_train = len(dataset) - n_val
    _, val_ds = random_split(dataset, [n_train, n_val],
                             generator=torch.Generator().manual_seed(42))

    collate_fn = partial(collate_con_resize, size=CONFIG["img_size"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=collate_fn)

    aariz_indices = [i for i in val_ds.indices if i < dataset.len_aariz]
    print(f"Immagini Aariz nel validation: {len(aariz_indices)}")
    val_loader_aariz = DataLoader(
        torch.utils.data.Subset(dataset, aariz_indices),
        batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    checkpoint = torch.load(CONFIG["model_path"], map_location=device, weights_only=False)
    model = CephNet(num_landmarks=CONFIG["num_landmarks"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Modello caricato (epoca {checkpoint['epoca']}, val_loss {checkpoint['val_loss']:.4f})")

    n_aariz_val = len(aariz_indices)
    n_altri_val = len(val_ds) - n_aariz_val
    pixel_size_media = (
        (n_aariz_val * pixel_size_aariz + n_altri_val * pixel_size_altri) /
        (n_aariz_val + n_altri_val)
    ) if (n_aariz_val + n_altri_val) > 0 else pixel_size_aariz

    mre, mre_per_lm, sdr, _, mm_norm = calcola_mre_sdr(
        model, val_loader, device, CONFIG, pixel_size_media
    )
    stampa_risultati(
        f"VALUTAZIONE SU TUTTO IL VALIDATION SET (pixel_size={pixel_size_media:.4f} mm/px)",
        mre, mre_per_lm, sdr, mm_norm, CONFIG
    )

    if len(aariz_indices) > 0:
        mre_a, mre_per_lm_a, sdr_a, _, mm_norm_a = calcola_mre_sdr(
            model, val_loader_aariz, device, CONFIG, pixel_size_aariz
        )
        stampa_risultati(
            f"VALUTAZIONE SOLO SU AARIZ ({len(aariz_indices)} immagini, pixel_size={pixel_size_aariz:.4f} mm/px)",
            mre_a, mre_per_lm_a, sdr_a, mm_norm_a, CONFIG
        )
    else:
        print("\nNessuna immagine Aariz nel validation set.")


if __name__ == "__main__":
    main()