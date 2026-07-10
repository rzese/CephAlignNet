import math
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from functools import partial
import torch.nn.functional as F
import torchvision.transforms.functional as TF_vision

from src.datasets import Datasets
from src.model_hrnet import HRNetLandmark
from src.heatmap_utils import regression_voting, coordinate_a_heatmap

CONFIG = {
    "aariz_path":        "data/Aariz",
    "isbi_path":         "data/Cepha400",
    "ceph_path":         "data/CephBiometrics400",
    "num_landmarks":     19,
    "batch_size":        4,
    "val_split":         0.2,
    "img_size":          512,
    "heatmap_size":      256,
    "sigma":             4.0,
    "model_path":        "modello_heatmap.pth",
    "larghezza_orig_px": 2400,
    "sdr_thresholds_mm": [2.0, 2.5, 3.0, 4.0],
    "hrnet_channels":    [128, 256, 512, 1024],
}

# Landmark 1-15: scheletrici (comuni a tutti i dataset)
# Landmark 16-19: dentali (LPM, LIT, LMT, UPM — solo Aariz ha mask=1)
CATEGORIE = {range(1, 16): "scheletrico", range(16, 20): "dentale"}

NOMI = ["A","ANS","B","Me","N","Or","Pog","PNS","R","S",
        "Ar","Co","Gn","Go","Po","LPM","LIT","LMT","UPM"]


def collate_con_heatmap(batch, img_size=512, heatmap_size=256, sigma=4.0):
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    immagini_resize = [
        F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
        for img in immagini
    ]
    max_pts = 19
    punti_pad, masks_pad = [], []
    for img_orig, p, m in zip(immagini, punti, masks):
        h_orig = img_orig.shape[1]
        w_orig = img_orig.shape[2]
        p_scaled = p.clone()
        p_scaled[:, 0] = p[:, 0] / w_orig
        p_scaled[:, 1] = p[:, 1] / h_orig
        n = p_scaled.shape[0]
        pad_p = torch.zeros(max_pts, 2)
        pad_m = torch.zeros(max_pts)
        pad_p[:n] = p_scaled[:n]
        pad_m[:n] = m[:n]
        punti_pad.append(pad_p)
        masks_pad.append(pad_m)
    heatmaps_batch = [
        coordinate_a_heatmap(pts, msk, img_size, heatmap_size, sigma)
        for pts, msk in zip(punti_pad, masks_pad)
    ]
    return (torch.stack(immagini_resize), torch.stack(punti_pad), torch.stack(masks_pad), torch.stack(heatmaps_batch), torch.stack(bboxes), torch.stack(pixel_sizes))


def carica_modello(config, device):
    """Carica HRNetLandmark provando diversi canali per compatibilità."""
    checkpoint = torch.load(config["model_path"], map_location=device, weights_only=False)
    for canali in [config.get("hrnet_channels"),
                   [64, 128, 256, 512, 1024],
                   [128, 256, 512, 1024],
                   [64, 128, 256, 512],
                   [32, 64, 128, 256],
                   [18, 36, 72, 144],
                   None]:
        try:
            model = HRNetLandmark(
                num_landmarks=config["num_landmarks"],
                pretrained=False,
                hrnet_channels=canali,
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            canali_usati = canali if canali else model.hrnet_channels
            print(f"Modello caricato (canali={canali_usati}, "
                  f"epoca={checkpoint.get('epoca', '?')}, "
                  f"val_loss={checkpoint.get('val_loss', 0):.6f})")
            return model
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Unexpected key" in str(e):
                continue
            raise
    raise RuntimeError("Impossibile caricare il modello HRNet.")


def tta_predict(model, immagini, config, device):
    """
    Test-Time Augmentation: media di predizioni su varianti dell'immagine.
    Varianti: originale, flip orizzontale, rotazione +3°, rotazione -3°.
    Restituisce coordinate normalizzate [B, N, 2].
    """
    n_lm = config["num_landmarks"]
    all_preds = []

    # 1. Originale
    pred_raw, _ = model(immagini)
    all_preds.append(regression_voting(pred_raw, n_lm).to(device))

    # 2. Flip orizzontale
    img_flip = torch.flip(immagini, dims=[3])
    pred_flip, _ = model(img_flip)
    coords_flip = regression_voting(pred_flip, n_lm).to(device)
    coords_flip[:, :, 0] = 1.0 - coords_flip[:, :, 0]  # inverti x
    all_preds.append(coords_flip)

    # 3-4. Piccole rotazioni (±3°)
    for angle in [3.0, -3.0]:
        img_rot = TF_vision.rotate(immagini, angle)
        pred_rot, _ = model(img_rot)
        coords_rot = regression_voting(pred_rot, n_lm).to(device)
        # Ruota indietro le coordinate
        theta = math.radians(-angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx = coords_rot[:, :, 0] - 0.5
        cy = coords_rot[:, :, 1] - 0.5
        coords_rot[:, :, 0] = cos_t * cx - sin_t * cy + 0.5
        coords_rot[:, :, 1] = sin_t * cx + cos_t * cy + 0.5
        all_preds.append(coords_rot)

    # Media di tutte le predizioni
    return torch.stack(all_preds).mean(dim=0)


def calcola_mre_sdr(model, loader, device, config):
    model.eval()
    errori_per_landmark = [[] for _ in range(config["num_landmarks"])]
    with torch.no_grad():
        for immagini, punti, mask, _, _, pixel_sizes in loader:
            immagini = immagini.to(device)
            punti = punti.to(device)
            mask = mask.to(device)
            pixel_sizes = pixel_sizes.to(device)

            # TTA: media di predizioni su originale + flip + rotazioni
            pred_coords = tta_predict(model, immagini, config, device)

            B, N = mask.shape
            pred_coords = pred_coords[:, :N, :]
            distanze_norm = torch.sqrt(((pred_coords - punti) ** 2).sum(dim=2))
            for b in range(B):
                mm = config["larghezza_orig_px"] * pixel_sizes[b].item()
                for i in range(N):
                    if mask[b, i].item() == 1.0:
                        errori_per_landmark[i].append(distanze_norm[b, i].item() * mm)

    tutti = [e for lista in errori_per_landmark for e in lista]
    mre = np.mean(tutti) if tutti else 0.0
    mre_lm = [np.mean(lista) if lista else None for lista in errori_per_landmark]
    sdr = {s: np.mean([e < s for e in tutti]) * 100 if tutti else 0.0
              for s in config["sdr_thresholds_mm"]}
    return mre, mre_lm, sdr


def stampa_risultati(mre, mre_lm, sdr):
    print(f"\n{'='*55}")
    print("VALUTAZIONE SUL VALIDATION SET")
    print(f"{'='*55}")
    print(f"MRE globale: {mre:.2f} mm")
    schel = [mre_lm[i] for i in range(15) if mre_lm[i] is not None]
    dent  = [mre_lm[i] for i in range(15, 19) if mre_lm[i] is not None]
    if schel:
        print(f"MRE scheletrici (1-15):  {np.mean(schel):.2f} mm")
    if dent:
        print(f"MRE dentali    (16-19):  {np.mean(dent):.2f} mm")
    print("\nSDR per soglia (mm):")
    for s, v in sdr.items():
        print(f"  entro {s}mm: {v:.1f}%")
    print("\nMRE per landmark:")
    for i, err in enumerate(mre_lm):
        cat  = next((v for k, v in CATEGORIE.items() if (i+1) in k), "")
        nome = NOMI[i] if i < len(NOMI) else str(i+1)
        if err is not None:
            print(f"  {nome:5s} [{cat:12s}]: {err:.2f} mm")
        else:
            print(f"  {nome:5s} [{cat:12s}]: nessun campione valido")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    dataset = Datasets(aariz_path=CONFIG["aariz_path"], isbi_path=CONFIG["isbi_path"], ceph_path=CONFIG["ceph_path"])
    n_val   = int(len(dataset) * CONFIG["val_split"])
    _, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))

    collate_fn = partial(collate_con_heatmap, img_size=CONFIG["img_size"], heatmap_size=CONFIG["heatmap_size"], sigma=CONFIG["sigma"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = carica_modello(CONFIG, device)

    mre, mre_lm, sdr = calcola_mre_sdr(model, val_loader, device, CONFIG)
    stampa_risultati(mre, mre_lm, sdr)


if __name__ == "__main__":
    main()