"""
train_refine.py — trains Stage 2 on crops of the ORIGINAL radiograph.

Stage 2 receives a finestra_mm x finestra_mm crop of the full-resolution image
centered on a (wrong) landmark position and has to find where the landmark
really is inside the crop (ImageRefinementNet, src/model_refine.py).

Training does not need Stage 1: the starting position is the ground-truth
point moved by a synthetic error, so Stage 2 cannot learn to "read back"
Stage 1's own belief (the problem of the old feature-based Stage 2).
The synthetic error has a distance UNIFORM between rumore_min_mm and
rumore_max_mm and a uniform direction. No assumption is made on the
distribution of the real Stage 1 errors: with the fusion, Stage 2 counts mostly
where Stage 1 is uncertain, so it must work equally well at every distance.
Training crops are also randomly rotated (the target is rotated with them) and
photometrically perturbed. Validation uses the same synthetic errors with a
fixed seed, so its numbers are comparable across epochs.

Loss: Huber on the predicted offset (mm) + peso_nll * NLL of the true position
under the crop heatmap.
"""
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from src.datasets import Datasets, crea_split
from src.heatmap_utils import nll_heatmap_loss, ruota_coordinate
from src.inferenza import ritaglia_originali
from src.model_refine import ImageRefinementNet

CONFIG = {
    "aariz_path":     "data/Aariz",
    "isbi_path":      "data/Cepha400",
    "tracciati_path": None,   # e.g. "data/tracciati_forniti"; not used for now
    "tracciati_tabella_scale": None,
    "num_landmarks":  26,
    "batch_size":     8,       # images per batch (each gives ~20-26 crops)
    "num_workers":    4,
    "num_epochs":     200,
    "learning_rate":  1e-3,
    "val_split":      0.2,
    "save_path":      "modello_refine.pth",
    "patience":       30,
    # ── crop and correction range ──
    "finestra_mm":    38.4,    # side of the crop (+-19.2 mm: the true point stays well inside even at 11.5 mm)
    "crop_px":        192,     # -> 0.2 mm per crop pixel
    "max_offset_mm":  11.5,    # largest correction Stage 2 is trained for / allowed to apply
    # ── synthetic starting errors: distance uniform in [min, max], uniform direction ──
    "rumore_min_mm":  0.5,
    "rumore_max_mm":  11.5,
    # ── augmentation ──
    "rotazione_max_gradi": 10.0,
    # ── loss ──
    "peso_nll":       1.0,
}


def collate_originali(batch):
    """Keeps the original-resolution images (list of [1, H, W]) and the
    landmark coordinates in pixels."""
    immagini, punti, masks, _, pixel_sizes = zip(*batch)
    return list(immagini), torch.stack(list(punti)), torch.stack(list(masks)), torch.stack(list(pixel_sizes))


def prepara_dati(config):
    """Same train/validation split as every other script (crea_split)."""
    dataset = Datasets(aariz_path=config["aariz_path"], isbi_path=config["isbi_path"],
                       tracciati_path=config.get("tracciati_path"),
                       tracciati_tabella_scale=config.get("tracciati_tabella_scale"))
    train_ds, val_ds = crea_split(dataset, config["val_split"], seed=42)
    kw = dict(batch_size=config["batch_size"], num_workers=config.get("num_workers", 0),
              collate_fn=collate_originali)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)


def errore_sintetico(K, config, generator=None):
    """[K, 2] synthetic errors in mm: distance uniform in [rumore_min_mm,
    rumore_max_mm], direction uniform over 360 degrees."""
    r0, r1 = config["rumore_min_mm"], config["rumore_max_mm"]
    raggio = r0 + torch.rand(K, generator=generator) * (r1 - r0)
    angolo = torch.rand(K, generator=generator) * (2 * torch.pi)
    return torch.stack([raggio * torch.cos(angolo), raggio * torch.sin(angolo)], dim=1)


def genera_batch(immagini, punti, masks, ps, config, device, training, generator=None):
    """
    Builds the Stage 2 samples of a batch: one crop per annotated landmark,
    centered on (true position + synthetic error).
    Returns crops [K,1,S,S], landmark indices [K], target offsets [K,2] in mm
    (from the crop center to the true point), or None if the batch is empty.
    """
    b_idx, l_idx = (masks == 1).nonzero(as_tuple=True)
    if len(b_idx) == 0:
        return None
    K = len(b_idx)
    errore_mm = errore_sintetico(K, config, generator)
    dims = torch.tensor([[im.shape[2], im.shape[1]] for im in immagini], dtype=torch.float32)
    mm_per_unita = dims[b_idx] * ps[b_idx].view(-1, 1)                 # mm per normalized unit
    vero = punti[b_idx, l_idx] / dims[b_idx]                             # normalized
    partenza = vero + errore_mm / mm_per_unita
    target_mm = -errore_mm

    crops = ritaglia_originali(immagini, partenza.to(device), b_idx.to(device), ps,
                               config["finestra_mm"], config["crop_px"])
    if training:
        crops, target_mm = augmenta_crop(crops, target_mm.to(device), config)
    return crops, l_idx.to(device), target_mm.to(device)


def augmenta_crop(crops, target_mm, config):
    """Random rotation (target rotated with the crop), contrast/brightness,
    Gaussian noise and occasional blur on standardized crops."""
    K = crops.shape[0]
    f = config["finestra_mm"]
    out = torch.empty_like(crops)
    t = target_mm.clone()
    for k in range(K):
        a = random.uniform(-config["rotazione_max_gradi"], config["rotazione_max_gradi"])
        out[k] = TF.rotate(crops[k], a, interpolation=TF.InterpolationMode.BILINEAR)
        t_norm = (target_mm[k] / f + 0.5).view(1, 2)
        t[k] = (ruota_coordinate(t_norm, a).view(2) - 0.5) * f
    contrasto = torch.empty(K, 1, 1, 1, device=crops.device).uniform_(0.7, 1.3)
    luminosita = torch.empty(K, 1, 1, 1, device=crops.device).uniform_(-0.3, 0.3)
    out = out * contrasto + luminosita + torch.randn_like(out) * random.uniform(0.0, 0.15)
    if random.random() < 0.3:
        out = TF.gaussian_blur(out, kernel_size=5, sigma=random.uniform(0.3, 1.2))
    return out, t


def calcola_loss(model, crops, l_idx, target_mm, config):
    pred_mm, logits = model(crops, l_idx)
    t_norm = (target_mm / config["finestra_mm"] + 0.5).clamp(0, 1)
    loss = F.huber_loss(pred_mm, target_mm, delta=1.0)
    if config["peso_nll"] > 0:
        uno = torch.ones(len(l_idx), 1, device=crops.device)
        loss = loss + config["peso_nll"] * nll_heatmap_loss(logits.unsqueeze(1), t_norm.unsqueeze(1), uno)
    return loss, pred_mm


def train_un_epoca(model, loader, optimizer, device, config):
    model.train()
    tot, n = 0.0, 0
    for immagini, punti, masks, ps in loader:
        lotto = genera_batch(immagini, punti, masks, ps, config, device, training=True)
        if lotto is None:
            continue
        loss, _ = calcola_loss(model, *lotto, config)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        tot += loss.item(); n += 1
    return tot / max(n, 1)


def valida(model, loader, device, config):
    """Loss and radial errors (mm) before/after Stage 2 on fixed synthetic errors."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    tot, n = 0.0, 0
    prima, dopo = [], []
    with torch.no_grad():
        for immagini, punti, masks, ps in loader:
            lotto = genera_batch(immagini, punti, masks, ps, config, device, training=False, generator=gen)
            if lotto is None:
                continue
            crops, l_idx, target_mm = lotto
            loss, pred_mm = calcola_loss(model, crops, l_idx, target_mm, config)
            tot += loss.item(); n += 1
            prima.append(target_mm.norm(dim=1).cpu())
            dopo.append((target_mm - pred_mm).norm(dim=1).cpu())
    prima, dopo = torch.cat(prima), torch.cat(dopo)
    return tot / max(n, 1), prima, dopo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 2 on the original radiograph: window {CONFIG['finestra_mm']} mm "
          f"({CONFIG['finestra_mm'] / CONFIG['crop_px']:.2f} mm/px), max offset {CONFIG['max_offset_mm']} mm, "
          f"synthetic error distance uniform in [{CONFIG['rumore_min_mm']}, {CONFIG['rumore_max_mm']}] mm")
    if CONFIG["rumore_max_mm"] > 0.5 * CONFIG["finestra_mm"] - 3.0:
        print("[WARN] rumore_max_mm is close to the crop border: enlarge finestra_mm")

    train_loader, val_loader = prepara_dati(CONFIG)
    model = ImageRefinementNet(num_landmarks=CONFIG["num_landmarks"], finestra_mm=CONFIG["finestra_mm"],
                               crop_px=CONFIG["crop_px"], max_offset_mm=CONFIG["max_offset_mm"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5, min_lr=1e-6)

    migliore, senza_miglioramento = float("inf"), 0
    for epoca in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_un_epoca(model, train_loader, optimizer, device, CONFIG)
        val_loss, prima, dopo = valida(model, val_loader, device, CONFIG)
        scheduler.step(val_loss)
        migliorati = (dopo < prima).float().mean().item()
        print(f"Epoch {epoca:3d}/{CONFIG['num_epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | val error {prima.mean():.2f} -> {dopo.mean():.2f} mm "
              f"(median {prima.median():.2f} -> {dopo.median():.2f}) | improved {migliorati:.1%}")
        if val_loss < migliore:
            migliore, senza_miglioramento = val_loss, 0
            torch.save({
                "epoca": epoca,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG,
                "arch": "ImageRefinementNet",
                "tipo_input": "immagine",
                "finestra_mm": CONFIG["finestra_mm"],
                "crop_px": CONFIG["crop_px"],
                "max_offset_mm": CONFIG["max_offset_mm"],
            }, CONFIG["save_path"])
            print(f"  → saved (val_loss: {val_loss:.4f})")
        else:
            senza_miglioramento += 1
            if senza_miglioramento >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoca}")
                break


if __name__ == "__main__":
    main()
