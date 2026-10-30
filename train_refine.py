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

Variants (VARIANTI, chosen with  esegui_esperimento.py refine --variante <name>):
  "base"  the original Stage 2 (ImageRefinementNet, 25.6 mm / 128 px crop,
          32 x 32 output, synthetic errors uniform in [0.5, 11] mm)
  "A"     ImageRefinementUNet: pretrained resnet34 encoder + U-Net decoder,
          38.4 mm / 256 px crop (0.15 mm/px), 128 x 128 output (0.3 mm cells),
          errors up to 15 mm, crop scale jitter +-10% (pixel-size errors)
  "B"     A + "hard" starting positions: 20% of the crops start on another
          landmark of the same image (within vicino_max_mm), 20% of the dental
          tips start one tooth away along the occlusal plane (7-11 mm). These are
          the real Stage 1 failures that random displacements do not reproduce.
  "C"     B with a stable training: crops shuffled before the memory slices
          (otherwise each slice holds the crops of ~5 images and the BatchNorm
          statistics jump from slice to slice), EMA of the weights (validation and
          checkpoint use the averaged model), lr 1e-4, longer patience, and no
          neighbour-landmark starts for the soft tissue points (Pn was pulled to Sn).
"""
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from src.datasets import Datasets, crea_split, impronta_split
from src.heatmap_utils import nll_heatmap_loss, ruota_coordinate
from src.inferenza import ritaglia_originali
from src.model_refine import crea_stage2_immagine

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
    "patience_lr":    8,       # ReduceLROnPlateau patience
    # ── crop and correction range ──
    "finestra_mm":    25.6,    # side of the crop (+-12.5 mm: the true point stays well inside even at 11.5 mm)
    "crop_px":        128,     # -> 0.2 mm per crop pixel
    "max_offset_mm":  11.0,    # largest correction Stage 2 is trained for / allowed to apply
    # ── synthetic starting errors: distance uniform in [min, max], uniform direction ──
    "rumore_min_mm":  0.5,
    "rumore_max_mm":  11.0,
    # ── augmentation ──
    "rotazione_max_gradi": 10.0,
    # ── loss ──
    "peso_nll":       1.0,
    # ── architecture ("base" = ImageRefinementNet, "unet" = ImageRefinementUNet) ──
    "arch":           "base",
    "encoder":        "resnet34",   # unet only (timm name, ImageNet weights)
    "pretrained":     True,         # unet only
    # ── extra augmentation / hard starting positions (0 = off) ──
    "scala_max":      0.0,          # crop scale jitter: mm/px multiplied by 1 +- scala_max
    "prob_vicino":    0.0,          # start on another landmark of the same image
    "vicino_max_mm":  15.0,
    "vicino_escludi": (),           # landmark indices that never get a neighbour start
    "prob_dente":     0.0,          # dental tips: start one tooth away along the occlusal plane
    "dente_mm":       (7.0, 11.0),
    # ── memory: crops per forward/backward pass (bf16 autocast on GPU) ──
    "max_crop_per_passo": 0,        # 0 = all the crops of the batch at once
    "amp":            False,
    # ── training stability (0 / False = off) ──
    "mescola_crop":   False,        # shuffle the crops of a batch before the memory slices
    "ema":            0.0,          # decay of the EMA of the weights (e.g. 0.999); 0 = no EMA
    "variante":       "base",
}

# Landmark indices (order of NOMI in train_heatmap.py)
_S, _N, _LPM, _LIT, _LMT, _UIT = 10, 4, 16, 17, 18, 19
PUNTE_DENTALI = (_LPM, _LIT, _LMT, _UIT)   # tips that can be confused with the next tooth
TESSUTI_MOLLI = (8, 22, 23, 24, 25)        # Pn, Ls, Li, Pog`, Sn

_VARIANTE_A = {
    "arch": "unet", "encoder": "resnet34", "pretrained": True,
    "finestra_mm": 38.4, "crop_px": 256, "max_offset_mm": 15.0, "rumore_max_mm": 15.0,
    "scala_max": 0.10, "learning_rate": 3e-4, "max_crop_per_passo": 128, "amp": True,
}
VARIANTI = {
    "base": {},
    "A": _VARIANTE_A,
    "B": {**_VARIANTE_A, "prob_vicino": 0.2, "prob_dente": 0.2},
}
VARIANTI["C"] = {**VARIANTI["B"], "mescola_crop": True, "ema": 0.998, "learning_rate": 1e-4,
                 "patience": 60, "patience_lr": 15, "vicino_escludi": TESSUTI_MOLLI}


def applica_variante(config, nome):
    """Updates config with the settings of variant nome (see VARIANTI)."""
    if nome not in VARIANTI:
        raise ValueError(f"unknown Stage 2 variant '{nome}', expected one of {list(VARIANTI)}")
    config.update(VARIANTI[nome])
    config["variante"] = nome


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


def _uniforme(a, b, generator=None):
    return a + (b - a) * torch.rand(1, generator=generator).item()


def direzione_occlusale(p_mm, m):
    """Unit vector (mm frame) from the incisors towards the molars of one image,
    or None. p_mm [N, 2] landmarks in mm, m [N] mask. Without the molar, the
    occlusal plane is taken as horizontal, pointing backwards (from N towards S)."""
    incisivi = [i for i in (_UIT, _LIT) if m[i] > 0]
    if incisivi and m[_LMT] > 0:
        v = p_mm[_LMT] - p_mm[incisivi].mean(dim=0)
    elif m[_S] > 0 and m[_N] > 0:
        v = torch.tensor([float(torch.sign(p_mm[_S, 0] - p_mm[_N, 0])) or 1.0, 0.0])
    else:
        return None
    n = v.norm()
    return v / n if n > 1e-6 else None


def errori_partenza(punti_mm, masks, b_idx, l_idx, config, generator=None):
    """
    [K, 2] starting error in mm (start position - true position) of every crop:
      - with probability prob_vicino: the position of another annotated landmark
        of the same image, between 2 and vicino_max_mm away (+ 0.5 mm jitter),
        except for the landmarks in vicino_escludi;
      - dental tips, with probability prob_dente: one tooth away along the
        occlusal plane (dente_mm, either direction, +-1 mm across);
      - otherwise (or when the above is not possible): the uniform synthetic
        error of errore_sintetico.
    Hard starts whose target would fall too close to the crop border fall back
    to the uniform error.
    """
    K = len(b_idx)
    err = errore_sintetico(K, config, generator)
    pv, pd = config.get("prob_vicino", 0.0), config.get("prob_dente", 0.0)
    if pv <= 0 and pd <= 0:
        return err
    limite = 0.5 * config["finestra_mm"] - 1.5
    direzioni = {}
    for k in range(K):
        b, l = int(b_idx[k]), int(l_idx[k])
        p, m = punti_mm[b], masks[b]
        u = torch.rand(1, generator=generator).item()
        nuovo = None
        if u < pv and l not in config.get("vicino_escludi", ()):
            dist = (p - p[l]).norm(dim=1)
            cand = ((m > 0) & (dist >= 2.0) & (dist <= config.get("vicino_max_mm", 15.0))).nonzero().flatten()
            cand = cand[cand != l]
            if len(cand):
                j = int(cand[torch.randint(len(cand), (1,), generator=generator)])
                nuovo = p[j] - p[l] + 0.5 * torch.randn(2, generator=generator)
        elif u < pv + pd and l in PUNTE_DENTALI:
            if b not in direzioni:
                direzioni[b] = direzione_occlusale(p, m)
            d = direzioni[b]
            if d is not None:
                lo, hi = config.get("dente_mm", (7.0, 11.0))
                segno = 1.0 if torch.rand(1, generator=generator).item() < 0.5 else -1.0
                perp = torch.tensor([-d[1], d[0]])
                nuovo = segno * _uniforme(lo, hi, generator) * d + _uniforme(-1.0, 1.0, generator) * perp
        if nuovo is not None and nuovo.abs().max() <= limite:
            err[k] = nuovo
    return err


def genera_batch(immagini, punti, masks, ps, config, device, training, generator=None):
    """
    Builds the Stage 2 samples of a batch: one crop per annotated landmark,
    centered on (true position + starting error, see errori_partenza).
    In training, with scala_max > 0, every image is cropped as if its pixel size
    were off by a factor in 1 +- scala_max (the target is scaled with it).
    Returns crops [K,1,S,S], landmark indices [K], target offsets [K,2] in mm
    (from the crop center to the true point), or None if the batch is empty.
    """
    b_idx, l_idx = (masks == 1).nonzero(as_tuple=True)
    if len(b_idx) == 0:
        return None
    dims = torch.tensor([[im.shape[2], im.shape[1]] for im in immagini], dtype=torch.float32)
    punti_mm = punti * ps.view(-1, 1, 1)
    errore_mm = errori_partenza(punti_mm, masks, b_idx, l_idx, config, generator)
    mm_per_unita = dims[b_idx] * ps[b_idx].view(-1, 1)                 # mm per normalized unit
    vero = punti[b_idx, l_idx] / dims[b_idx]                             # normalized
    partenza = vero + errore_mm / mm_per_unita
    target_mm = -errore_mm

    ps_crop = ps
    a = config.get("scala_max", 0.0)
    if training and a > 0:
        fattore = 1.0 + (2 * torch.rand(len(immagini), generator=generator) - 1) * a
        ps_crop = ps * fattore                     # the crop covers finestra_mm / fattore real mm
        target_mm = target_mm * fattore[b_idx].view(-1, 1)

    crops = ritaglia_originali(immagini, partenza.to(device), b_idx.to(device), ps_crop,
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
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.get("amp", False) and crops.is_cuda):
        pred_mm, logits = model(crops, l_idx)
    pred_mm, logits = pred_mm.float(), logits.float()
    t_norm = (target_mm / config["finestra_mm"] + 0.5).clamp(0, 1)
    loss = F.huber_loss(pred_mm, target_mm, delta=1.0)
    if config["peso_nll"] > 0:
        uno = torch.ones(len(l_idx), 1, device=crops.device)
        loss = loss + config["peso_nll"] * nll_heatmap_loss(logits.unsqueeze(1), t_norm.unsqueeze(1), uno)
    return loss, pred_mm


def _blocchi(K, config):
    """Slices of at most max_crop_per_passo crops (one slice if 0)."""
    passo = config.get("max_crop_per_passo", 0) or K
    return [slice(i, min(i + passo, K)) for i in range(0, K, passo)]


def train_un_epoca(model, loader, optimizer, device, config, ema=None):
    model.train()
    tot, n = 0.0, 0
    for immagini, punti, masks, ps in loader:
        lotto = genera_batch(immagini, punti, masks, ps, config, device, training=True)
        if lotto is None:
            continue
        crops, l_idx, target_mm = lotto
        K = len(l_idx)
        if config.get("mescola_crop", False):     # every slice mixes all the images of the batch
            perm = torch.randperm(K, device=crops.device)
            crops, l_idx, target_mm = crops[perm], l_idx[perm], target_mm[perm]
        optimizer.zero_grad()
        loss_lotto = 0.0
        for sl in _blocchi(K, config):           # gradient accumulated over the slices
            loss, _ = calcola_loss(model, crops[sl], l_idx[sl], target_mm[sl], config)
            (loss * (sl.stop - sl.start) / K).backward()
            loss_lotto += loss.item() * (sl.stop - sl.start) / K
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        if ema is not None:
            ema.update_parameters(model)
        tot += loss_lotto; n += 1
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
            for sl in _blocchi(len(l_idx), config):
                loss, pred_mm = calcola_loss(model, crops[sl], l_idx[sl], target_mm[sl], config)
                tot += loss.item(); n += 1
                prima.append(target_mm[sl].norm(dim=1).cpu())
                dopo.append((target_mm[sl] - pred_mm).norm(dim=1).cpu())
    prima, dopo = torch.cat(prima), torch.cat(dopo)
    return tot / max(n, 1), prima, dopo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 2 variant '{CONFIG.get('variante', 'base')}': arch {CONFIG['arch']}"
          + (f" ({CONFIG['encoder']}, pretrained={CONFIG['pretrained']})" if CONFIG["arch"] == "unet" else "")
          + f" | scale jitter +-{CONFIG.get('scala_max', 0.0):.0%} | hard starts: neighbour landmark "
          f"{CONFIG.get('prob_vicino', 0.0):.0%}, next tooth {CONFIG.get('prob_dente', 0.0):.0%}")
    print(f"Stage 2 on the original radiograph: window {CONFIG['finestra_mm']} mm "
          f"({CONFIG['finestra_mm'] / CONFIG['crop_px']:.2f} mm/px), max offset {CONFIG['max_offset_mm']} mm, "
          f"synthetic error distance uniform in [{CONFIG['rumore_min_mm']}, {CONFIG['rumore_max_mm']}] mm")
    if CONFIG["rumore_max_mm"] > 0.5 * CONFIG["finestra_mm"] - 3.0:
        print("[WARN] rumore_max_mm is close to the crop border: enlarge finestra_mm")

    train_loader, val_loader = prepara_dati(CONFIG)
    model = crea_stage2_immagine(CONFIG["arch"], num_landmarks=CONFIG["num_landmarks"],
                                 finestra_mm=CONFIG["finestra_mm"], crop_px=CONFIG["crop_px"],
                                 max_offset_mm=CONFIG["max_offset_mm"], encoder=CONFIG["encoder"],
                                 pretrained=CONFIG["pretrained"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=CONFIG.get("patience_lr", 8),
                                                     factor=0.5, min_lr=1e-6)
    # EMA of weights and BatchNorm buffers: validated and saved instead of the raw model
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(CONFIG["ema"]), use_buffers=True)
           if CONFIG.get("ema", 0.0) > 0 else None)
    if ema is not None:
        print(f"EMA of the weights, decay {CONFIG['ema']} | crops shuffled: {CONFIG.get('mescola_crop', False)}")

    migliore, senza_miglioramento = float("inf"), 0
    for epoca in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_un_epoca(model, train_loader, optimizer, device, CONFIG, ema)
        da_salvare = ema.module if ema is not None else model
        val_loss, prima, dopo = valida(da_salvare, val_loader, device, CONFIG)
        scheduler.step(val_loss)
        migliorati = (dopo < prima).float().mean().item()
        print(f"Epoch {epoca:3d}/{CONFIG['num_epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | val error {prima.mean():.2f} -> {dopo.mean():.2f} mm "
              f"(median {prima.median():.2f} -> {dopo.median():.2f}) | improved {migliorati:.1%}")
        if val_loss < migliore:
            migliore, senza_miglioramento = val_loss, 0
            torch.save({
                "epoca": epoca,
                "model_state_dict": da_salvare.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG,
                "arch": type(model).__name__,
                "arch_stage2": CONFIG["arch"],
                "encoder": CONFIG["encoder"],
                "variante": CONFIG.get("variante", "base"),
                "tipo_input": "immagine",
                "finestra_mm": CONFIG["finestra_mm"],
                "crop_px": CONFIG["crop_px"],
                "max_offset_mm": CONFIG["max_offset_mm"],
                "split_impronta": impronta_split(),
            }, CONFIG["save_path"])
            print(f"  → saved (val_loss: {val_loss:.4f})")
        else:
            senza_miglioramento += 1
            if senza_miglioramento >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoca}")
                break


if __name__ == "__main__":
    main()
