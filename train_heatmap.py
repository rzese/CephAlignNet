import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

from src.datasets import Datasets, crea_split, impronta_split
from src.heatmap_utils import (coordinate_a_heatmap, augmenta, zoom_crop_augmenta, combined_loss,
                               wing_loss_coord, geometric_consistency_loss, heatmap_a_coordinate,
                               HEATMAP_TRAIN_SIZE, SIGMA_PER_LANDMARK)
from src.incertezza import SigmaAddestrabile, MODALITA_HEATMAP, TEMPERATURA_DECODIFICA

try:
    from src.model_hrnet import HRNetLandmark as ModelClass
    MODEL_NAME = "HRNetLandmark (W48)"
except Exception as e:
    print(f"[WARN] HRNet not available ({e})")
    raise

NOMI = ["A","ANS","B","Me","N","Or","Pog","PNS","Pn","R","S","Ar","Co","Gn","Go","Po",
        "LPM","LIT","LMT","UIT","UIA","LIA","Ls","Li","Pog`","Sn"]

CONFIG = {
    "aariz_path":      "data/Aariz",
    "isbi_path":       "data/Cepha400",
    "tracciati_path":  None,   # e.g. "data/tracciati_forniti" (100 in-house images); not used for now
    # CSV/Excel with columns case_id, scale_px_per_mm (e.g. coordinate_radiografie.xlsx).
    # Without it Tracciati use the global fallback scale (see dataset_tracciati.py).
    "tracciati_tabella_scale": None,
    "num_landmarks":   26,   # 19 original + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
    "img_size":        512,
    "heatmap_size":    HEATMAP_TRAIN_SIZE,   # 256
    "sigma":           4.0,
    "batch_size":      2,
    "num_epochs":      600,        # more epochs: more data, slower convergence
    "learning_rate":   3e-5,
    "val_split":       0.2,
    "save_path":       "modello_heatmap.pth",
    "patience":        60,         # more patience for difficult landmarks
    "geo_loss_weight": 0.05,       # weight of the geometric consistency loss

    # ── Heatmap / uncertainty ────────────────────────────────────────────
    # "fixed"         fixed-sigma Gaussian targets + Adaptive Wing (original behaviour)
    # "learned_sigma" target sigma (sx, sy, rho) of every landmark trained with the network
    # "probabilistic" heatmap = logits of a spatial distribution, trained with NLL
    # See src/incertezza.py.
    "heatmap_mode":    "fixed",
    "griglia":         "centri",   # pixel-grid convention (see heatmap_utils.GRIGLIE)
    # learned_sigma only
    "sigma_penalita":  0.1,        # weight of the ellipse-area penalty (prevents sigma -> large)
    "lr_sigma":        1e-3,       # sigmas need a much larger LR than the network
    "sigma_min":       1.0,        # heatmap px
    "sigma_max":       12.0,       # heatmap px
}


def collate_con_heatmap(batch, img_size=512, heatmap_size=256, sigma=4.0, training=False,
                        genera_heatmap=True, griglia="centri"):
    """
    Collate function for the DataLoader: resizes images to img_size,
    normalizes landmark coordinates to [0, 1], zero-pads to 26 landmarks,
    optionally applies the training augmentation pipeline (plus a 40%
    chance of an extra zoom-crop on hard landmarks), and finally renders
    the target Gaussian heatmaps at heatmap_size (only for heatmap_mode
    "fixed": the other modes build their targets on the GPU, see calcola_loss).
    """
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    imgs_r = [F.interpolate(img.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
              for img in immagini]
    punti_pad, masks_pad = [], []
    for img_orig, p, m in zip(immagini, punti, masks):
        p_s = p.clone()
        p_s[:, 0] = p[:, 0] / img_orig.shape[2]
        p_s[:, 1] = p[:, 1] / img_orig.shape[1]
        n = p_s.shape[0]
        pp = torch.zeros(26, 2); pm = torch.zeros(26)   # 22 original + Ls + Li + Pog` + Sn
        pp[:n] = p_s[:n]; pm[:n] = m[:n]
        punti_pad.append(pp); masks_pad.append(pm)

    if training:
        augmented = []
        for img, pts, msk in zip(imgs_r, punti_pad, masks_pad):
            img_a, pts_a, msk_a = augmenta(img, pts.clone(), msk.clone())
            # With 40% probability: zoom-crop on hard landmarks (S, Or, LPM, UPM, Po)
            # Provides extra local detail where the model tends to make the most mistakes
            if random.random() < 0.4:
                img_a, pts_a, msk_a = zoom_crop_augmenta(img_a, pts_a, msk_a,
                                                          zoom_factor=2.2)
            augmented.append((img_a, pts_a, msk_a))
        imgs_r, punti_pad, masks_pad = zip(*augmented)

    if genera_heatmap:
        heatmaps = torch.stack([coordinate_a_heatmap(pts, msk, img_size, heatmap_size, sigma, griglia=griglia)
                                for pts, msk in zip(punti_pad, masks_pad)])
    else:
        heatmaps = torch.zeros(len(imgs_r))   # placeholder, targets built on the GPU
    return (torch.stack(list(imgs_r)), torch.stack(list(punti_pad)), torch.stack(list(masks_pad)), heatmaps, torch.stack(bboxes), torch.stack(pixel_sizes))


def prepara_dati(config):
    """Builds the aggregated Datasets object, splits it into train/val
    (fixed seed for reproducibility, same split as every other script),
    and returns the two DataLoaders."""
    dataset = Datasets(aariz_path=config["aariz_path"], isbi_path=config["isbi_path"],
                       tracciati_path=config.get("tracciati_path"),
                       tracciati_tabella_scale=config.get("tracciati_tabella_scale"))
    train_ds, val_ds = crea_split(dataset, config["val_split"], seed=42)
    fisso = config["heatmap_mode"] == "fixed"
    comuni = dict(img_size=config["img_size"], heatmap_size=config["heatmap_size"], sigma=config["sigma"],
                  genera_heatmap=fisso, griglia=config["griglia"])
    c_tr = partial(collate_con_heatmap, training=True, **comuni)
    c_va = partial(collate_con_heatmap, training=False, **comuni)
    return (DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False, collate_fn=c_tr), DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False, collate_fn=c_va))


def calcola_loss(model, sigma_mod, immagini, punti, mask, heatmaps_target, config):
    """
    Total Stage 1 loss for one batch, for every heatmap_mode:
      0.7 * (heatmap loss + offset loss)
      + (0.3 - geo_w) * Wing loss on soft-argmax coordinates
      + geo_w * geometric consistency loss
      [+ sigma_penalita * ellipse area, for learned_sigma]
    """
    modo, griglia = config["heatmap_mode"], config["griglia"]
    pred, _ = model(immagini)
    hs = config["heatmap_size"]
    if pred.shape[2] != hs or pred.shape[3] != hs:
        pred = F.interpolate(pred, size=(hs, hs), mode="bilinear", align_corners=False)

    if modo == "fixed":
        loss_comb = combined_loss(pred, heatmaps_target, punti, mask, griglia=griglia)
    elif modo == "learned_sigma":
        target = sigma_mod.genera(punti, mask, hs, griglia=griglia)
        loss_comb = combined_loss(pred, target, punti, mask, tipo_loss_heatmap="mse", griglia=griglia)
    else:  # probabilistic
        loss_comb = combined_loss(pred, None, punti, mask, tipo_loss_heatmap="nll", griglia=griglia)

    # Auxiliary Wing loss on coordinates (soft-argmax)
    pred_pts = heatmap_a_coordinate(pred[:, 0::3], temperatura=TEMPERATURA_DECODIFICA[modo], griglia=griglia)
    loss_coord = wing_loss_coord(pred_pts, punti, mask)

    # Geometric consistency loss between related landmarks; the face direction
    # is read from the ground truth, so flipped images are handled correctly
    loss_geo = geometric_consistency_loss(pred_pts, mask, gt_pts=punti)

    geo_w = config.get("geo_loss_weight", 0.05)
    loss = 0.7 * loss_comb + (0.3 - geo_w) * loss_coord + geo_w * loss_geo
    if modo == "learned_sigma":
        loss = loss + config["sigma_penalita"] * sigma_mod.penalita()
    return loss


def train_un_epoca(model, sigma_mod, loader, optimizer, device, config):
    """Runs one training epoch and returns the average training loss."""
    model.train()
    loss_totale = 0.0
    for immagini, punti, mask, heatmaps_target, _, _ in loader:
        immagini, punti, mask = immagini.to(device), punti.to(device), mask.to(device)
        heatmaps_target = heatmaps_target.to(device)

        loss = calcola_loss(model, sigma_mod, immagini, punti, mask, heatmaps_target, config)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        loss_totale += loss.item()
    return loss_totale / len(loader)


def valida(model, sigma_mod, loader, device, config):
    """Runs one validation pass (no gradient updates) and returns the
    average validation loss, using the same loss as training."""
    model.eval()
    loss_totale = 0.0
    with torch.no_grad():
        for immagini, punti, mask, heatmaps_target, _, _ in loader:
            immagini, punti, mask = immagini.to(device), punti.to(device), mask.to(device)
            heatmaps_target = heatmaps_target.to(device)
            loss_totale += calcola_loss(model, sigma_mod, immagini, punti, mask,
                                        heatmaps_target, config).item()
    return loss_totale / len(loader)


def main():
    """CLI entry point: trains Stage 1 (HRNet heatmap model) with early
    stopping and LR scheduling on validation loss, saving the best
    checkpoint to CONFIG['save_path']."""
    if CONFIG["heatmap_mode"] not in MODALITA_HEATMAP:
        raise ValueError(f"heatmap_mode must be one of {MODALITA_HEATMAP}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {MODEL_NAME} | heatmap_mode: {CONFIG['heatmap_mode']}")
    print(f"batch_size={CONFIG['batch_size']} | LR={CONFIG['learning_rate']:.1e} "
          f"| img_size={CONFIG['img_size']} | heatmap_size={CONFIG['heatmap_size']}")

    train_loader, val_loader = prepara_dati(CONFIG)
    model = ModelClass(num_landmarks=CONFIG["num_landmarks"]).to(device)

    sigma_mod = None
    gruppi = [{"params": model.parameters(), "lr": CONFIG["learning_rate"], "weight_decay": 1e-4}]
    if CONFIG["heatmap_mode"] == "learned_sigma":
        # Initialized from the per-landmark sigmas of src/landmark_stats.py (if present)
        sigma_mod = SigmaAddestrabile(num_landmarks=CONFIG["num_landmarks"], sigma_iniziale=CONFIG["sigma"],
                                      sigma_per_landmark=SIGMA_PER_LANDMARK,
                                      sigma_min=CONFIG["sigma_min"], sigma_max=CONFIG["sigma_max"],
                                      sigma_rif=CONFIG["sigma"]).to(device)
        gruppi.append({"params": sigma_mod.parameters(), "lr": CONFIG["lr_sigma"], "weight_decay": 0.0})
    optimizer = optim.AdamW(gruppi)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5, min_lr=5e-7
    )
    # ~mm per heatmap pixel for a 2400 px image at 0.1 mm/px (only for the logs)
    mm_per_px_hm = 2400 / CONFIG["heatmap_size"] * 0.1

    migliore_val_loss = float("inf")
    epoche_senza_miglio = 0
    epoca_inizio  = 1

    for epoca in range(epoca_inizio, CONFIG["num_epochs"] + 1):
        train_loss = train_un_epoca(model, sigma_mod, train_loader, optimizer, device, CONFIG)
        val_loss   = valida(model, sigma_mod, val_loader, device, CONFIG)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoca:3d}/{CONFIG['num_epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}")
        if sigma_mod is not None and epoca % 10 == 0:
            sx, sy, _ = sigma_mod.parametri()
            print(f"  learned sigma (heatmap px): mean sx={sx.mean():.2f} sy={sy.mean():.2f} | "
                  f"min {min(sx.min(), sy.min()):.2f} | max {max(sx.max(), sy.max()):.2f}")

        if val_loss < migliore_val_loss:
            migliore_val_loss   = val_loss
            epoche_senza_miglio = 0
            torch.save({
                "epoca":               epoca,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sigma_state_dict":    sigma_mod.state_dict() if sigma_mod is not None else None,
                "val_loss":            val_loss,
                "config":              CONFIG,
                "model_name":          MODEL_NAME,
                "backbone":            "hrnet_w48",
                "split_impronta":      impronta_split(),
            }, CONFIG["save_path"])
            print(f"  → saved (val_loss: {val_loss:.6f})")
        else:
            epoche_senza_miglio += 1
            if epoche_senza_miglio >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoca}")
                break

    if sigma_mod is not None:
        ckpt = torch.load(CONFIG["save_path"], map_location=device, weights_only=False)
        sigma_mod.load_state_dict(ckpt["sigma_state_dict"])
        print("\nLearned sigma per landmark (best checkpoint) — a larger sigma means the "
              "landmark is intrinsically harder to locate:")
        print(sigma_mod.descrivi(NOMI, mm_per_px_hm))

    # ── Automatic generation of src/landmark_stats.py ──────────────────
    # Computes the per-landmark MRE of the checkpoint just saved and regenerates
    # REFINE_LANDMARK_MASK / PESI_LANDMARK / SIGMA_PER_LANDMARK, so there is no
    # need to run src/analiz_land.py by hand anymore. If this is the very
    # first training run, the file does not exist yet and this training used
    # the default (fallback) values — the values generated here are ready for
    # a subsequent fine-tuning retrain.
    try:
        from src.analiz_land import genera_landmark_stats
        print("\nAutomatically generating src/landmark_stats.py from the best checkpoint...")
        genera_landmark_stats(stage1_path=CONFIG["save_path"],
                               aariz_path=CONFIG["aariz_path"],
                               isbi_path=CONFIG["isbi_path"],
                               device=device)
    except Exception as e:
        print(f"[WARN] Automatic generation of landmark_stats.py failed ({e}). "
              f"You can re-run it by hand with: python -m src.analiz_land")


if __name__ == "__main__":
    main()
