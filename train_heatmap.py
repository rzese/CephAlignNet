import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from functools import partial

from src.datasets import Datasets
from src.heatmap_utils import (coordinate_a_heatmap, heatmap_loss, augmenta, zoom_crop_augmenta, combined_loss, wing_loss_coord, geometric_consistency_loss, heatmap_a_coordinate, HEATMAP_TRAIN_SIZE)

try:
    from src.model_hrnet import HRNetLandmark as ModelClass
    MODEL_NAME = "HRNetLandmark (W48)"
except Exception as e:
    print(f"[WARN] HRNet not available ({e})")
    raise

CONFIG = {
    "aariz_path":      "data/Aariz",
    "isbi_path":       "data/Cepha400",
    "tracciati_path":  None, # "data/tracciati_forniti",  # 100 new in-house images
    "num_landmarks":   26,   # 19 original + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
    "img_size":        512,
    "heatmap_size":    HEATMAP_TRAIN_SIZE,   # 256
    "sigma":           4.0,
    "batch_size":      16,
    "num_epochs":      600,        # more epochs: more data, slower convergence
    "learning_rate":   3e-5,
    "val_split":       0.2,
    "save_path":       "modello_heatmap.pth",
    "patience":        60,         # more patience for difficult landmarks
    "geo_loss_weight": 0.05,       # weight of the geometric consistency loss
}


def collate_con_heatmap(batch, img_size=512, heatmap_size=256, sigma=4.0, training=False):
    """
    Collate function for the DataLoader: resizes images to img_size,
    normalizes landmark coordinates to [0, 1], zero-pads to 26 landmarks,
    optionally applies the training augmentation pipeline (plus a 40%
    chance of an extra zoom-crop on hard landmarks), and finally renders
    the target Gaussian heatmaps at heatmap_size.
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

    heatmaps = [coordinate_a_heatmap(pts, msk, img_size, heatmap_size, sigma)
                for pts, msk in zip(punti_pad, masks_pad)]
    return (torch.stack(list(imgs_r)), torch.stack(list(punti_pad)), torch.stack(list(masks_pad)), torch.stack(heatmaps), torch.stack(bboxes), torch.stack(pixel_sizes))


def prepara_dati(config):
    """Builds the aggregated Datasets object, splits it into train/val
    (fixed seed for reproducibility), and returns the two DataLoaders."""
    dataset = Datasets(aariz_path=config["aariz_path"], isbi_path=config["isbi_path"], tracciati_path=config.get("tracciati_path"))
    n_val   = int(len(dataset) * config["val_split"])
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val], generator=torch.Generator().manual_seed(42))
    c_tr = partial(collate_con_heatmap, img_size=config["img_size"], heatmap_size=config["heatmap_size"], sigma=config["sigma"], training=True)
    c_va = partial(collate_con_heatmap, img_size=config["img_size"], heatmap_size=config["heatmap_size"], sigma=config["sigma"], training=False)
    return (DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False, collate_fn=c_tr), DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False, collate_fn=c_va))


def train_un_epoca(model, loader, optimizer, device, config):
    """Runs one training epoch and returns the average training loss."""
    model.train()
    loss_totale = 0.0
    for immagini, punti, mask, heatmaps_target, _, _ in loader:
        immagini = immagini.to(device)
        punti = punti.to(device)
        mask = mask.to(device)
        heatmaps_target = heatmaps_target.to(device)

        pred, _ = model(immagini)
        if pred.shape[2:] != heatmaps_target.shape[2:]:
            pred = F.interpolate(pred, size=heatmaps_target.shape[2:], mode="bilinear", align_corners=False)

        # Combined loss: AWing on the heatmaps + L1 on the offset maps
        loss_comb = combined_loss(pred, heatmaps_target, punti, mask)

        # Auxiliary Wing loss on coordinates (soft-argmax)
        pred_hm  = pred[:, 0::3]
        pred_pts = heatmap_a_coordinate(pred_hm)
        loss_coord = wing_loss_coord(pred_pts, punti, mask)

        # Geometric consistency loss between related landmarks
        loss_geo = geometric_consistency_loss(pred_pts, mask)

        geo_w = config.get("geo_loss_weight", 0.05)
        loss = 0.7 * loss_comb + (0.3 - geo_w) * loss_coord + geo_w * loss_geo

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        loss_totale += loss.item()
    return loss_totale / len(loader)


def valida(model, loader, device, config):
    """Runs one validation pass (no gradient updates) and returns the
    average validation loss, using the same combined loss as training."""
    model.eval()
    loss_totale = 0.0
    with torch.no_grad():
        for immagini, punti, mask, heatmaps_target, _, _ in loader:
            immagini = immagini.to(device)
            punti = punti.to(device)
            mask = mask.to(device)
            heatmaps_target = heatmaps_target.to(device)

            pred, _ = model(immagini)
            if pred.shape[2:] != heatmaps_target.shape[2:]:
                pred = F.interpolate(pred, size=heatmaps_target.shape[2:],
                                     mode="bilinear", align_corners=False)

            loss_comb = combined_loss(pred, heatmaps_target, punti, mask)
            pred_hm = pred[:, 0::3]
            pred_pts = heatmap_a_coordinate(pred_hm)
            loss_coord = wing_loss_coord(pred_pts, punti, mask)
            loss_geo = geometric_consistency_loss(pred_pts, mask)
            geo_w = config.get("geo_loss_weight", 0.05)
            loss_totale += (0.7 * loss_comb + (0.3 - geo_w) * loss_coord + geo_w * loss_geo).item()
    return loss_totale / len(loader)


def main():
    """CLI entry point: trains Stage 1 (HRNet heatmap model) with early
    stopping and LR scheduling on validation loss, saving the best
    checkpoint to CONFIG['save_path']."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {MODEL_NAME}")
    print(f"batch_size={CONFIG['batch_size']} | LR={CONFIG['learning_rate']:.1e} "
          f"| img_size={CONFIG['img_size']} | heatmap_size={CONFIG['heatmap_size']}")

    train_loader, val_loader = prepara_dati(CONFIG)
    model = ModelClass(num_landmarks=CONFIG["num_landmarks"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5, min_lr=5e-7
    )

    migliore_val_loss = float("inf")
    epoche_senza_miglio = 0
    epoca_inizio  = 1

    for epoca in range(epoca_inizio, CONFIG["num_epochs"] + 1):
        train_loss = train_un_epoca(model, train_loader, optimizer, device, CONFIG)
        val_loss   = valida(model, val_loader, device, CONFIG)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoca:3d}/{CONFIG['num_epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < migliore_val_loss:
            migliore_val_loss   = val_loss
            epoche_senza_miglio = 0
            torch.save({
                "epoca":               epoca,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":            val_loss,
                "config":              CONFIG,
                "model_name":          MODEL_NAME,
                "backbone":            "hrnet_w48",
            }, CONFIG["save_path"])
            print(f"  → saved (val_loss: {val_loss:.6f})")
        else:
            epoche_senza_miglio += 1
            if epoche_senza_miglio >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoca}")
                break

    # ── Automatic generation of src/landmark_stats.py ──────────────────
    # Computes the per-landmark MRE of the checkpoint just saved and regenerates
    # REFINE_LANDMARK_MASK / PESI_LANDMARK / SIGMA_PER_LANDMARK, so there is no
    # need to run analyze_landmark_difficulty.py by hand anymore. If this is
    # the very first training run, the file does not exist yet and this
    # training used the default (fallback) values — the values generated
    # here are ready for a subsequent fine-tuning retrain.
    try:
        from src.analiz_land import genera_landmark_stats
        print("\nAutomatically generating src/landmark_stats.py from the best checkpoint...")
        genera_landmark_stats(stage1_path=CONFIG["save_path"],
                               aariz_path=CONFIG["aariz_path"],
                               isbi_path=CONFIG["isbi_path"],
                               device=device)
    except Exception as e:
        print(f"[WARN] Automatic generation of landmark_stats.py failed ({e}). "
              f"You can re-run it by hand with: python analyze_landmark_difficulty.py")


if __name__ == "__main__":
    main()