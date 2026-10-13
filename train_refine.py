import os, random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

from src.datasets import Datasets, crea_split
from src.model_refine import RefinementNet, MAX_OFFSET
from src.heatmap_utils import HEATMAP_TRAIN_SIZE
from src.inferenza import carica_stage1 as _carica_stage1, collate_base, decodifica, estrai_roi

CONFIG = {
    "aariz_path":     "data/Aariz",
    "isbi_path":      "data/Cepha400",
    "tracciati_path": None,   # e.g. "data/tracciati_forniti"; not used for now
    "tracciati_tabella_scale": None,   # see train_heatmap.py
    "num_landmarks":  26,   # 19 original + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
    "batch_size":     8,
    "num_epochs":     400,
    "learning_rate":  2e-4,
    "val_split":      0.2,
    "stage1_path":    "modello_heatmap.pth",
    "save_path":      "modello_refine.pth",
    "img_size":       512,
    "heatmap_size":   HEATMAP_TRAIN_SIZE,
    "scarto_soglia":  MAX_OFFSET * 1.5,
    "jitter_inizio":  MAX_OFFSET * 1.5,   # raised: forces Stage 2 to learn larger corrections
    "jitter_fine":    MAX_OFFSET * 0.2,
    "curriculum_ep":  150,
    "patience":       120,                # raised from 80: more time to converge
    # Checkpoint: if modello_refine.pth exists, resumes from it instead of starting over
    # Set to True to continue training, False to start from scratch
    "resume":         False,
}


def carica_stage1(config, device):
    """
    Loads the frozen Stage 1 model (weights not updated during Stage 2
    training), trying several possible HRNet channel configurations until
    one matches the saved checkpoint (see src/inferenza.py).
    """
    return _carica_stage1(config, device, congela=True)


def predici_stage1(stage1, imgs, config):
    """Runs Stage 1 on a batch and decodes its heatmaps into normalized
    coordinates (with the decoding of the loaded checkpoint), also returning
    the fused feature map (x_out) that Stage 2 will crop ROIs from."""
    with torch.no_grad():
        pred_raw, x_out = stage1(imgs)
        coords = decodifica(pred_raw, stage1, config["heatmap_size"])
    return coords, x_out


def collate_fn(batch, img_size=512):
    """Resizes images to img_size and normalizes landmark coordinates to
    [0, 1] (see src/inferenza.collate_base)."""
    return collate_base(batch, img_size)


def prepara_dati(config):
    """Builds the aggregated Datasets object, splits it into train/val
    (same split as train_heatmap.py and evaluate_refine.py), and
    returns the two DataLoaders."""
    dataset = Datasets(aariz_path=config["aariz_path"], isbi_path=config["isbi_path"],
                       tracciati_path=config.get("tracciati_path"),
                       tracciati_tabella_scale=config.get("tracciati_tabella_scale"))
    train_ds, val_ds = crea_split(dataset, config["val_split"], seed=42)
    fn = partial(collate_fn, img_size=config["img_size"])
    return (DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, collate_fn=fn), DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=fn))


def genera_batch_refine(stage1, imgs, punti, masks, config, device, training=False, epoca=0):
    """
    Builds a training/validation batch for Stage 2 from a batch of images:
    runs Stage 1 to get predicted coordinates, computes the (dx, dy) offset
    to the ground truth for every valid landmark, discards samples where
    Stage 1 was wildly wrong, and (during training) applies a curriculum
    jitter to the input coordinate so Stage 2 learns to correct non-trivial
    errors early on and finer ones later. Returns stacked (crops, targets,
    landmark_indices), or (None, None, None) if the batch produced no
    usable samples.
    """
    pred_coords, x_out = predici_stage1(stage1, imgs, config)
    crops_list, targets_list, indices_list = [], [], []
    soglia = config["scarto_soglia"]

    # Curriculum: jitter decreases linearly over epochs
    if training:
        t = min(epoca / max(config.get("curriculum_ep", 150), 1), 1.0)
        noise = (config.get("jitter_inizio", MAX_OFFSET * 0.8) * (1 - t) + config.get("jitter_fine",   MAX_OFFSET * 0.2) * t)
    else:
        noise = 0.0

    for b in range(pred_coords.shape[0]):
        for i in range(pred_coords.shape[1]):
            if masks[b, i].item() != 1.0:
                continue
            cx_pred = pred_coords[b, i, 0].item()
            cy_pred = pred_coords[b, i, 1].item()
            cx_real = punti[b, i, 0].item()
            cy_real = punti[b, i, 1].item()
            dx = cx_real - cx_pred
            dy = cy_real - cy_pred

            # Only discard huge errors (Stage 1 completely wrong)
            if abs(dx) > soglia or abs(dy) > soglia:
                continue

            dx = max(-MAX_OFFSET, min(MAX_OFFSET, dx))
            dy = max(-MAX_OFFSET, min(MAX_OFFSET, dy))

            if training and noise > 0:
                # Curriculum jitter: high early on to force the model to learn
                # non-trivial corrections, then reduced to refine precision
                cx_in = max(0.01, min(0.99, cx_pred + random.uniform(-noise, noise)))
                cy_in = max(0.01, min(0.99, cy_pred + random.uniform(-noise, noise)))
                dx = max(-MAX_OFFSET, min(MAX_OFFSET, cx_real - cx_in))
                dy = max(-MAX_OFFSET, min(MAX_OFFSET, cy_real - cy_in))
            else:
                cx_in, cy_in = cx_pred, cy_pred

            roi_crop = estrai_roi(x_out, torch.tensor([cx_in], device=device),
                                  torch.tensor([cy_in], device=device),
                                  torch.tensor([b], device=device))
            crops_list.append(roi_crop.squeeze(0))
            targets_list.append(torch.tensor([dx, dy], dtype=torch.float32))
            indices_list.append(torch.tensor(i, dtype=torch.long))

    if not crops_list:
        return None, None, None
    return (torch.stack(crops_list).to(device), torch.stack(targets_list).to(device), torch.stack(indices_list).to(device))


def train_un_epoca(stage1, refine, loader, optimizer, device, config, epoca=0):
    """Runs one Stage 2 training epoch (Stage 1 stays frozen/eval) and
    returns the average Huber loss on the predicted offsets."""
    stage1.eval(); refine.train()
    loss_totale, n_batch = 0.0, 0
    for imgs, punti, masks, _, _, _ in loader:
        imgs, punti, masks = imgs.to(device), punti.to(device), masks.to(device)
        crops, targets, indices = genera_batch_refine(stage1, imgs, punti, masks,config, device, training=True, epoca=epoca)
        if crops is None:
            continue
        loss = F.huber_loss(refine(crops, indices), targets, delta=MAX_OFFSET * 0.5)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(refine.parameters(), max_norm=2.0)
        optimizer.step()
        loss_totale += loss.item(); n_batch += 1
    return loss_totale / max(n_batch, 1)


def valida(stage1, refine, loader, device, config):
    """Runs one Stage 2 validation pass (no jitter, no gradient updates)
    and returns the average Huber loss."""
    stage1.eval(); refine.eval()
    loss_totale, n_batch = 0.0, 0
    with torch.no_grad():
        for imgs, punti, masks, _, _, _ in loader:
            imgs, punti, masks = imgs.to(device), punti.to(device), masks.to(device)
            crops, targets, indices = genera_batch_refine(stage1, imgs, punti, masks, config, device, training=False)
            if crops is None:
                continue
            loss_totale += F.huber_loss(refine(crops, indices), targets, delta=MAX_OFFSET * 0.5).item()
            n_batch += 1
    return loss_totale / max(n_batch, 1)


def main():
    """CLI entry point: trains Stage 2 (refinement network) on top of a
    frozen Stage 1 checkpoint, with curriculum jitter, LR scheduling,
    early stopping, and optional resume from a previous checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MAX_OFFSET: {MAX_OFFSET:.4f} (~{MAX_OFFSET*2400*0.1:.2f}mm on a 2400px image at 0.1mm/px)")
    print(f"Discard threshold: {CONFIG['scarto_soglia']:.4f}")

    stage1 = carica_stage1(CONFIG, device)

    with torch.no_grad():
        dummy = torch.zeros(1, 1, CONFIG["img_size"], CONFIG["img_size"], device=device)
        _, x_out = stage1(dummy)
        in_channels = x_out.shape[1]
        print(f"Feature map: {in_channels}ch × {x_out.shape[2]}×{x_out.shape[3]}")

    refine = RefinementNet(num_landmarks=CONFIG["num_landmarks"], in_channels=in_channels,
                           max_offset=MAX_OFFSET).to(device)

    optimizer = optim.AdamW([
        {"params": refine.stem.parameters(), "lr": CONFIG["learning_rate"] * 0.3},
        {"params": refine.res1.parameters(), "lr": CONFIG["learning_rate"] * 0.3},
        {"params": refine.res2.parameters(), "lr": CONFIG["learning_rate"] * 0.3},
        {"params": refine.res3.parameters(), "lr": CONFIG["learning_rate"] * 0.3},
        {"params": refine.down1.parameters(), "lr": CONFIG["learning_rate"] * 0.3},
        {"params": refine.down2.parameters(), "lr": CONFIG["learning_rate"] * 0.3},
        {"params": refine.landmark_embedding.parameters(), "lr": CONFIG["learning_rate"]},
        {"params": refine.regressor.parameters(), "lr": CONFIG["learning_rate"]},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', patience=25, factor=0.5, min_lr=1e-6)

# ── Checkpoint resume ───────────────────────────────────────────────────
    epoca_inizio      = 1
    migliore_val_loss = float('inf')
    epoche_no_migl    = 0

    if CONFIG.get("resume", False) and os.path.exists(CONFIG["save_path"]):
        try:
            ckpt = torch.load(CONFIG["save_path"], map_location=device, weights_only=False)
            refine.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            epoca_inizio = ckpt.get("epoca", 1) + 1
            migliore_val_loss = ckpt.get("val_loss", float('inf'))
            print(f"Resumed from epoch {epoca_inizio-1} | "
                  f"val_loss={migliore_val_loss:.6f}")
        except Exception as e:
            print(f"[WARN] Resume failed ({e}) — starting from scratch")
            epoca_inizio      = 1
            migliore_val_loss = float('inf')
    else:
        print("No checkpoint found — training from scratch")
    # ────────────────────────────────────────────────────────────────────────

    train_loader, val_loader = prepara_dati(CONFIG)
    PATIENCE = CONFIG.get("patience", 120)

    for epoca in range(epoca_inizio, CONFIG["num_epochs"] + 1):
        train_loss = train_un_epoca(stage1, refine, train_loader, optimizer, device, CONFIG, epoca=epoca)
        val_loss   = valida(stage1, refine, val_loader, device, CONFIG)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[6]['lr']
        t = min(epoca / CONFIG.get("curriculum_ep", 150), 1.0)
        jitter_now = (CONFIG.get("jitter_inizio", MAX_OFFSET*1.5) * (1-t) + CONFIG.get("jitter_fine",   MAX_OFFSET*0.2) * t)
        print(f"Epoch {epoca:3d}/{CONFIG['num_epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | jitter: {jitter_now:.5f}")
        if val_loss < migliore_val_loss:
            migliore_val_loss = val_loss; epoche_no_migl = 0
            torch.save({
                'epoca': epoca,
                'model_state_dict': refine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
                'max_offset': MAX_OFFSET,
                'arch': 'FeatureCropCNN',
                'in_channels': in_channels,
                'decodifica_stage1': 'soft-argmax',
                'heatmap_mode_stage1': stage1.heatmap_mode,
            }, CONFIG["save_path"])
            print(f"  → saved (val_loss: {val_loss:.6f})")
        else:
            epoche_no_migl += 1
            if epoche_no_migl >= PATIENCE:
                print(f"Early stopping at epoch {epoca}")
                break


if __name__ == "__main__":
    main()