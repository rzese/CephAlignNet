import os, math, random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from functools import partial
from torchvision.ops import roi_align

from src.datasets import Datasets
from src.model_hrnet import HRNetLandmark
from src.model_refine import (RefinementNet, MAX_OFFSET, ROI_OUTPUT_SIZE, ROI_HALF_SIZE, REFINE_LANDMARK_MASK)
from src.heatmap_utils import heatmap_a_coordinate, HEATMAP_TRAIN_SIZE

CONFIG = {
    "aariz_path":        "data/Aariz",
    "isbi_path":         "data/Cepha400",
    # Tracciati are deliberately NOT included in evaluation: SDR/MRE metrics
    # must be comparable to the baseline on the public gold-standard datasets.
    # tracciati_path intentionally absent.
    "num_landmarks":     26,   # 19 original + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
    "batch_size":        2,
    "val_split":         0.2,
    "img_size":          512,
    "heatmap_size":      HEATMAP_TRAIN_SIZE,
    "stage1_path":       "modello_heatmap.pth",
    "stage2_path":       "modello_refine.pth",
    "larghezza_orig_px": 2400,
    "sdr_thresholds_mm": [2.0, 2.5, 3.0, 4.0],
    "hrnet_channels":    [192, 384, 768, 1536],
    "usa_tta":           True,
    "tta_angoli":        [-5.0, -2.5, 0.0, 2.5, 5.0],
    "guardia_offset":    MAX_OFFSET * 1.2,
}

CATEGORIE = {range(1, 16): "skeletal", range(16, 22): "dental", range(22, 27): "soft tissue"}
NOMI = ["A","ANS","B","Me","N","Or","Pog","PNS","Pn","R","S","Ar","Co","Gn","Go","Po","LPM","LIT","LMT","UIT","UIA","LIA","Ls","Li","Pog`","Sn"]


def collate_fn(batch, img_size=512):
    """Resizes images to img_size, normalizes landmark coordinates to
    [0, 1], and zero-pads points/mask to NUM_LANDMARKS=26."""
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    imgs_r = [F.interpolate(img.unsqueeze(0), size=(img_size, img_size),
                             mode='bilinear', align_corners=False).squeeze(0)
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
    return (torch.stack(imgs_r), torch.stack(punti_pad),
            torch.stack(masks_pad), torch.stack(bboxes), torch.stack(pixel_sizes))


def carica_stage1(config, device):
    """
    Loads the Stage 1 checkpoint, trying several possible HRNet channel
    configurations until one matches the saved weights.
    """
    ckpt = torch.load(config["stage1_path"], map_location=device, weights_only=False)
    for canali in [config["hrnet_channels"], [192,384,768,1536], [128,256,512,1024], [64,128,256,512,1024], [64,128,256,512], [32,64,128,256]]:
        try:
            model = HRNetLandmark(num_landmarks=config["num_landmarks"],
                                  pretrained=False, hrnet_channels=canali).to(device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval()
            backbone = ckpt.get("backbone", "hrnet_w?")
            print(f"Stage 1 loaded ({backbone}, channels={canali}, "
                  f"epoch={ckpt.get('epoca','?')}, val_loss={ckpt.get('val_loss',0):.6f})")
            return model
        except RuntimeError as e:
            if "size mismatch" in str(e):
                continue
            raise
    raise RuntimeError("Could not load Stage 1.")


def decodifica(pred_raw, config):
    """Decodes raw Stage 1 heatmaps into normalized (x, y) coordinates
    via soft-argmax."""
    ts = config["heatmap_size"]
    if pred_raw.shape[2] != ts or pred_raw.shape[3] != ts:
        pred_raw = F.interpolate(pred_raw, size=(ts, ts), mode='bilinear', align_corners=False)
    return heatmap_a_coordinate(pred_raw[:, 0::3])


def ruota_e_de_ruota(coords, angle_deg):
    """Rotates a batch of coordinates back by -angle_deg around the image
    center, undoing a TTA rotation applied to the input images."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = coords.clone()
    cx = coords[:, :, 0] - 0.5; cy = coords[:, :, 1] - 0.5
    out[:, :, 0] = cos_t * cx - sin_t * cy + 0.5
    out[:, :, 1] = sin_t * cx + cos_t * cy + 0.5
    return out


def predici_con_tta(stage1, immagini, config):
    """
    Test-time augmentation for a batch: averages Stage 1 predictions across
    several rotated versions, a horizontal flip, and photometric variants.
    Returns the averaged coordinates and the fused feature map (x_out) from
    the un-augmented (angle=0) forward pass, for use by Stage 2.
    """
    angoli = config.get("tta_angoli", [-5.0, -2.5, 0.0, 2.5, 5.0])
    tutte  = []
    x_out_orig = None
    with torch.no_grad():
        for angle in angoli:
            if abs(angle) < 0.01:
                pred_raw, x_out_orig = stage1(immagini)
                tutte.append(decodifica(pred_raw, config))
            else:
                imgs_rot = TF.rotate(immagini, angle)
                pred_raw_rot, _ = stage1(imgs_rot)
                tutte.append(ruota_e_de_ruota(decodifica(pred_raw_rot, config), -angle))
        # Horizontal flip
        pred_raw_flip, _ = stage1(TF.hflip(immagini))
        pred_flip = decodifica(pred_raw_flip, config).clone()
        pred_flip[:, :, 0] = 1.0 - pred_flip[:, :, 0]
        tutte.append(pred_flip)
        # Photometric variants
        for _ in range(2):
            imgs_var = TF.adjust_brightness(immagini, random.uniform(0.88, 1.12))
            imgs_var = TF.adjust_contrast(imgs_var,   random.uniform(0.88, 1.12))
            pred_raw_var, _ = stage1(imgs_var)
            tutte.append(decodifica(pred_raw_var, config))
    return torch.stack(tutte).mean(dim=0), x_out_orig


def applica_stage2(stage2, x_out, coords_norm, config, device):
    """
    Applies Stage 2 refinement to the landmarks in REFINE_LANDMARK_MASK for
    a whole batch: crops a small ROI from x_out around each Stage 1
    prediction, regresses an (x, y) offset, and applies it only if it stays
    within a safety guard, to avoid large, unstable jumps.
    """
    guardia    = config.get("guardia_offset", MAX_OFFSET * 1.2)
    coords_out = coords_norm.clone()
    B, N, _    = coords_norm.shape
    _, _, H_feat, W_feat = x_out.shape
    with torch.no_grad():
        for b in range(B):
            for i in range(N):
                # Only applied to the difficult landmarks — the others are already fine
                if i not in REFINE_LANDMARK_MASK:
                    continue
                cx = coords_norm[b, i, 0].item()
                cy = coords_norm[b, i, 1].item()
                cx_f = cx * W_feat; cy_f = cy * H_feat
                x1 = max(0.0, cx_f - ROI_HALF_SIZE)
                y1 = max(0.0, cy_f - ROI_HALF_SIZE)
                x2 = min(float(W_feat), cx_f + ROI_HALF_SIZE)
                y2 = min(float(H_feat), cy_f + ROI_HALF_SIZE)
                rois   = torch.tensor([[b, x1, y1, x2, y2]],
                                      dtype=torch.float32, device=device)
                crop   = roi_align(x_out, rois, output_size=ROI_OUTPUT_SIZE,
                                   spatial_scale=1.0, aligned=True)
                idx    = torch.tensor([i], dtype=torch.long, device=device)
                offset = stage2(crop, idx)[0]
                dx, dy = offset[0].item(), offset[1].item()
                if abs(dx) <= guardia and abs(dy) <= guardia:
                    coords_out[b, i, 0] = cx + dx
                    coords_out[b, i, 1] = cy + dy
    return coords_out


def calcola_mre_sdr(stage1, stage2, loader, device, config):
    """
    Runs inference over the whole validation loader and computes, per
    landmark and globally: Mean Radial Error (MRE, in mm) and Successful
    Detection Rate (SDR) at each threshold in sdr_thresholds_mm.
    """
    stage1.eval()
    if stage2 is not None:
        stage2.eval()
    errori_per_landmark = [[] for _ in range(config["num_landmarks"])]
    with torch.no_grad():
        for immagini, punti, mask, _, pixel_sizes in loader:
            immagini = immagini.to(device)
            punti = punti.to(device)
            mask = mask.to(device)
            pixel_sizes = pixel_sizes.to(device)
            if config.get("usa_tta", True):
                pred_coords, x_out = predici_con_tta(stage1, immagini, config)
                pred_coords = pred_coords.to(device)
            else:
                pred_raw, x_out = stage1(immagini)
                pred_coords = decodifica(pred_raw, config).to(device)
            B, N = mask.shape
            pred_coords = pred_coords[:, :N, :]
            if stage2 is not None and x_out is not None:
                pred_coords = applica_stage2(stage2, x_out, pred_coords, config, device)
            distanze = torch.sqrt(((pred_coords - punti) ** 2).sum(dim=2))
            for b in range(B):
                mm = config["larghezza_orig_px"] * pixel_sizes[b].item()
                for i in range(N):
                    if mask[b, i].item() == 1.0:
                        errori_per_landmark[i].append(distanze[b, i].item() * mm)
    tutti  = [e for lista in errori_per_landmark for e in lista]
    mre    = np.mean(tutti) if tutti else 0.0
    mre_lm = [np.mean(lista) if lista else None for lista in errori_per_landmark]
    sdr    = {s: np.mean([e < s for e in tutti]) * 100 if tutti else 0.0
              for s in config["sdr_thresholds_mm"]}
    #Per-landmark SDR: % of samples within each threshold
    sdr_lm = []
    for lista in errori_per_landmark:
        if lista:
            sdr_lm.append({s: np.mean([e < s for e in lista]) * 100
                           for s in config["sdr_thresholds_mm"]})
        else:
            sdr_lm.append(None)
    return mre, mre_lm, sdr, sdr_lm


def stampa_risultati(titolo, mre, mre_lm, sdr, sdr_lm):
    """Prints a formatted MRE/SDR report: global metrics, metrics broken
    down by anatomical category (skeletal/dental/soft tissue), and a
    per-landmark table."""
    print(f"\n{'='*55}")
    print(titolo)
    print(f"{'='*55}")
    print(f"Global MRE: {mre:.2f} mm")
    schel = [mre_lm[i] for i in range(15)     if mre_lm[i] is not None]
    dent  = [mre_lm[i] for i in range(15, 22) if mre_lm[i] is not None]
    soft  = [mre_lm[i] for i in range(22, 26) if mre_lm[i] is not None]
    if schel: print(f"MRE skeletal   (1-15):  {np.mean(schel):.2f} mm")
    if dent:  print(f"MRE dental     (16-22): {np.mean(dent):.2f} mm")
    if soft:  print(f"MRE soft tissue (23-26): {np.mean(soft):.2f} mm")
    print("\nSDR per threshold (mm):")
    for s, v in sdr.items():
        print(f"  within {s}mm: {v:.1f}%")

    soglie = list(sdr.keys())
    # Per-landmark table header
    hdr_sdr = "  ".join(f"SDR{s:g}" for s in soglie)
    print(f"\n{'Landmark':<6} {'Cat':12}  {'MRE':>6}  {hdr_sdr}")
    print("-" * (6 + 14 + 8 + len(soglie) * 10))
    for i, err in enumerate(mre_lm):
        cat  = next((v for k, v in CATEGORIE.items() if (i+1) in k), "")
        nome = NOMI[i] if i < len(NOMI) else str(i+1)
        if err is not None:
            sdr_vals = "  ".join(
                f"{sdr_lm[i][s]:>6.1f}%" if sdr_lm[i] else " " * 7
                for s in soglie
            )
            print(f"  {nome:<5} [{cat:12s}]  {err:>5.2f}mm  {sdr_vals}")
        else:
            print(f"  {nome:<5} [{cat:12s}]  — no valid samples")


def main():
    """CLI entry point: evaluates Stage 1 alone and Stage 1+Stage 2 on the
    Aariz+ISBI validation split, printing MRE/SDR reports and the
    improvement from adding Stage 2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"TTA: {CONFIG['tta_angoli']} + flip + photometric")

    stage1 = carica_stage1(CONFIG, device)

    stage2 = None
    if os.path.exists(CONFIG["stage2_path"]):
        ckpt2 = torch.load(CONFIG["stage2_path"], map_location=device, weights_only=False)
        in_channels = ckpt2.get("in_channels", 240)
        stage2 = RefinementNet(num_landmarks=CONFIG["num_landmarks"], in_channels=in_channels).to(device)
        stage2.load_state_dict(ckpt2["model_state_dict"])
        arch = ckpt2.get("arch", "CNN")
        print(f"Stage 2 loaded ({arch}, in_ch={in_channels}, "
              f"epoch={ckpt2.get('epoca','?')}, val_loss={ckpt2.get('val_loss',0):.6f})")
    else:
        print("Stage 2 not found — evaluating Stage 1 only")

    dataset = Datasets(aariz_path=CONFIG["aariz_path"], isbi_path=CONFIG["isbi_path"])
    n_val   = int(len(dataset) * CONFIG["val_split"])
    _, val_ds = random_split(dataset, [len(dataset)-n_val, n_val], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, collate_fn=partial(collate_fn, img_size=CONFIG["img_size"]))

    mre1, mre_lm1, sdr1, sdr_lm1 = calcola_mre_sdr(stage1, None, val_loader, device, CONFIG)
    stampa_risultati("STAGE 1 — heatmap only", mre1, mre_lm1, sdr1, sdr_lm1)

    if stage2 is not None:
        mre2, mre_lm2, sdr2, sdr_lm2 = calcola_mre_sdr(stage1, stage2, val_loader, device, CONFIG)
        stampa_risultati("STAGE 1 + STAGE 2", mre2, mre_lm2, sdr2, sdr_lm2)
        print(f"\n{'='*55}\nIMPROVEMENT\n{'='*55}")
        print(f"MRE: {mre1:.2f} → {mre2:.2f} mm  ({mre1-mre2:+.2f} mm)")
        for s in CONFIG["sdr_thresholds_mm"]:
            d = sdr2[s] - sdr1[s]
            print(f"SDR {s}mm: {sdr1[s]:.1f}% → {sdr2[s]:.1f}%  ({d:+.1f}%)")


if __name__ == "__main__":
    main()