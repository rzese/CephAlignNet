"""
inferenza.py — code shared by train_refine.py, evaluate_refine.py,
predict_heatmap.py and src/analiz_land.py.

Previously each script had its own copy of model loading, decoding, TTA and
Stage 2 application, so the same bug had to be fixed in several places.
"""
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.ops import roi_align

from src.model_hrnet import HRNetLandmark
from src.model_refine import (RefinementNet, MAX_OFFSET, ROI_OUTPUT_SIZE, ROI_HALF_SIZE,
                              REFINE_LANDMARK_MASK)
from src.heatmap_utils import (heatmap_a_coordinate, regression_voting, ruota_coordinate,
                               HEATMAP_TRAIN_SIZE)
from src.incertezza import TEMPERATURA_DECODIFICA, covarianza_tta

# HRNet channel layouts tried in turn until the checkpoint loads with strict=True
CANALI_CANDIDATI = [[64, 128, 256, 512, 1024], [192, 384, 768, 1536], None,
                    [128, 256, 512, 1024], [64, 128, 256, 512], [32, 64, 128, 256]]


# ─── Loading ──────────────────────────────────────────────────────────────────
def carica_stage1(config, device, congela=False):
    """
    Loads the Stage 1 checkpoint config["stage1_path"], trying several HRNet
    channel layouts until one matches the saved weights.

    The returned model carries the decoding settings it was trained with:
      model.heatmap_mode  "fixed" | "learned_sigma" | "probabilistic"
      model.griglia       pixel-grid convention ("legacy" for checkpoints
                          saved before the grid fix)
      model.temperatura   soft-argmax temperature for that mode
    congela=True also disables gradients (Stage 2 training).
    """
    ckpt = torch.load(config["stage1_path"], map_location=device, weights_only=False)
    candidati = [config.get("hrnet_channels")] + CANALI_CANDIDATI
    provati = []
    for canali in candidati:
        if canali in provati:
            continue
        provati.append(canali)
        try:
            model = HRNetLandmark(num_landmarks=config["num_landmarks"],
                                  pretrained=False, hrnet_channels=canali).to(device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as e:
            msg = str(e)
            if "size mismatch" in msg or "Missing key" in msg or "Unexpected key" in msg:
                continue
            raise
        model.eval()
        if congela:
            for p in model.parameters():
                p.requires_grad = False

        cfg_ckpt = ckpt.get("config") or {}
        model.heatmap_mode = cfg_ckpt.get("heatmap_mode", "fixed")
        model.griglia      = cfg_ckpt.get("griglia", "legacy")
        model.temperatura  = TEMPERATURA_DECODIFICA[model.heatmap_mode]
        model.sigma_state  = ckpt.get("sigma_state_dict")
        print(f"Stage 1 loaded ({ckpt.get('backbone', 'hrnet_w?')}, channels={model.hrnet_channels}, "
              f"epoch={ckpt.get('epoca', '?')}, val_loss={ckpt.get('val_loss', 0):.6f}, "
              f"heatmap_mode={model.heatmap_mode}, grid={model.griglia})")
        return model
    raise RuntimeError("Could not load Stage 1.")


def carica_stage2(path, num_landmarks, device):
    """
    Loads a Stage 2 checkpoint, rebuilding RefinementNet with the in_channels
    and max_offset it was trained with (the current MAX_OFFSET constant only
    applies to new training runs). Returns None if the file does not exist.
    """
    import os
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels", 240)
    max_offset  = ckpt.get("max_offset", MAX_OFFSET)
    stage2 = RefinementNet(num_landmarks=num_landmarks, in_channels=in_channels,
                           max_offset=max_offset).to(device)
    stage2.load_state_dict(ckpt["model_state_dict"])
    stage2.eval()
    nota = "" if abs(max_offset - MAX_OFFSET) < 1e-9 else f" (current MAX_OFFSET constant: {MAX_OFFSET})"
    print(f"Stage 2 loaded ({ckpt.get('arch', 'CNN')}, in_ch={in_channels}, "
          f"max_offset={max_offset}{nota}, epoch={ckpt.get('epoca', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.6f})")
    return stage2


# ─── Batching ─────────────────────────────────────────────────────────────────
def collate_base(batch, img_size=512):
    """
    Resizes images to img_size x img_size, normalizes landmark coordinates to
    [0, 1] and keeps the original image size, needed to convert errors and
    uncertainties back to mm on non-square images.
    Returns (images, points, masks, bboxes, pixel_sizes, dims) with
    dims [B, 2] = original (W, H) in pixels.
    """
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    imgs_r = [F.interpolate(img.unsqueeze(0), size=(img_size, img_size),
                            mode="bilinear", align_corners=False).squeeze(0)
              for img in immagini]
    punti_n, dims = [], []
    for img_orig, p in zip(immagini, punti):
        H, W = img_orig.shape[1], img_orig.shape[2]
        p_s = p.clone()
        p_s[:, 0] = p[:, 0] / W
        p_s[:, 1] = p[:, 1] / H
        punti_n.append(p_s)
        dims.append(torch.tensor([W, H], dtype=torch.float32))
    return (torch.stack(imgs_r), torch.stack(punti_n), torch.stack(list(masks)),
            torch.stack(list(bboxes)), torch.stack(list(pixel_sizes)), torch.stack(dims))


def errori_mm(pred, gt, dims, pixel_size):
    """Radial error in mm [B, N] between normalized coordinates, scaling x by
    the original width and y by the original height of each image."""
    d = (pred - gt) * dims.view(-1, 1, 2).to(pred.device)
    return torch.sqrt((d ** 2).sum(dim=2)) * pixel_size.view(-1, 1).to(pred.device)


# ─── Decoding and TTA ─────────────────────────────────────────────────────────
def ridimensiona_heatmap(pred_raw, heatmap_size=HEATMAP_TRAIN_SIZE):
    if pred_raw.shape[2] != heatmap_size or pred_raw.shape[3] != heatmap_size:
        pred_raw = F.interpolate(pred_raw, size=(heatmap_size, heatmap_size),
                                 mode="bilinear", align_corners=False)
    return pred_raw


def decodifica(pred_raw, stage1, heatmap_size=HEATMAP_TRAIN_SIZE, usa_voting=False):
    """Raw Stage 1 output -> normalized coordinates [B, N, 2], using the
    decoding settings of the loaded checkpoint (mode, grid, temperature)."""
    pred_raw = ridimensiona_heatmap(pred_raw, heatmap_size)
    if usa_voting:
        return regression_voting(pred_raw, pred_raw.shape[1] // 3,
                                 target_size=heatmap_size, griglia=stage1.griglia)
    return heatmap_a_coordinate(pred_raw[:, 0::3], temperatura=stage1.temperatura,
                                griglia=stage1.griglia)


def predici_con_tta(stage1, immagini, config, usa_voting=False):
    """
    Runs Stage 1 on a batch, with test-time augmentation if config["usa_tta"]:
    rotations by config["tta_angoli"] (each prediction rotated back), a
    horizontal flip and two brightness/contrast variants, then averaged.

    Returns a dict:
      coords    [B, N, 2] mean normalized coordinates
      x_out     fused feature map of the un-augmented pass (for Stage 2)
      pred_raw  raw output of the un-augmented pass (heatmap uncertainty)
      cov_tta   [B, N, 2, 2] spread of the TTA predictions (zeros without TTA)
    """
    hs = config.get("heatmap_size", HEATMAP_TRAIN_SIZE)
    tutte = []
    with torch.no_grad():
        pred_raw0, x_out0 = stage1(immagini)
        tutte.append(decodifica(pred_raw0, stage1, hs, usa_voting))
        if config.get("usa_tta", True):
            for angle in config.get("tta_angoli", [-5.0, -2.5, 0.0, 2.5, 5.0]):
                if abs(angle) < 0.01:
                    continue
                pred_rot, _ = stage1(TF.rotate(immagini, angle))
                # Predictions live in the rotated frame: rotate them back by -angle
                tutte.append(ruota_coordinate(decodifica(pred_rot, stage1, hs, usa_voting), -angle))
            pred_flip_raw, _ = stage1(TF.hflip(immagini))
            pred_flip = decodifica(pred_flip_raw, stage1, hs, usa_voting).clone()
            pred_flip[..., 0] = 1.0 - pred_flip[..., 0]
            tutte.append(pred_flip)
            for _ in range(2):
                imgs_var = TF.adjust_brightness(immagini, random.uniform(0.88, 1.12))
                imgs_var = TF.adjust_contrast(imgs_var, random.uniform(0.88, 1.12))
                pred_var, _ = stage1(imgs_var)
                tutte.append(decodifica(pred_var, stage1, hs, usa_voting))
    tutte = torch.stack(tutte)
    return {
        "coords":   tutte.mean(dim=0),
        "x_out":    x_out0,
        "pred_raw": ridimensiona_heatmap(pred_raw0, hs),
        "cov_tta":  covarianza_tta(tutte),
    }


# ─── Stage 2 ──────────────────────────────────────────────────────────────────
def estrai_roi(feat_map, cx_norm, cy_norm, batch_idx):
    """[K] centers (normalized) and batch indices -> [K, C, S, S] ROI crops."""
    _, _, H_feat, W_feat = feat_map.shape
    cx = cx_norm * W_feat
    cy = cy_norm * H_feat
    rois = torch.stack([
        batch_idx.float(),
        (cx - ROI_HALF_SIZE).clamp(min=0.0),
        (cy - ROI_HALF_SIZE).clamp(min=0.0),
        (cx + ROI_HALF_SIZE).clamp(max=float(W_feat)),
        (cy + ROI_HALF_SIZE).clamp(max=float(H_feat)),
    ], dim=1)
    return roi_align(feat_map, rois, output_size=ROI_OUTPUT_SIZE, spatial_scale=1.0, aligned=True)


def applica_stage2(stage2, x_out, coords_norm, landmark_da_correggere=None, fattore_guardia=1.2):
    """
    Applies Stage 2 to the landmarks in landmark_da_correggere (default
    REFINE_LANDMARK_MASK) of a whole batch in one forward pass. A correction is
    kept only if |dx| and |dy| stay within fattore_guardia * stage2.max_offset.
    """
    landmark = sorted(REFINE_LANDMARK_MASK if landmark_da_correggere is None else landmark_da_correggere)
    B, N, _ = coords_norm.shape
    landmark = [i for i in landmark if i < N]
    out = coords_norm.clone()
    if not landmark:
        return out
    dev = coords_norm.device
    b_idx = torch.arange(B, device=dev).repeat_interleave(len(landmark))
    l_idx = torch.tensor(landmark, device=dev).repeat(B)
    c = coords_norm[b_idx, l_idx]
    with torch.no_grad():
        crops  = estrai_roi(x_out, c[:, 0], c[:, 1], b_idx)
        offset = stage2(crops, l_idx)
    guardia = fattore_guardia * stage2.max_offset
    ok = (offset.abs() <= guardia).all(dim=1)
    out[b_idx[ok], l_idx[ok]] = c[ok] + offset[ok]
    return out
