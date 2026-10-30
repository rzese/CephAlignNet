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

from src.datasets import registra_impronta_modello
from src.model_hrnet import HRNetLandmark
from src.model_refine import (RefinementNet, ImageRefinementNet, crea_stage2_immagine, MAX_OFFSET, ROI_OUTPUT_SIZE,
                              ROI_HALF_SIZE, REFINE_LANDMARK_MASK)
from src.heatmap_utils import (heatmap_a_coordinate, regression_voting, ruota_coordinate,
                               HEATMAP_TRAIN_SIZE)
from src.incertezza import TEMPERATURA_DECODIFICA, covarianza_tta, cov_in_mm, sigma_radiale

# Which points Stage 2 is applied to (see seleziona_punti)
MODI_SELEZIONE = ("tutti", "maschera", "incertezza", "fusione")

# Default parameters of the Stage 1 / Stage 2 fusion (see peso_fusione), chosen
# with strategie_stage2.py on the "probabilistic" Stage 1 + image-based Stage 2
# (2-fold CV over the validation images, objective = accuracy within 2-4 mm)
FUSIONE_DEFAULT = {"c": 0.32, "p": 2.0, "floor_mm": 0.0}


def peso_fusione(sigma1_mm, sigma2_mm, c=0.32, p=2.0, floor_mm=0.0):
    """
    Weight of Stage 2 in the fused position  S1 + w * (S2 - S1):
        w = s1^p / (s1^p + (c * s2)^p),   s1 = sqrt(sigma1^2 + floor^2)
    s1 = Stage 1 uncertainty (sigma_tta), s2 = Stage 2 uncertainty (spread of
    its crop heatmap), both in mm. With p = 2 it is the classic
    inverse-variance average of two measurements; c puts the two
    uncertainties on the same scale. w -> 1 where Stage 1 is unsure and
    Stage 2 sure, w -> 0 in the opposite case.
    """
    s1 = torch.sqrt(sigma1_mm ** 2 + floor_mm ** 2) + 1e-6
    s2 = sigma2_mm + 1e-6
    return s1 ** p / (s1 ** p + (c * s2) ** p)

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
              f"heatmap_mode={model.heatmap_mode}, grid={model.griglia}, "
              f"validation set {ckpt.get('split_impronta') or 'unknown (old checkpoint)'})")
        registra_impronta_modello("Stage 1", ckpt.get("split_impronta"))
        return model
    raise RuntimeError("Could not load Stage 1.")


def carica_stage2(path, num_landmarks, device):
    """
    Loads a Stage 2 checkpoint of either kind (returns None if the file does
    not exist):
      - tipo_input "immagine": crops of the original radiograph (train_refine.py),
        ImageRefinementNet (arch_stage2 "base", also for checkpoints without the
        key) or ImageRefinementUNet (arch_stage2 "unet", pretrained encoder);
      - tipo_input "feature" (old checkpoints): RefinementNet on crops of the
        Stage 1 feature map, rebuilt with the in_channels and max_offset it
        was trained with.
    The model carries .tipo_input, used by applica_stage2.
    """
    import os
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("tipo_input") == "immagine":
        arch = ckpt.get("arch_stage2", "base")
        # pretrained=False: the weights come from the checkpoint (compute nodes are offline)
        stage2 = crea_stage2_immagine(arch, num_landmarks=num_landmarks, finestra_mm=ckpt["finestra_mm"],
                                      crop_px=ckpt["crop_px"], max_offset_mm=ckpt["max_offset_mm"],
                                      encoder=ckpt.get("encoder", "resnet34"), pretrained=False).to(device)
        stage2.load_state_dict(ckpt["model_state_dict"])
        stage2.eval()
        print(f"Stage 2 loaded ({type(stage2).__name__} on the original radiograph, window "
              f"{stage2.finestra_mm} mm / {stage2.crop_px} px, max offset {stage2.max_offset_mm} mm, "
              f"epoch={ckpt.get('epoca', '?')}, val_loss={ckpt.get('val_loss', 0):.4f}, "
              f"validation set {ckpt.get('split_impronta') or 'unknown (old checkpoint)'})")
        registra_impronta_modello("Stage 2", ckpt.get("split_impronta"))
        return stage2

    in_channels = ckpt.get("in_channels", 240)
    max_offset  = ckpt.get("max_offset", MAX_OFFSET)
    stage2 = RefinementNet(num_landmarks=num_landmarks, in_channels=in_channels,
                           max_offset=max_offset).to(device)
    stage2.load_state_dict(ckpt["model_state_dict"])
    stage2.eval()
    nota = "" if abs(max_offset - MAX_OFFSET) < 1e-9 else f" (current MAX_OFFSET constant: {MAX_OFFSET})"
    print(f"Stage 2 loaded ({ckpt.get('arch', 'CNN')} on Stage 1 features, in_ch={in_channels}, "
          f"max_offset={max_offset}{nota}, epoch={ckpt.get('epoca', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.6f})")
    return stage2


# ─── Batching ─────────────────────────────────────────────────────────────────
def collate_base(batch, img_size=512, con_originali=False):
    """
    Resizes images to img_size x img_size, normalizes landmark coordinates to
    [0, 1] and keeps the original image size, needed to convert errors and
    uncertainties back to mm on non-square images.
    Returns (images, points, masks, bboxes, pixel_sizes, dims) with
    dims [B, 2] = original (W, H) in pixels; with con_originali=True also,
    as 7th element, the list of original-resolution images [1, H, W]
    (needed by the image-based Stage 2).
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
    out = (torch.stack(imgs_r), torch.stack(punti_n), torch.stack(list(masks)),
           torch.stack(list(bboxes)), torch.stack(list(pixel_sizes)), torch.stack(dims))
    return out + (list(immagini),) if con_originali else out


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


def ritaglia_originali(originali, centri_norm, b_idx, pixel_size, finestra_mm, crop_px):
    """
    Crops of the original radiographs: for every k, a finestra_mm x finestra_mm
    square centered on centri_norm[k] (normalized coordinates) of image
    originali[b_idx[k]], resampled to crop_px x crop_px with roi_align (zero
    outside the image) and standardized (zero mean, unit std) per crop.
    originali: list of [1, H, W] tensors; pixel_size [B] in mm/px.
    Returns [K, 1, crop_px, crop_px].
    """
    dev = centri_norm.device
    crops = torch.zeros(len(b_idx), 1, crop_px, crop_px, device=dev)
    for b in torch.unique(b_idx).tolist():
        sel = (b_idx == b).nonzero(as_tuple=True)[0]
        img = originali[b].to(dev).float().unsqueeze(0)               # [1, 1, H, W]
        H, W = img.shape[-2:]
        half = finestra_mm / 2.0 / float(pixel_size[b])                # px
        cx = centri_norm[sel, 0] * W
        cy = centri_norm[sel, 1] * H
        boxes = torch.stack([torch.zeros_like(cx), cx - half, cy - half, cx + half, cy + half], dim=1)
        crops[sel] = roi_align(img, boxes, output_size=crop_px, spatial_scale=1.0,
                               sampling_ratio=2, aligned=True)
    m = crops.mean(dim=(2, 3), keepdim=True)
    sd = crops.std(dim=(2, 3), keepdim=True)
    return (crops - m) / (sd + 1e-6)


def seleziona_punti(modo, B, N, device, cov_tta=None, dims=None, pixel_size=None, soglia_mm=None):
    """
    Which points Stage 2 is applied to, as a bool tensor [B, N]:
      "tutti"       every landmark
      "fusione"     every landmark (the weighting is done in applica_stage2)
      "maschera"    the landmarks in REFINE_LANDMARK_MASK (src/landmark_stats.py)
      "incertezza"  the points whose TTA spread (sigma_tta, mm) exceeds soglia_mm
                    — needs cov_tta (i.e. TTA on), dims and pixel_size
    """
    if modo in ("tutti", "fusione"):
        return torch.ones(B, N, dtype=torch.bool, device=device)
    if modo == "maschera":
        sel = torch.zeros(B, N, dtype=torch.bool, device=device)
        sel[:, [i for i in REFINE_LANDMARK_MASK if i < N]] = True
        return sel
    if modo == "incertezza":
        if cov_tta is None or soglia_mm is None:
            raise ValueError("selection 'incertezza' needs the TTA covariance and soglia_mm")
        sigma = sigma_radiale(cov_in_mm(cov_tta, dims, pixel_size))[:, :N]
        return sigma > soglia_mm
    raise ValueError(f"unknown Stage 2 selection '{modo}', expected one of {MODI_SELEZIONE}")


def applica_stage2(stage2, coords_norm, x_out=None, originali=None, dims=None, pixel_size=None,
                   landmark_da_correggere=None, seleziona=None, fattore_guardia=1.2,
                   fusione=None, sigma1_mm=None):
    """
    Applies Stage 2 to a batch of normalized coordinates [B, N, 2].

    Which points: those of landmark_da_correggere (default: all landmarks for
    the image-based Stage 2, REFINE_LANDMARK_MASK for the old one), further
    restricted by the bool mask seleziona [B, N] if given (see seleziona_punti).

    - image-based Stage 2 (stage2.tipo_input == "immagine"): crops of the
      original radiographs (originali, dims, pixel_size required); a correction
      is kept if |dx| and |dy| <= fattore_guardia * max_offset_mm.
    - old feature-based Stage 2: crops of the Stage 1 feature map x_out; a
      correction is kept if |dx| and |dy| <= fattore_guardia * max_offset.

    fusione (image-based Stage 2 only): dict with c, p, floor_mm (see
    peso_fusione) and sigma1_mm [B, N] = Stage 1 uncertainty (sigma_tta). The
    result is then the weighted average S1 + w * (S2 - S1) instead of S2.
    """
    B, N, _ = coords_norm.shape
    dev = coords_norm.device
    immagine = getattr(stage2, "tipo_input", "feature") == "immagine"
    if landmark_da_correggere is None:
        landmark_da_correggere = range(N) if immagine else REFINE_LANDMARK_MASK
    sel = torch.zeros(B, N, dtype=torch.bool, device=dev)
    sel[:, [i for i in landmark_da_correggere if i < N]] = True
    if seleziona is not None:
        sel &= seleziona.to(dev)
    out = coords_norm.clone()
    if not sel.any():
        return out
    b_idx, l_idx = sel.nonzero(as_tuple=True)
    c = coords_norm[b_idx, l_idx]

    with torch.no_grad():
        if immagine:
            crops = ritaglia_originali(originali, c, b_idx, pixel_size, stage2.finestra_mm, stage2.crop_px)
            offset_mm, logits = stage2(crops, l_idx)
            ok = (offset_mm.abs() <= fattore_guardia * stage2.max_offset_mm).all(dim=1)
            if fusione is not None:
                if sigma1_mm is None:
                    raise ValueError("the fusion needs sigma1_mm (Stage 1 sigma_tta, i.e. TTA on)")
                w = peso_fusione(sigma1_mm.to(dev)[b_idx, l_idx], varianza_crop(logits, stage2.finestra_mm).sqrt(),
                                 **fusione)
                offset_mm = offset_mm * w.unsqueeze(1)
            mm_per_unita = dims.to(dev)[b_idx] * pixel_size.to(dev)[b_idx].view(-1, 1)   # mm per normalized unit
            offset = offset_mm / mm_per_unita
        else:
            offset = stage2(estrai_roi(x_out, c[:, 0], c[:, 1], b_idx), l_idx)
            ok = (offset.abs() <= fattore_guardia * stage2.max_offset).all(dim=1)
    out[b_idx[ok], l_idx[ok]] = c[ok] + offset[ok]
    return out


# ─── Stage 2 (image-based): uncertainty and test-time augmentation ────────────
def varianza_crop(logits, finestra_mm):
    """Radial variance (mm^2, trace of the covariance) of the position
    distribution predicted by ImageRefinementNet inside each crop."""
    K, s, _ = logits.shape
    p = torch.softmax(logits.reshape(K, -1), dim=1).view(K, s, s)
    g = ((torch.arange(s, device=logits.device, dtype=torch.float32) + 0.5) / s - 0.5) * finestra_mm
    px, py = p.sum(dim=1), p.sum(dim=2)              # marginals along x and y
    vx = (px * g ** 2).sum(1) - (px * g).sum(1) ** 2
    vy = (py * g ** 2).sum(1) - (py * g).sum(1) ** 2
    return (vx + vy).clamp(min=0)


def stage2_predizioni(stage2, centri_norm, b_idx, l_idx, originali, pixel_size,
                      tta=False, spostamento_mm=0.4, angoli=(5.0, -5.0)):
    """
    Detailed predictions of the image-based Stage 2 for K points (no guard).

    centri_norm [K,2] normalized starting positions, b_idx/l_idx [K] image and
    landmark of each point, originali list of [1,H,W], pixel_size [B].
    With tta=True Stage 2 also runs on crops whose center is moved by
    +-spostamento_mm along x and y, and on crops rotated by each angle in
    angoli; every answer is brought back to the original crop frame and
    averaged.

    Returns a dict of [K,...] tensors, offsets in mm from the starting point:
      offset       single (un-augmented) prediction            [K,2]
      offset_tta   mean over the TTA variants (= offset without TTA)
      var_hm       mean radial variance of the crop distribution (mm^2) [K]
      var_tta      radial variance of the TTA answers (mm^2, 0 without TTA) [K]
    """
    f = stage2.finestra_mm
    dev = centri_norm.device
    dims = torch.tensor([[im.shape[2], im.shape[1]] for im in originali], dtype=torch.float32, device=dev)
    mm_per_unita = dims[b_idx] * pixel_size.to(dev)[b_idx].view(-1, 1)
    varianti = [(0.0, 0.0, 0.0)]
    if tta:
        d = spostamento_mm
        varianti += [(d, 0.0, 0.0), (-d, 0.0, 0.0), (0.0, d, 0.0), (0.0, -d, 0.0)]
        varianti += [(0.0, 0.0, a) for a in angoli]
    offs, varianze = [], []
    with torch.no_grad():
        for sx, sy, a in varianti:
            shift = torch.tensor([sx, sy], device=dev)
            crops = ritaglia_originali(originali, centri_norm + shift / mm_per_unita, b_idx, pixel_size,
                                       f, stage2.crop_px)
            if a:
                crops = TF.rotate(crops, a, interpolation=TF.InterpolationMode.BILINEAR)
            off, logits = stage2(crops, l_idx)
            if a:   # answer is in the rotated crop: rotate it back
                off = (ruota_coordinate(off / f + 0.5, -a) - 0.5) * f
            offs.append(off + shift)
            varianze.append(varianza_crop(logits, f))
    offs = torch.stack(offs)                                      # [V, K, 2]
    var_tta = ((offs - offs.mean(0)) ** 2).sum(-1).mean(0) if len(varianti) > 1 \
        else torch.zeros(offs.shape[1], device=dev)
    return {"offset": offs[0], "offset_tta": offs.mean(0),
            "var_hm": torch.stack(varianze).mean(0), "var_tta": var_tta}
