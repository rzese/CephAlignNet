import torch, cv2, numpy as np, sys, os, math, random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.ops import roi_align

from src.model_hrnet import HRNetLandmark
from src.model_refine import (RefinementNet, MAX_OFFSET, ROI_OUTPUT_SIZE,
                               ROI_HALF_SIZE, REFINE_LANDMARK_MASK)
from src.heatmap_utils import heatmap_a_coordinate, regression_voting, HEATMAP_TRAIN_SIZE

USE_VOTING = False   # False = soft-argmax (stable), True = regression voting

CONFIG = {
    "num_landmarks":  26,   # 19 original + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
    "img_size":       512,
    "heatmap_size":   HEATMAP_TRAIN_SIZE,
    "stage1_path":    "modello_heatmap.pth",
    "stage2_path":    "modello_refine.pth",
    "usa_stage2":     True,
    "usa_tta":        True,
    "tta_angoli":     [-5.0, -2.5, 0.0, 2.5, 5.0],
    "hrnet_channels": [192, 384, 768, 1536],
}

NOMI_LANDMARK = [
    "A","ANS","B","Me","N","Or","Pog","PNS","Pn","R",
    "S","Ar","Co","Gn","Go","Po","LPM","LIT","LMT",
    "UIT","UIA","LIA",   
    "Ls","Li","Pog`","Sn",  
]


def carica_stage1(config, device):
    """
    Loads the Stage 1 checkpoint, trying several possible HRNet channel
    configurations until one matches the saved weights (handles checkpoints
    saved with a different backbone/channel setup).
    """
    ckpt = torch.load(config["stage1_path"], map_location=device, weights_only=False)
    for canali in [config["hrnet_channels"], [192, 384, 768, 1536], [128, 256, 512, 1024],
                   [64, 128, 256, 512, 1024], [64, 128, 256, 512], [32, 64, 128, 256]]:
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
            if "size mismatch" in str(e) or "Unexpected key" in str(e):
                continue
            raise
    raise RuntimeError("Could not load Stage 1.")


def carica_modelli(config, device):
    """Loads both Stage 1 (heatmap) and, if available and enabled, Stage 2
    (refinement) models and returns them ready for inference (eval mode)."""
    stage1 = carica_stage1(config, device)
    stage2 = None
    if config["usa_stage2"] and os.path.exists(config["stage2_path"]):
        ckpt2 = torch.load(config["stage2_path"], map_location=device, weights_only=False)
        in_channels = ckpt2.get("in_channels", 192)
        stage2 = RefinementNet(num_landmarks=config["num_landmarks"],
                               in_channels=in_channels).to(device)
        stage2.load_state_dict(ckpt2["model_state_dict"])
        stage2.eval()
        print(f"Stage 2 loaded — in_ch={in_channels}, "
              f"epoch={ckpt2.get('epoca','?')}, val_loss={ckpt2.get('val_loss',0):.6f})")
    else:
        print("Stage 2 not available.")
    return stage1, stage2


def decodifica(pred_raw, config, device):
    """Decodes raw Stage 1 output into normalized (x, y) coordinates, using
    either regression voting or soft-argmax depending on USE_VOTING."""
    ts = config["heatmap_size"]
    if pred_raw.shape[2] != ts or pred_raw.shape[3] != ts:
        pred_raw = F.interpolate(pred_raw, size=(ts, ts),
                                 mode='bilinear', align_corners=False)
    n = config["num_landmarks"]
    if USE_VOTING:
        return regression_voting(pred_raw, n, target_size=ts)[0]
    else:
        return heatmap_a_coordinate(pred_raw[:, 0::3])[0]


def _de_ruota(coords, angle_deg):
    """Rotates predicted coordinates back by -angle_deg around the image
    center, undoing a TTA rotation applied to the input image."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = coords.clone()
    cx = coords[:, 0] - 0.5; cy = coords[:, 1] - 0.5
    out[:, 0] = cos_t * cx - sin_t * cy + 0.5
    out[:, 1] = sin_t * cx + cos_t * cy + 0.5
    return out


def predici_tta(stage1, img_tensor, config, device):
    """
    Test-time augmentation: averages Stage 1 predictions across several
    rotated versions of the input, a horizontal flip, and two random
    brightness/contrast variants, to obtain more robust coordinates.
    Also returns the fused feature map (x_out) from the un-augmented
    (angle=0) forward pass, for use by Stage 2.
    """
    tutte = []; x_out_orig = None
    angoli = config.get("tta_angoli", [-5.0, -2.5, 0.0, 2.5, 5.0])
    with torch.no_grad():
        for angle in angoli:
            if abs(angle) < 0.01:
                pred_raw, x_out_orig = stage1(img_tensor)
                tutte.append(decodifica(pred_raw, config, device))
            else:
                img_rot = TF.rotate(img_tensor, angle)
                pred_raw_rot, _ = stage1(img_rot)
                tutte.append(_de_ruota(decodifica(pred_raw_rot, config, device), -angle))
        img_flip = TF.hflip(img_tensor)
        pred_raw_flip, _ = stage1(img_flip)
        pred_flip = decodifica(pred_raw_flip, config, device).clone()
        pred_flip[:, 0] = 1.0 - pred_flip[:, 0]
        tutte.append(pred_flip)
        for _ in range(2):
            img_var = TF.adjust_brightness(img_tensor, random.uniform(0.90, 1.10))
            img_var = TF.adjust_contrast(img_var, random.uniform(0.90, 1.10))
            pred_raw_var, _ = stage1(img_var)
            tutte.append(decodifica(pred_raw_var, config, device))
    return torch.stack(tutte).mean(dim=0), x_out_orig


def applica_stage2(stage2, x_out, coords_norm, config, device):
    """
    Applies the Stage 2 refinement network to the landmarks in
    REFINE_LANDMARK_MASK: crops a small ROI from x_out around each Stage 1
    prediction, regresses an (x, y) offset, and applies it only if it stays
    within a safety guard (1.2x MAX_OFFSET) to avoid large, unstable jumps.
    """
    _, _, H_feat, W_feat = x_out.shape
    out = coords_norm.clone(); guardia = MAX_OFFSET * 1.2
    with torch.no_grad():
        for i in range(coords_norm.shape[0]):
            if i not in REFINE_LANDMARK_MASK:
                continue
            cx = coords_norm[i, 0].item(); cy = coords_norm[i, 1].item()
            cx_f = cx * W_feat; cy_f = cy * H_feat
            x1 = max(0.0, cx_f - ROI_HALF_SIZE); y1 = max(0.0, cy_f - ROI_HALF_SIZE)
            x2 = min(float(W_feat), cx_f + ROI_HALF_SIZE)
            y2 = min(float(H_feat), cy_f + ROI_HALF_SIZE)
            rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)
            crop = roi_align(x_out, rois, output_size=ROI_OUTPUT_SIZE,
                             spatial_scale=1.0, aligned=True)
            idx    = torch.tensor([i], dtype=torch.long, device=device)
            offset = stage2(crop, idx)[0]
            dx, dy = offset[0].item(), offset[1].item()
            if abs(dx) <= guardia and abs(dy) <= guardia:
                out[i, 0] = cx + dx; out[i, 1] = cy + dy
    return out


def predici(stage1, stage2, img_path, device, config):
    """
    Full single-image inference pipeline: loads the image, resizes it to
    the training resolution, runs Stage 1 (with or without TTA), optionally
    refines the result with Stage 2, and rescales the coordinates back to
    the original image size.
    Returns (original_grayscale_image, landmarks_in_pixels).
    """
    img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h_orig, w_orig = img_orig.shape
    img_resized = cv2.resize(img_orig, (config["img_size"], config["img_size"]))
    img_tensor  = (torch.from_numpy(img_resized).float()
                   .unsqueeze(0).unsqueeze(0) / 255.0).to(device)
    if config["usa_tta"]:
        coords_norm, x_out = predici_tta(stage1, img_tensor, config, device)
    else:
        with torch.no_grad():
            pred_raw, x_out = stage1(img_tensor)
        coords_norm = decodifica(pred_raw, config, device)
    if stage2 is not None and x_out is not None:
        coords_norm = applica_stage2(stage2, x_out, coords_norm, config, device)
    coords = coords_norm.cpu().numpy()
    coords[:, 0] *= w_orig; coords[:, 1] *= h_orig
    return img_orig, coords


def disegna_landmark(img_gray, landmarks):
    """Draws the predicted landmarks (with their names) over the image and
    displays it in an OpenCV window, scaled down if larger than 1000px."""
    img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w    = img_vis.shape[:2]
    raggio  = max(6, int(min(h, w) / 150))
    font_scale = max(0.4, min(h, w) / 1500)
    spessore   = max(1, int(min(h, w) / 600))
    for i, pt in enumerate(landmarks):
        x, y = int(pt[0]), int(pt[1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        cv2.circle(img_vis, (x, y), raggio, (0, 220, 255), -1)
        cv2.circle(img_vis, (x, y), raggio, (0, 0, 0), 1)
        nome = NOMI_LANDMARK[i] if i < len(NOMI_LANDMARK) else str(i + 1)
        cv2.putText(img_vis, nome, (x + raggio + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255),
                    spessore, cv2.LINE_AA)
    scala     = min(1.0, 1000 / max(h, w))
    img_show  = cv2.resize(img_vis, (int(w * scala), int(h * scala)))
    cv2.imshow("Landmark prediction", img_show)
    print("Press any key to close..."); cv2.waitKey(0); cv2.destroyAllWindows()


def main():
    """CLI entry point: predicts landmarks for a single image (path given
    as the first argument, or a default sample) and shows/prints the result."""
    img_path = "data/Aariz/test/Cephalograms/cks2ip8fp29yl0yuf6ry9266i.png"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modo   = "regression_voting" if USE_VOTING else "soft-argmax"
    print(f"Device: {device} | Backbone: HRNet-W48 | Mode: {modo}")
    print(f"Landmarks: {CONFIG['num_landmarks']} (19 original + UIT + UIA + LIA)")
    stage1, stage2 = carica_modelli(CONFIG, device)
    img_orig, landmarks = predici(stage1, stage2, img_path, device, CONFIG)
    disegna_landmark(img_orig, landmarks)
    print("\nPredicted coordinates (pixels):")
    for i, pt in enumerate(landmarks):
        nome = NOMI_LANDMARK[i] if i < len(NOMI_LANDMARK) else str(i + 1)
        print(f"  {nome:5s}: x={pt[0]:.1f}, y={pt[1]:.1f}")


if __name__ == "__main__":
    main()