import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

SIGMA_PER_LANDMARK = {
    # Anatomically ambiguous landmarks -> wider sigma = more robust heatmap
    5:  5.0,   # Or  
    9:  6.5,   # S   
    10: 5.5,   # Ar
    11: 5.5,   # Co
    14: 6.0,   # Po  
    15: 6.0,   # LPM 
    18: 6.0,   # UPM
    19: 5.0,   # UIT 
    20: 5.5,   # UIA  
    21: 5.5,   # LIA 
}

PESI_LANDMARK = {
    # Difficult landmarks -> higher penalty in the loss
    5:  2.0,   # Or
    9:  3.5,   # S
    10: 2.0,   # Ar
    11: 2.0,   # Co
    12: 2.5,   # Gn
    13: 2.0,   # Go
    14: 3.0,   # Po
    15: 3.5,   # LPM
    16: 2.0,   # LIT
    17: 2.0,   # LMT
    18: 3.0,   # UPM
    19: 2.0,   # UIT 
    20: 2.5,   # UIA 
    21: 2.5,   # LIA 
}

# Resolution the offsets are trained at (must match heatmap_size in CONFIG)
HEATMAP_TRAIN_SIZE = 256


def regression_voting(output, n_landmarks, R=40, target_size=HEATMAP_TRAIN_SIZE):
    """
    output: [B, N*3, H, W]  — heatmap + 2 offset channels per landmark
    n_landmarks: int
    R: voting radius in pixels at target_size resolution (default 40 @ 256px)
    target_size: resolution the model was trained at.

    Returns normalized coordinates [B, N, 2], obtained by having the top
    heatmap activations "vote" for the landmark location using their
    predicted local offsets (DSNT/offset-voting style decoding).
    """
    if output.shape[2] != target_size or output.shape[3] != target_size:
        output = F.interpolate(output, size=(target_size, target_size),
                               mode='bilinear', align_corners=False)

    B, _, H, W = output.shape
    heatmaps = output[:, 0::3]
    offset_x = output[:, 1::3]
    offset_y = output[:, 2::3]

    coords = []
    for b in range(B):
        pts = []
        for i in range(n_landmarks):
            hm = heatmaps[b, i]
            ox = offset_x[b, i]
            oy = offset_y[b, i]

            flat = hm.view(-1)
            k    = min(int(math.pi * R ** 2), flat.numel())
            topk = torch.topk(flat, k).indices
            rows = topk // W
            cols = topk % W

            vote_x = cols.float() + ox.view(-1)[topk] * R
            vote_y = rows.float() + oy.view(-1)[topk] * R

            px = vote_x.mean() / W
            py = vote_y.mean() / H
            pts.append(torch.stack([px.clamp(0, 1), py.clamp(0, 1)]))
        coords.append(torch.stack(pts))
    return torch.stack(coords)   # [B, N, 2]


def combined_loss(pred, heatmap_gt, pts_gt, mask, R=40, alpha=2/3):
    """Combines the AWing heatmap loss with an L1 loss on the local offset
    maps, restricted to pixels within radius R of the ground-truth point."""

    B, _, H, W = pred.shape
    N = heatmap_gt.shape[1]

    pred_hm = pred[:, 0::3]
    pred_ox = pred[:, 1::3]
    pred_oy = pred[:, 2::3]

    loss_hm = awing_loss(pred_hm, heatmap_gt, mask)

    grid_x = torch.arange(W, device=pred.device).float().view(1, 1, 1, W).expand(B, N, H, W)
    grid_y = torch.arange(H, device=pred.device).float().view(1, 1, H, 1).expand(B, N, H, W)
    gt_x   = (pts_gt[:, :, 0] * W).view(B, N, 1, 1).expand_as(grid_x)
    gt_y   = (pts_gt[:, :, 1] * H).view(B, N, 1, 1).expand_as(grid_y)

    gt_ox = (gt_x - grid_x) / R
    gt_oy = (gt_y - grid_y) / R

    dist2  = (grid_x - gt_x) ** 2 + (grid_y - gt_y) ** 2
    within = (dist2 <= R ** 2).float()

    mask_4d  = mask.view(B, N, 1, 1)
    n_valid  = (within * mask_4d).sum().clamp(min=1)
    loss_off  = ((pred_ox - gt_ox).abs() * within * mask_4d).sum() / n_valid
    loss_off += ((pred_oy - gt_oy).abs() * within * mask_4d).sum() / n_valid

    return alpha * loss_hm + (1 - alpha) * loss_off


def crea_heatmap_gaussiana(h, w, cx, cy, sigma=4.0):
    """Renders a 2D Gaussian heatmap of size (h, w) centered at (cx, cy)."""
    x = np.arange(0, w, 1, np.float32)
    y = np.arange(0, h, 1, np.float32)[:, np.newaxis]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def coordinate_a_heatmap(pts, mask, img_size=512, heatmap_size=256, sigma=4.0):
    """Converts normalized ground-truth (x, y) coordinates into per-landmark
    Gaussian heatmaps used as the Stage 1 training target."""

    N = pts.shape[0]
    heatmaps = np.zeros((N, heatmap_size, heatmap_size), dtype=np.float32)
    for i in range(N):
        if mask[i].item() == 1.0:
            s  = SIGMA_PER_LANDMARK.get(i, sigma)
            cx = pts[i, 0].item() * heatmap_size
            cy = pts[i, 1].item() * heatmap_size
            if 0 <= cx < heatmap_size and 0 <= cy < heatmap_size:
                heatmaps[i] = crea_heatmap_gaussiana(heatmap_size, heatmap_size, cx, cy, s)
    return torch.from_numpy(heatmaps)


def heatmap_a_coordinate(heatmaps, temperatura=100.0):
    """Soft-argmax decoding: converts predicted heatmaps into normalized
    (x, y) coordinates via a temperature-scaled softmax expectation.
    This is the default, stable decoding used at inference (USE_VOTING=False)."""
    B, N, H, W = heatmaps.shape
    flat    = heatmaps.view(B, N, -1) * temperatura
    weights = torch.softmax(flat, dim=2).view(B, N, H, W)
    grid_x  = torch.linspace(0, 1, W, device=heatmaps.device).view(1, 1, 1, W)
    grid_y  = torch.linspace(0, 1, H, device=heatmaps.device).view(1, 1, H, 1)
    x = (weights * grid_x).sum(dim=[2, 3])
    y = (weights * grid_y).sum(dim=[2, 3])
    return torch.stack([x, y], dim=2)


def heatmap_a_coordinate_legacy(heatmaps):
    """Older/alternative soft-argmax decoding (unweighted softmax, marginal
    sums along each axis). Kept for reference / backward compatibility."""
    B, N, H, W = heatmaps.shape
    heatmaps_soft = torch.softmax(heatmaps.view(B, N, -1), dim=2).view(B, N, H, W)
    grid_x = torch.linspace(0, 1, W, device=heatmaps.device)
    grid_y = torch.linspace(0, 1, H, device=heatmaps.device)
    x = (heatmaps_soft.sum(dim=2) * grid_x).sum(dim=2)
    y = (heatmaps_soft.sum(dim=3) * grid_y).sum(dim=2)
    return torch.stack([x, y], dim=2)


def awing_loss(pred, target, mask, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
    """Adaptive Wing loss (Wang et al., 2019) for heatmap regression, with
    per-landmark weighting (PESI_LANDMARK) so that historically difficult
    landmarks contribute more to the gradient."""

    B, N, H, W = pred.shape

    A = (omega * (1.0 / (1.0 + (theta / epsilon) ** (alpha - target)))
         * (alpha - target) * ((theta / epsilon) ** (alpha - target - 1))
         * (1.0 / epsilon))
    C = theta * A - omega * torch.log(1.0 + (theta / epsilon) ** (alpha - target))

    diff        = torch.abs(pred - target)
    loss_wing   = omega * torch.log(1.0 + (diff / epsilon) ** (alpha - target))
    loss_linear = A * diff - C
    loss_map    = torch.where(diff < theta, loss_wing, loss_linear)

    pesi = torch.ones(N, device=pred.device)
    for idx, peso in PESI_LANDMARK.items():
        if idx < N:
            pesi[idx] = peso

    mask_pesata = (mask * pesi.unsqueeze(0)).unsqueeze(2).unsqueeze(3)
    loss_pesata = loss_map * mask_pesata
    num_validi  = mask_pesata.sum() * H * W

    if num_validi == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return loss_pesata.sum() / num_validi


def heatmap_loss(pred, target, mask):
    return awing_loss(pred, target, mask)


def wing_loss_coord(pred, gt, mask, w=10, eps=2.0):
    """Auxiliary Wing loss computed directly on normalized coordinates
    (decoded via soft-argmax), complementing the heatmap-space loss."""
    diff = (pred - gt).abs()
    C    = w - w * math.log(1 + w / eps)
    loss = torch.where(diff < w,
                       w * torch.log(1 + diff / eps),
                       diff - C)
    m4 = mask.unsqueeze(-1)
    return (loss * m4).sum() / m4.sum().clamp(min=1)


def geometric_consistency_loss(pred_pts, mask):
    """
    Penalizes geometrically inconsistent predictions between related
    landmarks.

    Enforced anatomical relationships:
      - Gn (12) must be close to the midpoint of Me (3) and Pog (6)
      - S (9) must be to the left (smaller x) of N (4) on a right-facing
        lateral cephalogram
      - Me (3) must be lower (larger y) than B (2)
      - B (2) must be lower than A (0)
      - ANS (1) must be to the right (larger x) of PNS (7)
      - Co (11) must be higher (smaller y) than Go (13)
      - UIT (19) must be higher (smaller y) than UIA (20)  
      - LIT (16) must be lower (larger y) than LIA (21)   
    """
    B    = pred_pts.shape[0]
    loss = torch.tensor(0.0, device=pred_pts.device)
    n_terms = 0

    IDX = {"A": 0, "ANS": 1, "B": 2, "Me": 3, "N": 4, "Pog": 6, "PNS": 7, "S": 9,
           "Co": 11, "Gn": 12, "Go": 13,
           "LIT": 16, "UIT": 19, "UIA": 20, "LIA": 21}

    for b in range(B):
        pts = pred_pts[b]
        msk = mask[b]

        # 1. Gn ~ midpoint(Me, Pog)
        if msk[IDX["Gn"]] and msk[IDX["Me"]] and msk[IDX["Pog"]]:
            mid = (pts[IDX["Me"]] + pts[IDX["Pog"]]) / 2.0
            loss += ((pts[IDX["Gn"]] - mid) ** 2).sum()
            n_terms += 1

        # 2. Me lower than B (y_Me > y_B)
        if msk[IDX["Me"]] and msk[IDX["B"]]:
            viol = torch.clamp(pts[IDX["B"], 1] - pts[IDX["Me"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

        # 3. B lower than A
        if msk[IDX["B"]] and msk[IDX["A"]]:
            viol = torch.clamp(pts[IDX["A"], 1] - pts[IDX["B"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

        # 4. ANS to the right of PNS
        if msk[IDX["ANS"]] and msk[IDX["PNS"]]:
            viol = torch.clamp(pts[IDX["PNS"], 0] - pts[IDX["ANS"], 0], min=0)
            loss += viol ** 2
            n_terms += 1

        # 5. Co higher than Go
        if msk[IDX["Co"]] and msk[IDX["Go"]]:
            viol = torch.clamp(pts[IDX["Co"], 1] - pts[IDX["Go"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

        # 6. S to the left of N
        if msk[IDX["S"]] and msk[IDX["N"]]:
            viol = torch.clamp(pts[IDX["S"], 0] - pts[IDX["N"], 0], min=0)
            loss += viol ** 2
            n_terms += 1

        # 7. UIT higher than UIA (crown tip above root apex)
        if msk[IDX["UIT"]] and msk[IDX["UIA"]]:
            viol = torch.clamp(pts[IDX["UIT"], 1] - pts[IDX["UIA"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

        # 8. LIA lower than LIT (lower root apex below crown tip)
        if msk[IDX["LIA"]] and msk[IDX["LIT"]]:
            viol = torch.clamp(pts[IDX["LIT"], 1] - pts[IDX["LIA"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

    return loss / max(n_terms, 1)


def zoom_crop_augmenta(img, pts, mask, zoom_factor=2.2):
    """
    Multi-scale augmentation for hard landmarks: crops a region centered
    on the average position of the historically difficult landmarks
    (index list below) and resizes it back to the original resolution,
    giving the model extra local detail where it usually makes mistakes.
    Difficult landmarks by index: 5=Or, 9=S, 14=Po, 15=LPM, 18=UPM,
    19=UIT, 20=UIA, 21=LIA
    """
    HARD = [5, 9, 14, 15, 18, 19, 20, 21]   # added UIT, UIA, LIA
    C, H, W = img.shape

    valid_pts = []
    for i in HARD:
        if i < pts.shape[0] and mask[i] == 1.0:
            valid_pts.append((pts[i, 0].item(), pts[i, 1].item()))

    if not valid_pts:
        return img, pts, mask

    cx = sum(p[0] for p in valid_pts) / len(valid_pts)
    cy = sum(p[1] for p in valid_pts) / len(valid_pts)

    crop_w = 1.0 / zoom_factor
    crop_h = 1.0 / zoom_factor

    x1 = max(0.0, cx - crop_w / 2)
    y1 = max(0.0, cy - crop_h / 2)
    x2 = min(1.0, x1 + crop_w)
    y2 = min(1.0, y1 + crop_h)
    x1 = max(0.0, x2 - crop_w)
    y1 = max(0.0, y2 - crop_h)

    x1_px = int(x1 * W); y1_px = int(y1 * H)
    x2_px = int(x2 * W); y2_px = int(y2 * H)
    x2_px = max(x2_px, x1_px + 1); y2_px = max(y2_px, y1_px + 1)

    img_crop = img[:, y1_px:y2_px, x1_px:x2_px]
    img_out  = F.interpolate(img_crop.unsqueeze(0), size=(H, W),
                              mode='bilinear', align_corners=False).squeeze(0)

    pts_out  = pts.clone()
    mask_out = mask.clone()
    for i in range(pts.shape[0]):
        if mask[i] == 1.0:
            nx = (pts[i, 0].item() - x1) / (x2 - x1)
            ny = (pts[i, 1].item() - y1) / (y2 - y1)
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                pts_out[i, 0] = nx
                pts_out[i, 1] = ny
            else:
                mask_out[i]   = 0.0
                pts_out[i, 0] = 0.0
                pts_out[i, 1] = 0.0

    return img_out, pts_out, mask_out


def augmenta(img, pts, mask):
    """Main training-time augmentation pipeline: random horizontal flip,
    random rotation, random translation, random scale (crop/pad), photometric
    jitter (brightness/contrast/gamma), Gaussian noise, and occasional blur.
    Landmarks that fall outside the image after a geometric transform are
    masked out (mask=0) rather than clipped, to avoid feeding wrong labels."""
    C, H, W = img.shape

    if random.random() > 0.5:
        img = TF.hflip(img)
        for i in range(pts.shape[0]):
            if mask[i] == 1.0:
                pts[i, 0] = 1.0 - pts[i, 0]

    angle = random.uniform(-20, 20)
    img = TF.rotate(img, angle)
    theta_rad = math.radians(-angle)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    for i in range(pts.shape[0]):
        if mask[i] == 1.0:
            cx = pts[i, 0].item() - 0.5
            cy = pts[i, 1].item() - 0.5
            nx = cos_t * cx - sin_t * cy + 0.5
            ny = sin_t * cx + cos_t * cy + 0.5
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                pts[i, 0] = nx
                pts[i, 1] = ny
            else:
                mask[i]   = 0.0
                pts[i, 0] = 0.0
                pts[i, 1] = 0.0

    if random.random() > 0.4:
        tx = random.uniform(-0.08, 0.08)
        ty = random.uniform(-0.08, 0.08)
        tx_px = int(tx * W)
        ty_px = int(ty * H)
        img = TF.affine(img, angle=0, translate=[tx_px, ty_px], scale=1.0, shear=0)
        for i in range(pts.shape[0]):
            if mask[i] == 1.0:
                nx = pts[i, 0].item() + tx
                ny = pts[i, 1].item() + ty
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    pts[i, 0] = nx
                    pts[i, 1] = ny
                else:
                    mask[i]   = 0.0
                    pts[i, 0] = 0.0
                    pts[i, 1] = 0.0

    if random.random() > 0.4:
        scale    = random.uniform(0.85, 1.15)
        new_h    = int(H * scale)
        new_w    = int(W * scale)
        img_scaled = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        if scale < 1.0:
            pad_top  = (H - new_h) // 2
            pad_left = (W - new_w) // 2
            img = torch.zeros(C, H, W)
            img[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_scaled
            offset_x_v = pad_left / W
            offset_y_v = pad_top  / H
            for i in range(pts.shape[0]):
                if mask[i] == 1.0:
                    pts[i, 0] = pts[i, 0].item() * scale + offset_x_v
                    pts[i, 1] = pts[i, 1].item() * scale + offset_y_v
        else:
            crop_top  = (new_h - H) // 2
            crop_left = (new_w - W) // 2
            img = img_scaled[:, crop_top:crop_top + H, crop_left:crop_left + W]
            offset_x_v = crop_left / new_w
            offset_y_v = crop_top  / new_h
            for i in range(pts.shape[0]):
                if mask[i] == 1.0:
                    nx = (pts[i, 0].item() - offset_x_v) / scale
                    ny = (pts[i, 1].item() - offset_y_v) / scale
                    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                        pts[i, 0] = nx
                        pts[i, 1] = ny
                    else:
                        mask[i]   = 0.0
                        pts[i, 0] = 0.0
                        pts[i, 1] = 0.0

    img = TF.adjust_brightness(img, random.uniform(0.65, 1.35))
    img = TF.adjust_contrast(img,   random.uniform(0.65, 1.35))

    if random.random() > 0.5:
        gamma = random.uniform(0.7, 1.4)
        img = torch.clamp(img, 1e-6, 1.0).pow(gamma)

    img = torch.clamp(img + torch.randn_like(img) * random.uniform(0.01, 0.04), 0.0, 1.0)

    if random.random() > 0.7:
        sigma_blur = random.uniform(0.3, 1.0)
        k      = 3
        x_kern = torch.arange(k, dtype=torch.float32) - k // 2
        kern1d = torch.exp(-x_kern ** 2 / (2 * sigma_blur ** 2))
        kern1d = kern1d / kern1d.sum()
        kern2d = (kern1d[:, None] @ kern1d[None, :]).unsqueeze(0).unsqueeze(0)
        img = F.conv2d(img.unsqueeze(0), kern2d, padding=k // 2).squeeze(0)
        img = torch.clamp(img, 0.0, 1.0)

    return img, pts, mask