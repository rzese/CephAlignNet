import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

try:
    from src.landmark_stats import SIGMA_PER_LANDMARK, PESI_LANDMARK
except ImportError:
    try:
        from landmark_stats import SIGMA_PER_LANDMARK, PESI_LANDMARK
    except ImportError:
        print("[WARN] src/landmark_stats.py non trovato — uso valori di fallback "
              "(esegui `python -m src.analiz_land` per generarli dai dati).")

        SIGMA_PER_LANDMARK = {
            5:  5.0,   # Or
            9:  6.5,   # R  
            10: 5.5,   # S  
            11: 5.5,   # Ar 
            14: 6.0,   # Go 
            15: 6.0,   # Po 
            18: 6.0,   # LMT 
            19: 5.0,   # UIT
            20: 5.5,   # UIA
            21: 5.5,   # LIA
        }
        PESI_LANDMARK = {
            5:  2.0,   # Or
            9:  3.5,   # R  
            10: 2.0,   # S  
            11: 2.0,   # Ar 
            12: 2.5,   # Co 
            13: 2.0,   # Gn 
            14: 3.0,   # Go 
            15: 3.5,   # Po 
            16: 2.0,   # LPM 
            17: 2.0,   # LIT 
            18: 3.0,   # LMT 
            19: 2.0,   # UIT
            20: 2.5,   # UIA
            21: 2.5,   # LIA
        }

# Resolution the offsets are trained at (must match heatmap_size in CONFIG)
HEATMAP_TRAIN_SIZE = 256

# Pixel-grid convention linking normalized coordinates in [0, 1] to heatmap pixels.
#   "centri": pixel j has its center at (j + 0.5) / size — the same convention as
#             F.interpolate(align_corners=False). Used by every new training run.
#   "legacy": the convention of checkpoints trained before the fix: targets were
#             rendered at x * size but soft-argmax decoded pixel j as j / (size - 1),
#             a mismatch of up to ~0.5 heatmap px. Kept only to decode old checkpoints
#             exactly as they were trained/evaluated.
GRIGLIE = ("centri", "legacy")


def griglia_normalizzata(n, device=None, griglia="centri"):
    """Normalized coordinate of each of the n pixel centers along one axis."""
    if griglia == "legacy":
        return torch.linspace(0, 1, n, device=device)
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n


def coordinate_in_pixel_heatmap(coord_norm, size, griglia="centri"):
    """Normalized coordinate -> continuous position in heatmap pixel units
    (inverse of griglia_normalizzata for the 'centri' grid)."""
    if griglia == "legacy":
        return coord_norm * size
    return coord_norm * size - 0.5


def ruota_coordinate(coords, angle_deg):
    """
    Applies to normalized coordinates [..., 2] the same transform that
    TF.rotate(img, angle_deg) applies to a square image (counter-clockwise
    rotation around the center, y pointing down).
    To undo a rotation use ruota_coordinate(coords, -angle_deg).
    """
    theta = math.radians(-angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = coords.clone()
    cx = coords[..., 0] - 0.5
    cy = coords[..., 1] - 0.5
    out[..., 0] = cos_t * cx - sin_t * cy + 0.5
    out[..., 1] = sin_t * cx + cos_t * cy + 0.5
    return out


def regression_voting(output, n_landmarks, R=40, target_size=HEATMAP_TRAIN_SIZE, griglia="centri"):
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
    centro = 0.0 if griglia == "legacy" else 0.5
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

            vote_x = cols.float() + centro + ox.view(-1)[topk] * R
            vote_y = rows.float() + centro + oy.view(-1)[topk] * R

            px = vote_x.mean() / W
            py = vote_y.mean() / H
            pts.append(torch.stack([px.clamp(0, 1), py.clamp(0, 1)]))
        coords.append(torch.stack(pts))
    return torch.stack(coords)   # [B, N, 2]


def combined_loss(pred, heatmap_gt, pts_gt, mask, R=40, alpha=2/3,
                  tipo_loss_heatmap="awing", griglia="centri"):
    """Combines a heatmap loss with an L1 loss on the local offset maps,
    restricted to pixels within radius R of the ground-truth point.

    tipo_loss_heatmap:
      "awing" — Adaptive Wing against a fixed-sigma target (heatmap_mode "fixed")
      "mse"   — MSE against a target rendered with learnable sigma ("learned_sigma")
      "nll"   — the heatmap channel is a logit map; negative log-likelihood of the
                ground-truth position under its spatial softmax ("probabilistic").
                heatmap_gt is ignored (may be None).
    """

    B, _, H, W = pred.shape
    N = pts_gt.shape[1]

    pred_hm = pred[:, 0::3]
    pred_ox = pred[:, 1::3]
    pred_oy = pred[:, 2::3]

    if tipo_loss_heatmap == "awing":
        loss_hm = awing_loss(pred_hm, heatmap_gt, mask)
    elif tipo_loss_heatmap == "mse":
        loss_hm = mse_heatmap_loss(pred_hm, heatmap_gt, mask)
    elif tipo_loss_heatmap == "nll":
        loss_hm = nll_heatmap_loss(pred_hm, pts_gt, mask, griglia=griglia)
    else:
        raise ValueError(f"tipo_loss_heatmap sconosciuto: {tipo_loss_heatmap}")

    grid_x = torch.arange(W, device=pred.device).float().view(1, 1, 1, W).expand(B, N, H, W)
    grid_y = torch.arange(H, device=pred.device).float().view(1, 1, H, 1).expand(B, N, H, W)
    gt_x   = coordinate_in_pixel_heatmap(pts_gt[:, :, 0], W, griglia).view(B, N, 1, 1).expand_as(grid_x)
    gt_y   = coordinate_in_pixel_heatmap(pts_gt[:, :, 1], H, griglia).view(B, N, 1, 1).expand_as(grid_y)

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


def coordinate_a_heatmap(pts, mask, img_size=512, heatmap_size=256, sigma=4.0, griglia="centri"):
    """Converts normalized ground-truth (x, y) coordinates into per-landmark
    Gaussian heatmaps (fixed sigma, peak 1) used as the Stage 1 training target."""

    N = pts.shape[0]
    heatmaps = np.zeros((N, heatmap_size, heatmap_size), dtype=np.float32)
    for i in range(N):
        if mask[i].item() == 1.0:
            s  = SIGMA_PER_LANDMARK.get(i, sigma)
            cx = coordinate_in_pixel_heatmap(pts[i, 0].item(), heatmap_size, griglia)
            cy = coordinate_in_pixel_heatmap(pts[i, 1].item(), heatmap_size, griglia)
            if -0.5 <= cx < heatmap_size and -0.5 <= cy < heatmap_size:
                heatmaps[i] = crea_heatmap_gaussiana(heatmap_size, heatmap_size, cx, cy, s)
    return torch.from_numpy(heatmaps)


def heatmap_a_coordinate(heatmaps, temperatura=100.0, griglia="centri"):
    """Soft-argmax decoding: converts predicted heatmaps into normalized
    (x, y) coordinates via a temperature-scaled softmax expectation.
    This is the default, stable decoding used at inference (USE_VOTING=False).
    For heatmap_mode "probabilistic" the heatmap is a logit map and
    temperatura=1 gives the expectation of the predicted distribution."""
    B, N, H, W = heatmaps.shape
    flat    = heatmaps.reshape(B, N, -1) * temperatura
    weights = torch.softmax(flat, dim=2).view(B, N, H, W)
    grid_x  = griglia_normalizzata(W, heatmaps.device, griglia).view(1, 1, 1, W)
    grid_y  = griglia_normalizzata(H, heatmaps.device, griglia).view(1, 1, H, 1)
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


def _pesi_landmark(N, device):
    pesi = torch.ones(N, device=device)
    for idx, peso in PESI_LANDMARK.items():
        if idx < N:
            pesi[idx] = peso
    return pesi


def mse_heatmap_loss(pred, target, mask, energia_rif=None):
    """
    MSE between predicted heatmaps and targets rendered with learnable sigma
    (heatmap_mode "learned_sigma", Payer et al. 2020). The squared error is
    summed over the pixels of each landmark and divided by energia_rif — the
    energy of a peak-1 Gaussian with the reference sigma (pi * 4^2) — so that
    1.0 means "target completely missed" whatever the heatmap size.
    Weighted per landmark with PESI_LANDMARK, like awing_loss.
    """
    B, N, H, W = pred.shape
    if energia_rif is None:
        energia_rif = math.pi * 4.0 ** 2
    peso = mask * _pesi_landmark(N, pred.device).unsqueeze(0)          # [B, N]
    per_lm = ((pred - target) ** 2).sum(dim=(2, 3)) / energia_rif       # [B, N]
    return (per_lm * peso).sum() / peso.sum().clamp(min=1e-6)


def nll_heatmap_loss(logits, pts, mask, griglia="centri"):
    """
    Negative log-likelihood of the ground-truth position under the spatial
    softmax of each logit map (heatmap_mode "probabilistic"). The continuous
    position is spread over its 4 neighbouring pixels with bilinear weights.
    Minimizing it makes the predicted distribution match the model's actual
    localization error, so its spread can be read as uncertainty.
    Divided by log(H*W) so that a uniform prediction gives ~1.0.
    """
    B, N, H, W = logits.shape
    logp = F.log_softmax(logits.reshape(B, N, -1), dim=2)

    u = coordinate_in_pixel_heatmap(pts[..., 0], W, griglia).clamp(0, W - 1)
    v = coordinate_in_pixel_heatmap(pts[..., 1], H, griglia).clamp(0, H - 1)
    u0 = u.floor().clamp(max=W - 2)
    v0 = v.floor().clamp(max=H - 2)
    du, dv = u - u0, v - v0
    u0, v0 = u0.long(), v0.long()

    def lp(vv, uu):
        return torch.gather(logp, 2, (vv * W + uu).unsqueeze(-1)).squeeze(-1)

    log_lik = ((1 - du) * (1 - dv) * lp(v0, u0) + du * (1 - dv) * lp(v0, u0 + 1)
               + (1 - du) * dv * lp(v0 + 1, u0) + du * dv * lp(v0 + 1, u0 + 1))

    peso = mask * _pesi_landmark(N, logits.device).unsqueeze(0)
    nll = -(log_lik * peso).sum() / peso.sum().clamp(min=1e-6)
    return nll / math.log(H * W)


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


def geometric_consistency_loss(pred_pts, mask, gt_pts=None):
    """
    Penalizes geometrically inconsistent predictions between related
    landmarks.

    Enforced anatomical relationships:
      - Gn (13) must be close to the midpoint of Me (3) and Pog (6)
      - S (10) must be behind N (4) along the face direction
      - Me (3) must be lower (larger y) than B (2)
      - B (2) must be lower than A (0)
      - ANS (1) must be in front of PNS (7) along the face direction
      - Co (12) must be higher (smaller y) than Go (14)
      - UIT (19) must be lower (larger y) than UIA (20): crown tip below apex
      - LIA (21) must be lower (larger y) than LIT (17): apex below crown tip

    The two horizontal constraints depend on which way the face looks. With
    gt_pts the direction is read per sample from the ground truth (sign of
    N_x - S_x), so they stay correct on horizontally flipped images and on
    left-facing radiographs; if S or N are not annotated those two terms are
    skipped. Without gt_pts a right-facing face is assumed (old behaviour).
    """
    B    = pred_pts.shape[0]
    loss = torch.tensor(0.0, device=pred_pts.device)
    n_terms = 0

    IDX = {"A": 0, "ANS": 1, "B": 2, "Me": 3, "N": 4, "Pog": 6, "PNS": 7, "S": 10,
           "Co": 12, "Gn": 13, "Go": 14,
           "LIT": 17, "UIT": 19, "UIA": 20, "LIA": 21}

    for b in range(B):
        pts = pred_pts[b]
        msk = mask[b]

        # +1: face looks right (N has larger x than S), -1: face looks left,
        # None: unknown -> skip the horizontal constraints
        if gt_pts is None:
            verso = 1.0
        elif msk[IDX["S"]] and msk[IDX["N"]]:
            verso = 1.0 if gt_pts[b, IDX["N"], 0] >= gt_pts[b, IDX["S"], 0] else -1.0
        else:
            verso = None

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

        # 4. ANS in front of PNS
        if verso is not None and msk[IDX["ANS"]] and msk[IDX["PNS"]]:
            viol = torch.clamp(verso * (pts[IDX["PNS"], 0] - pts[IDX["ANS"], 0]), min=0)
            loss += viol ** 2
            n_terms += 1

        # 5. Co higher than Go
        if msk[IDX["Co"]] and msk[IDX["Go"]]:
            viol = torch.clamp(pts[IDX["Co"], 1] - pts[IDX["Go"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

        # 6. S behind N
        if verso is not None and msk[IDX["S"]] and msk[IDX["N"]]:
            viol = torch.clamp(verso * (pts[IDX["S"], 0] - pts[IDX["N"], 0]), min=0)
            loss += viol ** 2
            n_terms += 1

        # 7. UIT lower than UIA: the upper incisor root apex sits up in the
        #    maxilla, its crown tip points down (larger y)
        if msk[IDX["UIT"]] and msk[IDX["UIA"]]:
            viol = torch.clamp(pts[IDX["UIA"], 1] - pts[IDX["UIT"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

        # 8. LIA lower than LIT (lower root apex below crown tip)
        if msk[IDX["LIA"]] and msk[IDX["LIT"]]:
            viol = torch.clamp(pts[IDX["LIT"], 1] - pts[IDX["LIA"], 1], min=0)
            loss += viol ** 2
            n_terms += 1

    return loss / max(n_terms, 1)


def crop_campo_augmenta(img, pts, mask, pixel_size, margine_mm=5.0):
    """
    Field-of-view crop on the ORIGINAL image (before the resize to 512), to
    simulate radiographs taken with a narrower field ("zoomed" images, like part
    of the UNIFE tracings). Each side is cut by a random amount between 0 and the
    largest cut that keeps every annotated landmark at least margine_mm inside
    the image, so no landmark is lost; the sides are independent, so the aspect
    ratio changes too. Not-annotated landmarks (mask 0) stay at [0, 0].
    img [C, H, W], pts [N, 2] in pixels -> (cropped img, shifted pts).
    """
    C, H, W = img.shape
    validi = mask > 0
    if not validi.any():
        return img, pts
    m_px = margine_mm / float(pixel_size)
    v = pts[validi]
    x_min, y_min = v[:, 0].min().item() - m_px, v[:, 1].min().item() - m_px
    x_max, y_max = v[:, 0].max().item() + m_px, v[:, 1].max().item() + m_px
    x1 = int(random.uniform(0.0, max(0.0, x_min)))
    y1 = int(random.uniform(0.0, max(0.0, y_min)))
    x2 = int(math.ceil(random.uniform(min(float(W), x_max), float(W))))
    y2 = int(math.ceil(random.uniform(min(float(H), y_max), float(H))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return img, pts
    pts_out = pts.clone()
    pts_out[validi, 0] -= x1
    pts_out[validi, 1] -= y1
    return img[:, y1:y2, x1:x2], pts_out


def zoom_crop_augmenta(img, pts, mask, zoom_factor=2.2):
    """
    Multi-scale augmentation for hard landmarks: crops a region centered
    on the average position of the historically difficult landmarks
    (index list below) and resizes it back to the original resolution,
    giving the model extra local detail where it usually makes mistakes.
    Difficult landmarks by index: 5=Or, 10=S, 15=Po, 16=LPM,
    19=UIT, 20=UIA, 21=LIA
    """

    HARD = [5, 10, 15, 16, 19, 20, 21]
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