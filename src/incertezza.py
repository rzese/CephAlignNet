"""
incertezza.py — learnable heatmap sigma and per-landmark uncertainty.

Three Stage 1 heatmap modes (CONFIG["heatmap_mode"] in train_heatmap.py):

  "fixed"          Original behaviour: Gaussian targets with a fixed sigma per
                   landmark (SIGMA_PER_LANDMARK), Adaptive Wing loss.
  "learned_sigma"  The target sigma of every landmark (sigma_x, sigma_y and the
                   correlation rho, i.e. an oriented ellipse) is a trainable
                   parameter, optimized together with the network
                   (Payer et al., 2020). A penalty on the ellipse area keeps
                   sigma from growing without bound. The learned sigma is a
                   per-landmark (dataset-level) uncertainty.
  "probabilistic"  The heatmap channel is a logit map; its spatial softmax is a
                   probability distribution over the landmark position, trained
                   with the negative log-likelihood of the ground truth. Its
                   spread is a per-image uncertainty.

In every mode a per-image uncertainty can be read from the predicted heatmap
(momenti_heatmap: mean and 2x2 covariance) and from the disagreement between
TTA predictions (covarianza_tta). Raw spreads are not calibrated: they are
mapped to radii in mm with a per-landmark conformal factor estimated on the
validation set (analisi_incertezza, evaluate_refine.py).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.heatmap_utils import griglia_normalizzata, coordinate_in_pixel_heatmap

MODALITA_HEATMAP = ("fixed", "learned_sigma", "probabilistic")

# Soft-argmax temperature used to decode the heatmaps of each mode.
# "probabilistic" heatmaps are logits: T=1 is the expectation of the distribution.
TEMPERATURA_DECODIFICA = {"fixed": 100.0, "learned_sigma": 100.0, "probabilistic": 1.0}

# Uncertainty sources compared in evaluation and usable at inference
FONTI_INCERTEZZA = ("heatmap", "tta", "totale")


def _inv_softplus(y):
    return math.log(math.expm1(y))


class SigmaAddestrabile(nn.Module):
    """
    One trainable anisotropic Gaussian per landmark (sigma_x, sigma_y in
    heatmap pixels, correlation rho), used to render the Stage 1 targets in
    heatmap_mode "learned_sigma".

    Targets are normalized Gaussians scaled by gamma = 2*pi*sigma_rif^2, so a
    target with sigma = sigma_rif has peak 1 (same scale as the fixed targets)
    and a wider target has a lower peak: widening sigma is not free, which
    together with penalita() prevents the trivial "predict everything blurry"
    solution.
    """
    def __init__(self, num_landmarks=26, sigma_iniziale=4.0, sigma_per_landmark=None,
                 sigma_min=1.0, sigma_max=12.0, sigma_rif=4.0):
        super().__init__()
        sigma_per_landmark = sigma_per_landmark or {}
        init = [float(sigma_per_landmark.get(i, sigma_iniziale)) for i in range(num_landmarks)]
        raw = torch.tensor([_inv_softplus(max(s - sigma_min, 1e-3)) for s in init])
        self.raw_sx  = nn.Parameter(raw.clone())
        self.raw_sy  = nn.Parameter(raw.clone())
        self.raw_rho = nn.Parameter(torch.zeros(num_landmarks))
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_rif = float(sigma_rif)
        self.gamma     = 2 * math.pi * self.sigma_rif ** 2

    def parametri(self):
        """Returns (sigma_x, sigma_y, rho), each [N]; sigmas in heatmap pixels."""
        sx  = (self.sigma_min + F.softplus(self.raw_sx)).clamp(max=self.sigma_max)
        sy  = (self.sigma_min + F.softplus(self.raw_sy)).clamp(max=self.sigma_max)
        rho = 0.9 * torch.tanh(self.raw_rho)
        return sx, sy, rho

    def genera(self, pts, mask, size, griglia="centri"):
        """
        Renders the targets for a batch: pts [B, N, 2] normalized, mask [B, N]
        -> heatmaps [B, N, size, size]. Differentiable w.r.t. the sigmas.
        """
        sx, sy, rho = self.parametri()
        B, N, _ = pts.shape
        dev = pts.device
        u = coordinate_in_pixel_heatmap(pts[..., 0], size, griglia).view(B, N, 1, 1)
        v = coordinate_in_pixel_heatmap(pts[..., 1], size, griglia).view(B, N, 1, 1)
        xs = torch.arange(size, device=dev, dtype=torch.float32).view(1, 1, 1, size)
        ys = torch.arange(size, device=dev, dtype=torch.float32).view(1, 1, size, 1)
        sx, sy, r = sx.view(1, N, 1, 1), sy.view(1, N, 1, 1), rho.view(1, N, 1, 1)
        dx = (xs - u) / sx
        dy = (ys - v) / sy
        q = (dx ** 2 - 2 * r * dx * dy + dy ** 2) / (1 - r ** 2)
        norm = self.gamma / (2 * math.pi * sx * sy * torch.sqrt(1 - r ** 2))
        return norm * torch.exp(-0.5 * q) * mask.view(B, N, 1, 1)

    def penalita(self):
        """Mean ellipse area relative to the reference sigma (1.0 at sigma_rif)."""
        sx, sy, rho = self.parametri()
        return (sx * sy * torch.sqrt(1 - rho ** 2)).mean() / self.sigma_rif ** 2

    def descrivi(self, nomi=None, mm_per_px_heatmap=None):
        """One line per landmark with the learned sigmas, for logs."""
        sx, sy, rho = (t.detach().cpu() for t in self.parametri())
        righe = []
        for i in range(len(sx)):
            nome = nomi[i] if nomi and i < len(nomi) else str(i)
            riga = f"  {i:2d} {nome:5s} sx={sx[i]:.2f}px sy={sy[i]:.2f}px rho={rho[i]:+.2f}"
            if mm_per_px_heatmap:
                riga += f"  (~{sx[i]*mm_per_px_heatmap:.2f} x {sy[i]*mm_per_px_heatmap:.2f} mm)"
            righe.append(riga)
        return "\n".join(righe)


# ─── Uncertainty from a predicted heatmap ─────────────────────────────────────
def momenti_heatmap(heatmaps, modalita="fixed", griglia="centri", temperatura=1.0,
                    raggio_finestra=16):
    """
    Treats each predicted heatmap as a distribution over the landmark position
    and returns its mean [B, N, 2] and covariance [B, N, 2, 2] in normalized
    coordinates, plus the peak value [B, N].

    - "probabilistic": the distribution is softmax(logits / temperatura) over
      the whole map — exactly what the model was trained to output.
    - other modes: the heatmap is a regression target, not a distribution;
      negative values are clipped and only a (2R+1)^2 window around the maximum
      is used, so the near-zero background over 65k pixels does not inflate the
      variance. The result includes the width of the training target itself
      (a floor of about sigma_target), which calibration takes into account.
    """
    B, N, H, W = heatmaps.shape
    dev = heatmaps.device
    gx = griglia_normalizzata(W, dev, griglia).view(1, 1, 1, W)
    gy = griglia_normalizzata(H, dev, griglia).view(1, 1, H, 1)
    flat = heatmaps.reshape(B, N, -1)

    if modalita == "probabilistic":
        p = torch.softmax(flat / temperatura, dim=2).view(B, N, H, W)
        picco = p.reshape(B, N, -1).max(dim=2).values
    else:
        picco, idx = flat.max(dim=2)
        r0 = (idx // W).view(B, N, 1, 1)
        c0 = (idx % W).view(B, N, 1, 1)
        rows = torch.arange(H, device=dev).view(1, 1, H, 1)
        cols = torch.arange(W, device=dev).view(1, 1, 1, W)
        finestra = ((rows - r0).abs() <= raggio_finestra) & ((cols - c0).abs() <= raggio_finestra)
        p = heatmaps.clamp(min=0) * finestra
        p = p / p.sum(dim=(2, 3), keepdim=True).clamp(min=1e-12)

    mx = (p * gx).sum(dim=(2, 3))
    my = (p * gy).sum(dim=(2, 3))
    dx = gx - mx.view(B, N, 1, 1)
    dy = gy - my.view(B, N, 1, 1)
    cxx = (p * dx * dx).sum(dim=(2, 3))
    cyy = (p * dy * dy).sum(dim=(2, 3))
    cxy = (p * dx * dy).sum(dim=(2, 3))
    cov = torch.stack([torch.stack([cxx, cxy], -1), torch.stack([cxy, cyy], -1)], -2)
    return torch.stack([mx, my], dim=-1), cov, picco


def covarianza_tta(predizioni):
    """predizioni [K, B, N, 2] (one entry per TTA variant) -> covariance
    [B, N, 2, 2] of their spread (zeros if K == 1)."""
    K = predizioni.shape[0]
    if K < 2:
        B, N = predizioni.shape[1:3]
        return torch.zeros(B, N, 2, 2, device=predizioni.device)
    d = predizioni - predizioni.mean(dim=0, keepdim=True)
    return (d.unsqueeze(-1) * d.unsqueeze(-2)).mean(dim=0)


def cov_in_mm(cov, dims, pixel_size):
    """
    Normalized covariance [B, N, 2, 2] -> mm^2, for images of original size
    dims [B, 2] = (W, H) in pixels and pixel_size [B] in mm/px.
    x and y are scaled separately (non-square images).
    """
    s = dims.float() * pixel_size.float().view(-1, 1)           # mm per normalized unit, [B, 2]
    s = s.view(-1, 1, 2)
    return cov * s.unsqueeze(-1) * s.unsqueeze(-2)


def sigma_radiale(cov_mm):
    """sqrt(trace): RMS radial spread in mm of the distribution with that covariance."""
    return torch.sqrt((cov_mm[..., 0, 0] + cov_mm[..., 1, 1]).clamp(min=0))


# ─── Calibration and evaluation of the uncertainty ────────────────────────────
def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _quantile_conforme(punteggi, copertura):
    """Split-conformal quantile: with n calibration scores, the
    ceil((n+1)*c)/n empirical quantile guarantees coverage >= c."""
    n = len(punteggi)
    if n == 0:
        return None
    livello = min(1.0, math.ceil((n + 1) * copertura) / n)
    return float(np.quantile(punteggi, livello, method="higher"))


def analisi_incertezza(record, num_landmarks, copertura=0.9, fonti=FONTI_INCERTEZZA):
    """
    record: dict of 1-D arrays, one entry per (image, landmark):
        'img' (image index), 'lm' (landmark index), 'err' (radial error, mm)
        and 'sigma_<fonte>' (uncertainty in mm) for every source.

    For every source returns:
      - Spearman correlation between uncertainty and error (global and per
        landmark): does a higher uncertainty really mean a larger error?
      - mean error per uncertainty quintile
      - per-landmark conformal factor q: radius = q * sigma contains the true
        point with probability >= copertura. q is estimated on the even images
        and the coverage is checked on the odd ones; the q saved for
        inference is re-estimated on all the images.
    """
    img = np.asarray(record["img"])
    lm  = np.asarray(record["lm"])
    err = np.asarray(record["err"], float)
    cal = (img % 2) == 0
    risultati = {"copertura": copertura, "fonti": {}}

    for fonte in fonti:
        sig = np.asarray(record[f"sigma_{fonte}"], float)
        valido = sig > 0
        if not valido.any():
            continue
        punteggio = np.where(valido, err / np.maximum(sig, 1e-9), np.inf)

        q_cal, q_tutti, rho_lm = [None] * num_landmarks, [None] * num_landmarks, [None] * num_landmarks
        coperti, n_test, raggi = 0, 0, []
        for i in range(num_landmarks):
            sel = (lm == i) & valido
            if sel.sum() < 4:
                continue
            rho_lm[i]   = _spearman(sig[sel], err[sel])
            q_tutti[i]  = _quantile_conforme(punteggio[sel], copertura)
            q_cal[i]    = _quantile_conforme(punteggio[sel & cal], copertura)
            test = sel & ~cal
            if q_cal[i] is not None and test.any():
                r = q_cal[i] * sig[test]
                coperti += int((err[test] <= r).sum())
                n_test  += int(test.sum())
                raggi.extend(r.tolist())

        quintili = []
        s_v, e_v = sig[valido], err[valido]
        if len(s_v) >= 5:
            bordi = np.quantile(s_v, [0, .2, .4, .6, .8, 1])
            for k in range(5):
                sel = (s_v >= bordi[k]) & ((s_v <= bordi[k + 1]) if k == 4 else (s_v < bordi[k + 1]))
                quintili.append(float(e_v[sel].mean()) if sel.any() else float("nan"))

        risultati["fonti"][fonte] = {
            "spearman": _spearman(s_v, e_v),
            "spearman_per_landmark": rho_lm,
            "errore_per_quintile": quintili,
            "copertura_test": coperti / n_test if n_test else float("nan"),
            "raggio_medio_test_mm": float(np.mean(raggi)) if raggi else float("nan"),
            "q": q_tutti,
        }
    return risultati


def stampa_analisi_incertezza(risultati, nomi=None):
    """Formatted report of analisi_incertezza()."""
    c = risultati["copertura"]
    print(f"\n{'='*55}\nUNCERTAINTY (target coverage {c:.0%})\n{'='*55}")
    print("A useful uncertainty has a clearly positive Spearman correlation with the error,\n"
          "a mean error that grows across quintiles, and test coverage close to the target.")
    for fonte, r in risultati["fonti"].items():
        quint = " / ".join(f"{v:.2f}" for v in r["errore_per_quintile"])
        print(f"\n[{fonte}]  Spearman(sigma, error) = {r['spearman']:+.3f}")
        print(f"  mean error per uncertainty quintile (mm, low -> high): {quint}")
        print(f"  coverage on held-out half: {r['copertura_test']:.1%} "
              f"| mean radius {r['raggio_medio_test_mm']:.2f} mm")
        per_lm = [
            f"{(nomi[i] if nomi else i)}:{v:+.2f}"
            for i, v in enumerate(r["spearman_per_landmark"]) if v is not None and not math.isnan(v)
        ]
        if per_lm:
            print("  Spearman per landmark: " + "  ".join(per_lm))
