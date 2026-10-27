"""
Unit tests on synthetic data (CPU, no dataset or checkpoint needed).

    PYTHONPATH=. python Test/test_unita.py        # plain runner
    PYTHONPATH=. python -m pytest Test/test_unita.py
"""
import math
import os
import random
import sys
import tempfile
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.heatmap_utils import (ruota_coordinate, augmenta, coordinate_a_heatmap, heatmap_a_coordinate,
                               combined_loss, nll_heatmap_loss, regression_voting,
                               geometric_consistency_loss)
from src.incertezza import (SigmaAddestrabile, momenti_heatmap, cov_in_mm, sigma_radiale,
                            covarianza_tta, analisi_incertezza)
from src.inferenza import errori_mm, applica_stage2, collate_base, ritaglia_originali, seleziona_punti
from src.model_refine import RefinementNet, ImageRefinementNet


def _immagine_con_punto(px, py, H=512, W=512):
    img = torch.zeros(1, 1, H, W)
    img[0, 0, py - 2:py + 3, px - 2:px + 3] = 1.0
    return img


def _baricentro(img, soglia=0.3):
    ys, xs = torch.nonzero(img[0, 0] > soglia, as_tuple=True)
    return torch.tensor([[[(xs.float().mean() + 0.5) / img.shape[3],
                           (ys.float().mean() + 0.5) / img.shape[2]]]])


def test_derotazione_tta_inverte_la_rotazione():
    """Bug 1: TTA predictions must be rotated back by -angle."""
    for angle in (-5.0, -2.5, 2.5, 5.0, 15.0):
        img = _immagine_con_punto(400, 150)
        vero = _baricentro(img)
        ruotato = _baricentro(TF.rotate(img, angle))
        assert torch.allclose(ruota_coordinate(vero, angle), ruotato, atol=3e-3), angle
        assert torch.allclose(ruota_coordinate(ruotato, -angle), vero, atol=3e-3), angle


def test_augmenta_coordinate_coerenti_con_immagine():
    """The landmarks moved by augmenta must land on the moved image content."""
    random.seed(0); torch.manual_seed(0)
    for _ in range(20):
        img = torch.zeros(1, 512, 512)
        pts = torch.tensor([[0.3, 0.3], [0.6, 0.4], [0.45, 0.7]])
        for x, y in pts:
            c, r = int(x * 512), int(y * 512)
            img[0, r - 3:r + 4, c - 3:c + 4] = 1.0
        img_a, pts_a, mask_a = augmenta(img.clone(), pts.clone(), torch.ones(3))
        for (x, y), m in zip(pts_a, mask_a):
            if m == 0:
                continue
            c, r = x.item() * 512, y.item() * 512
            r0, c0 = int(r), int(c)
            win = img_a[0, max(r0 - 8, 0):r0 + 9, max(c0 - 8, 0):c0 + 9]
            ys, xs = torch.meshgrid(torch.arange(win.shape[0]), torch.arange(win.shape[1]), indexing="ij")
            w = (win - win.median()).clamp(min=0)
            cy = (w * ys).sum() / w.sum() + max(r0 - 8, 0)
            cx = (w * xs).sum() / w.sum() + max(c0 - 8, 0)
            assert abs(cx - c) < 2.5 and abs(cy - r) < 2.5, (cx, c, cy, r)


def test_crop_campo_tiene_tutti_i_punti_e_li_sposta_con_l_immagine():
    from src.heatmap_utils import crop_campo_augmenta
    H, W, ps, margine = 600, 800, 0.1, 5.0
    punti = torch.tensor([[200.0, 150.0], [600.0, 450.0], [0.0, 0.0]])
    mask = torch.tensor([1.0, 1.0, 0.0])
    img = torch.zeros(1, H, W)
    img[0, 148:153, 198:203] = 1.0                       # 5x5 spot centred on pixel (200, 150)
    for _ in range(50):
        r, p = crop_campo_augmenta(img, punti, mask, ps, margine)
        h, w = r.shape[1:]
        assert h <= H and w <= W
        for k in range(2):   # annotated points stay at least margine_mm inside
            assert margine / ps - 1 <= p[k, 0] <= w - margine / ps + 1
            assert margine / ps - 1 <= p[k, 1] <= h - margine / ps + 1
        assert p[2].tolist() == [0.0, 0.0]
        ys, xs = torch.nonzero(r[0] > 0.5, as_tuple=True)
        assert abs(xs.float().mean() - p[0, 0]) < 0.5 and abs(ys.float().mean() - p[0, 1]) < 0.5


def test_griglia_centri_codifica_decodifica_senza_bias():
    """Grid fix: target rendering and soft-argmax decoding agree."""
    pts = torch.rand(26, 2) * 0.8 + 0.1
    hm = coordinate_a_heatmap(pts, torch.ones(26), heatmap_size=256, sigma=4.0, griglia="centri")
    dec = heatmap_a_coordinate(hm.unsqueeze(0), temperatura=100.0, griglia="centri")[0]
    assert (dec - pts).abs().max() < 5e-4          # < 0.12 mm on a 2400 px image
    # the legacy convention has a systematic scale bias of ~1/256
    hm_l = coordinate_a_heatmap(pts, torch.ones(26), heatmap_size=256, sigma=4.0, griglia="legacy")
    dec_l = heatmap_a_coordinate(hm_l.unsqueeze(0), temperatura=100.0, griglia="legacy")[0]
    assert (dec_l - pts).abs().max() > 1e-3


def test_regression_voting_offset_perfetti():
    pts = torch.tensor([[[0.3, 0.6], [0.71, 0.22]]])
    hm = coordinate_a_heatmap(pts[0], torch.ones(2), heatmap_size=64, sigma=2.0).unsqueeze(0)
    gx = torch.arange(64).float().view(1, 1, 1, 64)
    gy = torch.arange(64).float().view(1, 1, 64, 1)
    R = 10
    canali = []
    for i in range(2):
        u, v = pts[0, i, 0] * 64 - 0.5, pts[0, i, 1] * 64 - 0.5
        canali += [hm[:, i:i + 1], ((u - gx) / R).expand(1, 1, 64, 64), ((v - gy) / R).expand(1, 1, 64, 64)]
    out = torch.cat(canali, dim=1)
    dec = regression_voting(out, 2, R=R, target_size=64)
    assert (dec - pts).abs().max() < 1e-4


def test_sigma_addestrabile_converge_verso_la_larghezza_vera():
    """learned_sigma: with a fixed prediction of width 6 px and no penalty,
    optimizing only the sigma must recover ~6 px (and the gradient flows)."""
    torch.manual_seed(0)
    pts = torch.tensor([[[0.5, 0.5]]])
    vero = SigmaAddestrabile(1, sigma_iniziale=6.0, sigma_rif=4.0)
    with torch.no_grad():
        pred = vero.genera(pts, torch.ones(1, 1), 64)
    mod = SigmaAddestrabile(1, sigma_iniziale=3.0, sigma_rif=4.0)
    opt = torch.optim.Adam(mod.parameters(), lr=0.05)
    for _ in range(400):
        target = mod.genera(pts, torch.ones(1, 1), 64)
        loss = ((pred - target) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    sx, sy, rho = mod.parametri()
    assert abs(sx.item() - 6.0) < 0.2 and abs(sy.item() - 6.0) < 0.2 and abs(rho.item()) < 0.05
    # a target with sigma = sigma_rif has peak ~1
    rif = SigmaAddestrabile(1, sigma_iniziale=4.0, sigma_rif=4.0).genera(pts, torch.ones(1, 1), 64)
    assert abs(rif.max().item() - 1.0) < 0.05


def test_nll_impara_distribuzione_centrata_e_larga_quanto_l_errore():
    """probabilistic: free logits trained with NLL on noisy targets learn a
    distribution centered on the mean with spread ~ the noise."""
    torch.manual_seed(0)
    H = W = 64
    logits = torch.zeros(1, 1, H, W, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    centro, rumore = torch.tensor([0.4, 0.6]), 2.0 / 64      # 2 heatmap px of noise
    for _ in range(800):
        pts = (centro + torch.randn(32, 2) * rumore).view(32, 1, 2)
        loss = nll_heatmap_loss(logits.expand(32, 1, H, W), pts, torch.ones(32, 1))
        opt.zero_grad(); loss.backward(); opt.step()
    media, cov, _ = momenti_heatmap(logits.detach(), "probabilistic")
    assert (media[0, 0] - centro).abs().max() < 0.01
    std = cov[0, 0].diagonal().sqrt() * 64
    assert ((std - 2.0).abs() < 0.6).all(), std


def test_momenti_gaussiana_nota():
    pts = torch.tensor([[[0.5, 0.4]]])
    mod = SigmaAddestrabile(1, sigma_iniziale=3.0)
    with torch.no_grad():
        mod.raw_rho.fill_(0.0)
        hm = mod.genera(pts, torch.ones(1, 1), 128)
    media, cov, _ = momenti_heatmap(hm, "fixed", raggio_finestra=20)
    assert (media - pts).abs().max() < 1e-4
    assert abs(cov[0, 0, 0, 0].sqrt().item() * 128 - 3.0) < 0.05
    # probabilistic: logits = log of the Gaussian -> same covariance
    media_p, cov_p, _ = momenti_heatmap(torch.log(hm + 1e-30), "probabilistic")
    assert abs(cov_p[0, 0, 1, 1].sqrt().item() * 128 - 3.0) < 0.05


def test_errori_e_covarianze_in_mm_su_immagini_non_quadrate():
    pred = torch.tensor([[[0.5, 0.5]]]); gt = torch.tensor([[[0.6, 0.5]]])
    dims = torch.tensor([[2000.0, 1000.0]]); ps = torch.tensor([0.1])
    assert abs(errori_mm(pred, gt, dims, ps).item() - 20.0) < 1e-4       # 0.1 * 2000 px * 0.1 mm
    gt_y = torch.tensor([[[0.5, 0.6]]])
    assert abs(errori_mm(pred, gt_y, dims, ps).item() - 10.0) < 1e-4     # y scaled by the height
    cov = torch.diag(torch.tensor([0.01 ** 2, 0.01 ** 2])).view(1, 1, 2, 2)
    cmm = cov_in_mm(cov, dims, ps)
    assert torch.allclose(cmm[0, 0].diagonal(), torch.tensor([4.0, 1.0]))
    assert abs(sigma_radiale(cmm).item() - math.sqrt(5.0)) < 1e-5
    preds = torch.stack([torch.zeros(1, 1, 2), torch.ones(1, 1, 2) * 0.2])
    assert torch.allclose(covarianza_tta(preds)[0, 0], torch.full((2, 2), 0.01))


def test_vincoli_geometrici_rispettano_il_verso_del_volto():
    """Bug 4/12: the correct anatomy gives zero loss, facing right or left."""
    pts = torch.zeros(1, 26, 2)
    pos = {"A": (0.7, 0.55), "ANS": (0.72, 0.5), "B": (0.68, 0.75), "Me": (0.62, 0.9),
           "N": (0.75, 0.25), "Pog": (0.67, 0.83), "PNS": (0.45, 0.5), "S": (0.4, 0.3),
           "Co": (0.3, 0.38), "Gn": (0.645, 0.865), "Go": (0.35, 0.72),
           "LIT": (0.71, 0.62), "UIT": (0.72, 0.63), "UIA": (0.68, 0.5), "LIA": (0.65, 0.75)}
    idx = {"A": 0, "ANS": 1, "B": 2, "Me": 3, "N": 4, "Pog": 6, "PNS": 7, "S": 10,
           "Co": 12, "Gn": 13, "Go": 14, "LIT": 17, "UIT": 19, "UIA": 20, "LIA": 21}
    mask = torch.zeros(1, 26)
    for k, (x, y) in pos.items():
        pts[0, idx[k]] = torch.tensor([x, y]); mask[0, idx[k]] = 1
    assert geometric_consistency_loss(pts, mask, gt_pts=pts).item() < 1e-8
    flip = pts.clone(); flip[..., 0] = 1 - flip[..., 0]
    assert geometric_consistency_loss(flip, mask, gt_pts=flip).item() < 1e-8
    # a prediction with S in front of N on a right-facing face is penalized
    sbagliato = pts.clone(); sbagliato[0, 10, 0] = 0.9
    assert geometric_consistency_loss(sbagliato, mask, gt_pts=pts).item() > 0


def test_loss_tutte_le_modalita_fanno_backward():
    torch.manual_seed(0)
    B, N, H = 2, 26, 32
    pred = torch.randn(B, N * 3, H, H, requires_grad=True)
    pts = torch.rand(B, N, 2) * 0.8 + 0.1
    mask = torch.ones(B, N); mask[0, 5] = 0
    hm = torch.stack([coordinate_a_heatmap(pts[b], mask[b], heatmap_size=H, sigma=2.0) for b in range(B)])
    sig = SigmaAddestrabile(N, sigma_iniziale=2.0, sigma_rif=2.0)
    for tipo, target in (("awing", hm), ("mse", sig.genera(pts, mask, H)), ("nll", None)):
        loss = combined_loss(pred, target, pts, mask, R=8, tipo_loss_heatmap=tipo)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    assert sig.raw_sx.grad is not None and sig.raw_sx.grad.abs().sum() > 0


def test_stage2_vettorizzato_e_guardia():
    torch.manual_seed(0)
    net = RefinementNet(num_landmarks=26, in_channels=8, max_offset=0.016).eval()
    assert net.max_offset == 0.016
    x_out = torch.randn(2, 8, 64, 64)
    coords = torch.rand(2, 26, 2) * 0.6 + 0.2
    out = applica_stage2(net, coords, x_out=x_out, landmark_da_correggere={3, 7})
    diff = (out - coords).abs()
    assert diff[:, [i for i in range(26) if i not in (3, 7)]].max() == 0
    assert diff.max() <= 0.016 * 1.2 + 1e-7


def _immagine_con_punti(H, W, punti_px, raggio=4):
    img = torch.zeros(1, H, W)
    for x, y in punti_px:
        img[0, int(y) - raggio:int(y) + raggio + 1, int(x) - raggio:int(x) + raggio + 1] = 1.0
    return img


def test_ritaglio_originale_centrato_in_mm():
    """The crop is centered on the requested point and has the requested size in mm."""
    H, W, ps = 600, 500, torch.tensor([0.1])
    img = _immagine_con_punti(H, W, [(300, 200)], raggio=2)
    errore_mm = torch.tensor([[2.0, -3.0]])                  # start 2 mm right, 3 mm up of the dot
    centro = torch.tensor([[300.5 / W, 200.5 / H]]) + errore_mm / (torch.tensor([[W, H]]) * ps)
    crop = ritaglia_originali([img], centro, torch.tensor([0]), ps, finestra_mm=12.8, crop_px=64)
    assert crop.shape == (1, 1, 64, 64)
    yy, xx = torch.nonzero(crop[0, 0] > crop[0, 0].mean() + 2 * crop[0, 0].std(), as_tuple=True)
    pos_mm = (torch.stack([xx.float().mean(), yy.float().mean()]) + 0.5) / 64 * 12.8 - 6.4
    assert torch.allclose(pos_mm, -errore_mm[0], atol=0.25), pos_mm


def test_stage2_immagine_forma_guardia_e_selezione():
    torch.manual_seed(0)
    net = ImageRefinementNet(num_landmarks=4, finestra_mm=12.8, crop_px=64, max_offset_mm=2.0).eval()
    off, logits = net(torch.randn(5, 1, 64, 64), torch.tensor([0, 1, 2, 3, 0]))
    assert off.shape == (5, 2) and logits.shape == (5, 16, 16)
    assert off.abs().max() <= 6.4                               # inside the crop
    H, W = 300, 400
    originali = [torch.rand(1, H, W), torch.rand(1, H, W)]
    coords = torch.rand(2, 4, 2) * 0.6 + 0.2
    dims, ps = torch.tensor([[W, H], [W, H]], dtype=torch.float32), torch.tensor([0.1, 0.1])
    sel = torch.tensor([[True, False, False, True], [False, False, False, False]])
    out = applica_stage2(net, coords, originali=originali, dims=dims, pixel_size=ps, seleziona=sel)
    cambiati = (out != coords).any(dim=2)
    assert not cambiati[~sel].any()                             # only selected points can move
    spost_mm = (out - coords) * dims.view(-1, 1, 2) * ps.view(-1, 1, 1)
    assert spost_mm.abs().max() <= 1.2 * 2.0 + 1e-5             # guard
    # selection by uncertainty
    cov = torch.zeros(2, 4, 2, 2); cov[0, 2] = torch.eye(2) * 1e-4   # sigma = sqrt(2e-4) * 40 mm = 0.57 mm
    s = seleziona_punti("incertezza", 2, 4, "cpu", cov_tta=cov, dims=dims, pixel_size=ps, soglia_mm=0.45)
    assert s.sum() == 1 and s[0, 2]


def test_stage2_immagine_impara_a_ritrovare_il_punto():
    """On synthetic radiographs (a landmark = a bright square), Stage 2 trained
    with the train_refine.py sample generator learns to move a perturbed
    point back onto the landmark: the error must drop by more than half."""
    import train_refine as tr
    torch.manual_seed(0); random.seed(0)
    cfg = dict(tr.CONFIG, finestra_mm=12.8, crop_px=64, max_offset_mm=3.0, rumore_min_mm=0.5,
               rumore_max_mm=3.0, rotazione_max_gradi=10.0, peso_nll=1.0)
    rng = np.random.default_rng(0)

    def lotto(n_img=6):
        imgs, pts = [], []
        for _ in range(n_img):
            p = [(rng.uniform(80, 320), rng.uniform(80, 220)) for _ in range(2)]
            im = _immagine_con_punti(300, 400, p, raggio=3) * 0.8 + torch.rand(1, 300, 400) * 0.2
            imgs.append(im); pts.append(torch.tensor(p, dtype=torch.float32) + 0.5)
        return imgs, torch.stack(pts), torch.ones(n_img, 2), torch.full((n_img,), 0.1)

    net = ImageRefinementNet(num_landmarks=2, finestra_mm=12.8, crop_px=64, max_offset_mm=3.0)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    net.train()
    for _ in range(120):
        crops, l_idx, target = tr.genera_batch(*lotto(), cfg, "cpu", training=True)
        loss, _ = tr.calcola_loss(net, crops, l_idx, target, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        crops, l_idx, target = tr.genera_batch(*lotto(20), cfg, "cpu", training=False,
                                               generator=torch.Generator().manual_seed(1))
        pred, _ = net(crops, l_idx)
    prima = target.norm(dim=1).mean().item()
    dopo = (target - pred).norm(dim=1).mean().item()
    assert dopo < 0.5 * prima, (prima, dopo)


def test_stage2_tta_riporta_le_risposte_nel_ritaglio_originale():
    """With a 'perfect' Stage 2 (soft-argmax of the brightest spot), shifted
    and rotated crops must give back the same offset as the plain crop."""
    from src.inferenza import stage2_predizioni

    class Perfetto(torch.nn.Module):
        finestra_mm, crop_px, tipo_input = 12.8, 64, "immagine"

        def forward(self, crop, l_idx):
            # softmax(log(intensity)) = intensity-weighted centroid: exact and linear
            logits = torch.log(F.avg_pool2d(crop.clamp(min=0), 4)[:, 0] + 1e-6)   # [K, 16, 16]
            K, s, _ = logits.shape
            p = torch.softmax(logits.reshape(K, -1), 1).view(K, s, s)
            g = ((torch.arange(s, dtype=torch.float32) + 0.5) / s - 0.5) * self.finestra_mm
            return torch.stack([(p.sum(1) * g).sum(1), (p.sum(2) * g).sum(1)], 1), logits

    H, W, ps = 400, 500, torch.tensor([0.1])
    img = _immagine_con_punti(H, W, [(250, 200)], raggio=6)
    centri = torch.tensor([[250.5 / W, 200.5 / H], [(250.5 + 20) / W, (200.5 - 15) / H]])
    out = stage2_predizioni(Perfetto(), centri, torch.tensor([0, 0]), torch.tensor([0, 0]), [img], ps,
                            tta=True, spostamento_mm=0.4, angoli=(5.0, -5.0))
    atteso = torch.tensor([[0.0, 0.0], [-2.0, 1.5]])               # mm from each center to the spot
    assert torch.allclose(out["offset"], atteso, atol=0.3), out["offset"]
    assert torch.allclose(out["offset_tta"], out["offset"], atol=0.15), (out["offset_tta"], out["offset"])
    assert (out["var_tta"] < 0.05).all() and (out["var_hm"] > 0).all()


def test_fusione_stage1_stage2():
    """The fused point lies between Stage 1 and Stage 2, closer to the more
    reliable one; with an uncertain Stage 1 it is (almost) Stage 2."""
    from src.inferenza import peso_fusione
    s = torch.tensor([0.1, 0.5, 2.0])
    w = peso_fusione(s, torch.full((3,), 1.0), c=0.32, p=2.0)
    assert ((w > 0) & (w < 1)).all() and (w[1:] > w[:-1]).all() and w[2] > 0.95
    torch.manual_seed(0)
    net = ImageRefinementNet(num_landmarks=3, finestra_mm=12.8, crop_px=64, max_offset_mm=3.0).eval()
    H, W = 300, 400
    orig = [torch.rand(1, H, W)]
    coords = torch.rand(1, 3, 2) * 0.6 + 0.2
    dims, ps = torch.tensor([[W, H]], dtype=torch.float32), torch.tensor([0.1])
    pieno = applica_stage2(net, coords, originali=orig, dims=dims, pixel_size=ps)
    fuso = applica_stage2(net, coords, originali=orig, dims=dims, pixel_size=ps,
                          fusione={"c": 0.32, "p": 2.0, "floor_mm": 0.0}, sigma1_mm=torch.tensor([[0.01, 0.3, 50.0]]))
    spost_pieno, spost_fuso = (pieno - coords).norm(dim=2), (fuso - coords).norm(dim=2)
    assert (spost_fuso <= spost_pieno + 1e-7).all()
    # direction preserved, and with a hugely uncertain Stage 1 the fused point is Stage 2
    assert torch.allclose(fuso[0, 2], pieno[0, 2], atol=1e-5)
    assert spost_fuso[0, 0] < 0.05 * spost_pieno[0, 0] + 1e-7


def test_ordine_aariz_e_impronta_validazione():
    """The Aariz file list does not depend on the order the filesystem returns
    the files in, and a checkpoint trained on another validation set is
    reported when loaded together with this split."""
    import io, contextlib
    from src.dataset import AarizDataset
    from src import datasets as ds
    with tempfile.TemporaryDirectory() as base:
        img_dir = os.path.join(base, "train", "Cephalograms")
        ann_dir = os.path.join(base, "train", "Annotations", "Cephalometric Landmarks", "Senior Orthodontists")
        os.makedirs(img_dir); os.makedirs(ann_dir)
        for nome in ["c3", "a1", "b2", "e5", "d4"]:          # created out of order
            open(os.path.join(img_dir, nome + ".png"), "w").close()
            open(os.path.join(ann_dir, nome + ".json"), "w").close()
        a = AarizDataset(base, split="train")
    assert [s[0] for s in a.samples] == ["a1", "b2", "c3", "d4", "e5"]

    class Finto:
        def __init__(self, nomi):
            self.aariz = type("A", (), {"samples": [(n, "train", ".png") for n in nomi]})()
            self.isbi = type("I", (), {"samples": []})()
            self.len_aariz, self.len_isbi, self.len_tracciati = len(nomi), 0, 0
        def __len__(self):
            return self.len_aariz
    nomi = [f"img{k:03d}" for k in range(50)]
    ds.crea_split(Finto(nomi), 0.2, 42, modo="casuale"); imp1 = ds.impronta_split()
    ds.crea_split(Finto(nomi[::-1]), 0.2, 42, modo="casuale"); imp2 = ds.impronta_split()
    assert imp1 != imp2
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ds.registra_impronta_modello("Stage 1", imp1)
    assert "NOT valid" in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ds.registra_impronta_modello("Stage 1", imp2)
    assert "NOT valid" not in out.getvalue()
    ds._IMPRONTE["modelli"].clear()


def test_split_ufficiale_e_insiemi_di_test():
    """Official split: Aariz train/valid/test folders, ISBI TrainingData with
    every 5th image in validation, Test1/Test2 as test sets; Tracciati in
    training; the sets do not overlap and cover every image."""
    from src import datasets as ds

    class Finto:
        def __init__(self):
            self.aariz = type("A", (), {"samples": [(f"a{k}", sp, ".png") for k, sp in
                                                    enumerate(["train"] * 7 + ["valid"] * 2 + ["test"] * 3)]})()
            self.isbi = type("I", (), {"samples": [(n, f"{n:03d}.bmp") for n in range(1, 401)]})()
            self.len_aariz, self.len_isbi, self.len_tracciati = 12, 400, 4
        def __len__(self):
            return 416
    d = Finto()
    tr, va = ds.crea_split(d, modo="ufficiale")
    test = {n: ds.crea_insieme(d, n, modo="ufficiale").indices for n in ("test_aariz", "test_isbi1", "test_isbi2")}
    assert len(va.indices) == 2 + 30 and len(test["test_aariz"]) == 3
    assert len(test["test_isbi1"]) == 150 and len(test["test_isbi2"]) == 100
    assert set(range(412, 416)) <= set(tr.indices)                    # Tracciati in training
    tutti = [tr.indices, va.indices] + list(test.values())
    assert sum(len(x) for x in tutti) == 416 and len(set().union(*map(set, tutti))) == 416
    assert len(ds.crea_insieme(d, "test", modo="ufficiale").indices) == 253
    ds._IMPRONTE["modelli"].clear()


def test_unife_coordinate_e_numeri():
    """UNIFE CSV: names or row numbers (same order), ';' or ',', ImageJ numbers
    also with dots as thousands separators; only the mapped landmarks are kept."""
    from src.dataset_unife import leggi_coordinate, numero, ORDINE_UNIFE, MAPPA_UNIFE
    assert numero("1.155.000") == 1155.0 and numero("659.500") == 659.5 and numero("3,5") == 3.5
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
        with open(a, "w") as f:
            f.write(";X;Y\nS;97.304;55.539\nNa;161.683;31.846\n")
        with open(b, "w") as f:
            f.write(" ,X,Y\n" + "".join(f"{k + 1},{k}.5,{2 * k}.25\n" for k in range(35)))
        pa, pb = leggi_coordinate(a), leggi_coordinate(b)
    assert pa == {"S": (97.304, 55.539), "Na": (161.683, 31.846)}
    assert pb["S"] == (1.5, 2.25) and pb["Me'"] == (34.5, 68.25) and len(pb) == 35
    assert set(MAPPA_UNIFE) <= set(ORDINE_UNIFE) and len(set(MAPPA_UNIFE.values())) == 22


def test_crea_split_stabile_con_o_senza_tracciati():
    from torch.utils.data import random_split
    from src.datasets import crea_split

    class Finto:
        def __init__(self, a, i, t):
            self.len_aariz, self.len_isbi, self.n = a, i, a + i + t
        def __len__(self):
            return self.n
        def __getitem__(self, k):
            return k

    senza, con = Finto(100, 40, 0), Finto(100, 40, 25)
    tr0, va0 = crea_split(senza, 0.2, 42, modo="casuale")
    tr1, va1 = crea_split(con, 0.2, 42, modo="casuale")
    assert va0.indices == va1.indices
    assert set(range(140, 165)) <= set(tr1.indices) and not set(va1.indices) & set(range(140, 165))
    # identical to the old random_split when there are no Tracciati
    vecchio_tr, vecchio_va = random_split(senza, [112, 28], generator=torch.Generator().manual_seed(42))
    assert list(vecchio_va.indices) == va0.indices and list(vecchio_tr.indices) == tr0.indices


def test_tracciati_scala_per_caso():
    import openpyxl
    from src.dataset_tracciati import TracciatiDataset, leggi_scala_docx
    import cv2
    with tempfile.TemporaryDirectory() as d:
        for cid, docx_scala in (("334", 11.6), ("500", None)):
            case = os.path.join(d, "tracciato 50", cid)
            os.makedirs(case)
            cv2.imwrite(os.path.join(case, f"{cid}.jpg"), np.zeros((2000, 2400), np.uint8))
            wb = openpyxl.Workbook(); ws = wb.active
            ws.append([None, "X", "Y"]); ws.append(["S", 43678, 36684]); ws.append(["Na", 120000, 30000])
            wb.save(os.path.join(case, f"{cid}-Results.xlsx"))
            if docx_scala:
                xml = (f"<w:document><w:body><w:t>Distance in pixels: 11</w:t><w:t>6.0 0</w:t>"
                       f"<w:t> Known distance: 10 Pixel aspect ratio: 1.0</w:t></w:body></w:document>")
                with zipfile.ZipFile(os.path.join(case, f"{cid} Distance in pixels.docx"), "w") as z:
                    z.writestr("word/document.xml", xml)
        assert abs(leggi_scala_docx(os.path.join(d, "tracciato 50", "334", "334 Distance in pixels.docx")) - 11.6) < 1e-6
        ds = TracciatiDataset(base_path=d, subfolder_names=["tracciato 50"], coord_scale=0.01,
                              tabella_scale={"500": 5.0})
        assert len(ds) == 2
        _, pts, ps = ds[0]                                   # 334: scale from the .docx
        assert torch.allclose(pts[10], torch.tensor([43.678 * 11.6, 36.684 * 11.6]), atol=1e-2)
        assert torch.allclose(pts[4], torch.tensor([120.0 * 11.6, 30.0 * 11.6]), atol=1e-2)
        assert abs(ps.item() - 1 / 11.6) < 1e-6
        _, pts, ps = ds[1]                                   # 500: scale from the table
        assert torch.allclose(pts[10], torch.tensor([43.678 * 5.0, 36.684 * 5.0]), atol=1e-2)
        assert abs(ps.item() - 0.2) < 1e-6


def test_collate_base_conserva_dimensioni():
    batch = [(torch.zeros(1, 100, 200), torch.tensor([[100.0, 50.0]]), torch.ones(1), torch.zeros(4), torch.tensor(0.1))]
    imgs, pts, masks, _, ps, dims = collate_base(batch, img_size=64)
    assert imgs.shape == (1, 1, 64, 64)
    assert torch.allclose(pts, torch.tensor([[[0.5, 0.5]]])) and dims.tolist() == [[200.0, 100.0]]


def test_analisi_incertezza_copertura_conforme():
    rng = np.random.default_rng(0)
    n_img, N = 400, 3
    rec = {"img": [], "lm": [], "err": [], "sigma_heatmap": [], "sigma_tta": [], "sigma_totale": []}
    for im in range(n_img):
        for i in range(N):
            s = rng.uniform(0.5, 3.0)
            e = np.hypot(*rng.normal(0, s / math.sqrt(2), 2))
            rec["img"].append(im); rec["lm"].append(i); rec["err"].append(e)
            rec["sigma_heatmap"].append(s); rec["sigma_tta"].append(rng.uniform(0.5, 3.0)); rec["sigma_totale"].append(s)
    r = analisi_incertezza(rec, N, copertura=0.9)
    assert abs(r["fonti"]["heatmap"]["copertura_test"] - 0.9) < 0.04
    assert r["fonti"]["heatmap"]["spearman"] > 0.3
    assert abs(r["fonti"]["tta"]["spearman"]) < 0.1          # unrelated source: no correlation


if __name__ == "__main__":
    test = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    falliti = 0
    for nome, fn in test:
        try:
            fn()
            print(f"OK    {nome}")
        except Exception as e:
            falliti += 1
            print(f"FAIL  {nome}: {type(e).__name__}: {e}")
    print(f"\n{len(test) - falliti}/{len(test)} passed")
    sys.exit(1 if falliti else 0)
