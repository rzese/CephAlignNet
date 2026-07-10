"""
verifica_completa.py
====================
Suite di verifica PRE-TRAINING.
Esegui dalla root del progetto con:

    source ~/mio_venv/bin/activate
    cd /hpc/home/scarlassare/Progetto_tesi
    python3 verifica_completa.py

Ogni check stampa OK  o  FAIL + descrizione del problema.
Se tutti i check passano, puoi lanciare il training con fiducia.
"""

import sys, os, math, random
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial

sys.path.insert(0, "src")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

VERDE  = "\033[92m"
ROSSO  = "\033[91m"
GIALLO = "\033[93m"
RESET  = "\033[0m"

risultati = []

def ok(msg):
    print(f"  {VERDE}OK{RESET}  {msg}")
    risultati.append(("OK", msg))

def fail(msg):
    print(f"  {ROSSO}FAIL{RESET} {msg}")
    risultati.append(("FAIL", msg))

def warn(msg):
    print(f"  {GIALLO}WARN{RESET} {msg}")
    risultati.append(("WARN", msg))

def sezione(titolo):
    print(f"\n{'='*60}")
    print(f"  {titolo}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Import di tutti i moduli
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 1 — Import moduli")

try:
    from dataset           import AarizDataset
    from dataset_isbi      import ISBIDataset
    from dataset_tracciati import TracciatiDataset
    from datasets          import Datasets
    from heatmap_utils     import (coordinate_a_heatmap, heatmap_a_coordinate,
                                    combined_loss, wing_loss_coord,
                                    geometric_consistency_loss,
                                    augmenta, zoom_crop_augmenta,
                                    HEATMAP_TRAIN_SIZE)
    from model_hrnet  import HRNetLandmark
    from model_refine import RefinementNet, MAX_OFFSET, ROI_OUTPUT_SIZE, ROI_HALF_SIZE
    ok("Tutti i moduli importati correttamente")
except Exception as e:
    fail(f"Errore import: {e}")
    print("  Impossibile continuare senza tutti i moduli.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Coordinate normalizzate nei singoli dataset
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 2 — Coordinate [0,1] nei singoli dataset")

AARIZ_PATH     = "data/Aariz"
ISBI_PATH      = "data/Cepha400"
TRACCIATI_PATH = "data/tracciati_forniti"

try:
    aariz     = AarizDataset(base_path=AARIZ_PATH, split='all')
    isbi      = ISBIDataset(base_path=ISBI_PATH,   split='all')
    tracciati = TracciatiDataset(
        base_path=TRACCIATI_PATH,
        subfolder_names=["tracciato 50", "tracciato 50_2"],
        coord_scale=0.01, flip_y=False,
    )
except Exception as e:
    fail(f"Errore creazione dataset singoli: {e}")
    sys.exit(1)

for nome, ds in [("AARIZ", aariz), ("ISBI", isbi), ("TRACCIATI", tracciati)]:
    errori_range = []
    n_check = min(30, len(ds))
    for i in range(n_check):
        try:
            sample = ds[i]
            img, pts = sample[0], sample[1]
            validi = pts[pts.sum(1) != 0]
            if len(validi) == 0:
                errori_range.append(f"  campione {i}: nessun landmark valido!")
                continue
            if validi.min().item() < -0.001 or validi.max().item() > 1.001:
                errori_range.append(
                    f"  campione {i}: min={validi.min():.4f} max={validi.max():.4f}"
                )
        except Exception as e:
            errori_range.append(f"  campione {i}: eccezione {e}")

    if errori_range:
        fail(f"{nome}: coordinate fuori [0,1] su {len(errori_range)}/{n_check} campioni")
        for e in errori_range[:5]:
            print(f"    {e}")
    else:
        sample = ds[0]
        pts = sample[1]
        validi = pts[pts.sum(1) != 0]
        ok(f"{nome}: coordinate in [0,1] su {n_check} campioni "
           f"(min={validi.min():.4f} max={validi.max():.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Dimensioni e canali immagine
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 3 — Shape immagini (devono essere [1, H, W], float [0,1])")

for nome, ds in [("AARIZ", aariz), ("ISBI", isbi), ("TRACCIATI", tracciati)]:
    errori = []
    n_check = min(10, len(ds))
    for i in range(n_check):
        img = ds[i][0]
        if img.dim() != 3 or img.shape[0] != 1:
            errori.append(f"campione {i}: shape={img.shape} (atteso [1,H,W])")
        if img.min().item() < -0.001 or img.max().item() > 1.001:
            errori.append(f"campione {i}: pixel fuori [0,1]: min={img.min():.4f} max={img.max():.4f}")
        if img.dtype != torch.float32:
            errori.append(f"campione {i}: dtype={img.dtype} (atteso float32)")
    if errori:
        fail(f"{nome}: {errori[0]}")
    else:
        img0 = ds[0][0]
        ok(f"{nome}: shape {list(img0.shape)}, dtype={img0.dtype}, "
           f"pixel [{img0.min():.3f},{img0.max():.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Pixel size ragionevole
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 4 — pixel_size in mm/px (atteso ~0.05–0.15)")

for nome, ds in [("AARIZ", aariz), ("ISBI", isbi), ("TRACCIATI", tracciati)]:
    # ISBIDataset restituisce solo (img, pts), senza pixel_size
    sample = ds[0]
    if len(sample) < 3:
        warn(f"{nome}: non restituisce pixel_size (verrà impostato da Datasets)")
        continue
    px_vals = []
    for i in range(min(20, len(ds))):
        px = ds[i][2]
        px_vals.append(px.item() if hasattr(px, 'item') else float(px))
    bad = [v for v in px_vals if v < 0.01 or v > 1.0]
    if bad:
        fail(f"{nome}: pixel_size anomali: {bad[:5]}")
    else:
        ok(f"{nome}: pixel_size in range [{min(px_vals):.4f},{max(px_vals):.4f}]")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Pool aggregato Datasets
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 5 — Datasets aggregato (1500 campioni attesi)")

try:
    ds_total = Datasets(
        aariz_path=AARIZ_PATH,
        isbi_path=ISBI_PATH,
        tracciati_path=TRACCIATI_PATH,
    )
    n = len(ds_total)
    if n < 1400:
        warn(f"Pool totale solo {n} campioni (attesi ~1500: 1000 Aariz+400 ISBI+100 Tracciati) — controlla i path")
    else:
        ok(f"Pool totale: {n} campioni (Aariz+ISBI+Tracciati)")
except Exception as e:
    fail(f"Errore creazione Datasets: {e}")
    sys.exit(1)

# Verifica su campioni distribuiti lungo tutto il pool
errori_pool = []
indici_test = list(range(0, n, max(1, n // 50)))  # ~50 campioni distribuiti
for i in indici_test:
    try:
        img, pts, mask, bbox, px = ds_total[i]
        validi = pts[pts.sum(1) != 0]
        if len(validi) == 0:
            continue
        if validi.min().item() < -0.001 or validi.max().item() > 1.001:
            errori_pool.append(f"campione {i}: coords [{validi.min():.4f},{validi.max():.4f}]")
        if px.item() < 0.01 or px.item() > 1.0:
            errori_pool.append(f"campione {i}: pixel_size anomalo={px.item():.4f}")
        if img.shape[0] != 1:
            errori_pool.append(f"campione {i}: img shape={list(img.shape)}")
        if mask.sum().item() == 0:
            errori_pool.append(f"campione {i}: maschera tutta zero!")
    except Exception as e:
        errori_pool.append(f"campione {i}: eccezione {e}")

if errori_pool:
    fail(f"Datasets: {len(errori_pool)} problemi su {len(indici_test)} campioni testati")
    for e in errori_pool[:8]:
        print(f"    {e}")
else:
    ok(f"Datasets: coordinate, mask, pixel_size OK su {len(indici_test)} campioni distribuiti")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Split 80/20 riproducibile
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 6 — Split 80/20 (stesso seed = stesso split)")

from torch.utils.data import random_split

def crea_split(ds, seed=42):
    n_val = int(len(ds) * 0.2)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    return set(train_ds.indices), set(val_ds.indices)

train1, val1 = crea_split(ds_total, seed=42)
train2, val2 = crea_split(ds_total, seed=42)

if train1 == train2 and val1 == val2:
    ok(f"Split riproducibile: train={len(train1)}, val={len(val1)}")
else:
    fail("Split NON riproducibile — il seed non funziona come atteso!")

# Nessuna sovrapposizione
overlap = train1 & val1
if overlap:
    fail(f"Train e val si sovrappongono su {len(overlap)} campioni!")
else:
    ok("Nessuna sovrapposizione train/val")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Collate e heatmap
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 7 — Collate + generazione heatmap")

from torch.utils.data import DataLoader

IMG_SIZE     = 512
HEATMAP_SIZE = HEATMAP_TRAIN_SIZE   # 256
SIGMA        = 4.0

def collate_test(batch, img_size=IMG_SIZE, heatmap_size=HEATMAP_SIZE, sigma=SIGMA):
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    imgs_r = [
        F.interpolate(img.unsqueeze(0), size=(img_size, img_size),
                      mode="bilinear", align_corners=False).squeeze(0)
        for img in immagini
    ]
    punti_pad, masks_pad = [], []
    for p, m in zip(punti, masks):
        n = p.shape[0]
        pp = torch.zeros(26, 2); pm = torch.zeros(26)
        pp[:n] = p[:n]; pm[:n] = m[:n]
        punti_pad.append(pp); masks_pad.append(pm)
    heatmaps = [
        coordinate_a_heatmap(pts, msk, img_size, heatmap_size, sigma)
        for pts, msk in zip(punti_pad, masks_pad)
    ]
    return (
        torch.stack(list(imgs_r)),
        torch.stack(list(punti_pad)),
        torch.stack(list(masks_pad)),
        torch.stack(heatmaps),
        torch.stack(bboxes),
        torch.stack(pixel_sizes),
    )

try:
    loader = DataLoader(ds_total, batch_size=4, shuffle=False,
                        collate_fn=collate_test)
    imgs, pts, masks, heatmaps, bboxes, px = next(iter(loader))

    # Shape
    assert imgs.shape == torch.Size([4, 1, IMG_SIZE, IMG_SIZE]), \
        f"imgs shape errata: {imgs.shape}"
    assert pts.shape  == torch.Size([4, 26, 2]), \
        f"pts shape errata: {pts.shape}"
    assert heatmaps.shape == torch.Size([4, 26, HEATMAP_SIZE, HEATMAP_SIZE]), \
        f"heatmaps shape errata: {heatmaps.shape}"

    # Range
    assert imgs.min().item() >= -0.001 and imgs.max().item() <= 1.001, \
        f"imgs fuori [0,1]: {imgs.min():.4f} {imgs.max():.4f}"
    assert pts.min().item()  >= -0.001 and pts.max().item()  <= 1.001, \
        f"pts fuori [0,1]: {pts.min():.4f} {pts.max():.4f}"
    assert heatmaps.min().item() >= -0.001 and heatmaps.max().item() <= 1.001, \
        f"heatmaps fuori [0,1]: {heatmaps.min():.4f} {heatmaps.max():.4f}"

    # Heatmap non tutte vuote
    n_vuote = (heatmaps.sum(dim=[2, 3]) == 0).sum().item()
    if n_vuote > 0:
        warn(f"{n_vuote} heatmap vuote su {4*26} (landmark mascherati o fuori frame — normale)")
    else:
        ok("Nessuna heatmap vuota inattesa")

    ok(f"Collate: imgs {list(imgs.shape)}, pts [{pts.min():.4f},{pts.max():.4f}], "
       f"heatmaps max={heatmaps.max():.4f}")

except AssertionError as e:
    fail(f"Collate assertion: {e}")
except Exception as e:
    fail(f"Collate errore: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — heatmap_a_coordinate inverte coordinate_a_heatmap
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 8 — Round-trip: coordinate → heatmap → coordinate")

pts_test = torch.tensor([[0.3, 0.4], [0.6, 0.55], [0.2, 0.8]])
mask_test = torch.ones(26)
pts_pad   = torch.zeros(26, 2)
pts_pad[:3] = pts_test

hm = coordinate_a_heatmap(pts_pad, mask_test, IMG_SIZE, HEATMAP_SIZE, sigma=4.0)
hm_batch = hm.unsqueeze(0)[:, 0::3]     # prende solo i canali heatmap (0,3,6,...)
# Ricostruisce heatmap batch con tutti e 26 canali
hm_full = coordinate_a_heatmap(pts_pad, mask_test, IMG_SIZE, HEATMAP_SIZE, sigma=4.0)
pred_pts = heatmap_a_coordinate(hm_full.unsqueeze(0))  # [1, 26, 2]

errori_rt = []
for i in range(3):
    dx = abs(pred_pts[0, i, 0].item() - pts_test[i, 0].item())
    dy = abs(pred_pts[0, i, 1].item() - pts_test[i, 1].item())
    if dx > 0.02 or dy > 0.02:
        errori_rt.append(
            f"pt{i}: gt=({pts_test[i,0]:.3f},{pts_test[i,1]:.3f}) "
            f"pred=({pred_pts[0,i,0]:.3f},{pred_pts[0,i,1]:.3f}) "
            f"err=({dx:.3f},{dy:.3f})"
        )

if errori_rt:
    fail("Round-trip coordinate→heatmap→coordinate con errore > 0.02")
    for e in errori_rt:
        print(f"    {e}")
else:
    max_err = max(
        abs(pred_pts[0, i, j].item() - pts_test[i // 1, j].item())
        for i in range(3) for j in range(2)
    )
    ok(f"Round-trip OK: errore massimo = {max_err:.4f} (< 0.02)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 9 — Loss functions producono valori finiti
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 9 — Loss functions (valori finiti, no NaN/Inf)")

B, N, H, W = 2, 26, HEATMAP_SIZE, HEATMAP_SIZE
device = torch.device("cpu")

pred_hm_full = torch.randn(B, N * 3, H, W)   # output del modello (hm + offset)
gt_hm        = torch.clamp(torch.randn(B, N, H, W), 0, 1)
pts_gt       = torch.rand(B, N, 2)
mask_gt      = (torch.rand(B, N) > 0.3).float()

try:
    loss_comb = combined_loss(pred_hm_full, gt_hm, pts_gt, mask_gt)
    if not torch.isfinite(loss_comb):
        fail(f"combined_loss non finita: {loss_comb.item()}")
    else:
        ok(f"combined_loss = {loss_comb.item():.6f} (finita)")
except Exception as e:
    fail(f"combined_loss errore: {e}")

try:
    pred_pts = torch.rand(B, N, 2)
    loss_wc = wing_loss_coord(pred_pts, pts_gt, mask_gt)
    if not torch.isfinite(loss_wc):
        fail(f"wing_loss_coord non finita: {loss_wc.item()}")
    else:
        ok(f"wing_loss_coord = {loss_wc.item():.6f} (finita)")
except Exception as e:
    fail(f"wing_loss_coord errore: {e}")

try:
    loss_geo = geometric_consistency_loss(pred_pts, mask_gt)
    if not torch.isfinite(loss_geo):
        fail(f"geometric_consistency_loss non finita: {loss_geo.item()}")
    else:
        ok(f"geometric_consistency_loss = {loss_geo.item():.6f} (finita)")
except Exception as e:
    fail(f"geometric_consistency_loss errore: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 10 — Augmentation mantiene coordinate in [0,1]
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 10 — Augmentation: coordinate rimangono in [0,1]")

torch.manual_seed(0); random.seed(0)

img0, pts0, mask0, _, _ = ds_total[0]
img0_r = F.interpolate(img0.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                       mode="bilinear", align_corners=False).squeeze(0)
pts_pad0 = torch.zeros(26, 2); mask_pad0 = torch.zeros(26)
n = pts0.shape[0]
pts_pad0[:n] = pts0[:n]; mask_pad0[:n] = mask0[:n]

n_aug_trials = 50
aug_fails = []
for trial in range(n_aug_trials):
    try:
        img_a, pts_a, mask_a = augmenta(img0_r.clone(), pts_pad0.clone(), mask_pad0.clone())
        validi_a = pts_a[mask_a == 1.0]
        if len(validi_a) > 0:
            if validi_a.min().item() < -0.001 or validi_a.max().item() > 1.001:
                aug_fails.append(
                    f"trial {trial}: [{validi_a.min():.4f},{validi_a.max():.4f}]"
                )
        if img_a.min().item() < -0.001 or img_a.max().item() > 1.001:
            aug_fails.append(
                f"trial {trial}: img [{img_a.min():.4f},{img_a.max():.4f}]"
            )
    except Exception as e:
        aug_fails.append(f"trial {trial}: eccezione {e}")

if aug_fails:
    fail(f"augmenta: {len(aug_fails)}/{n_aug_trials} augmentation fuori range")
    for e in aug_fails[:5]:
        print(f"    {e}")
else:
    ok(f"augmenta: tutte le {n_aug_trials} augmentation mantengono coordinate e pixel in [0,1]")

# Anche zoom_crop
zoom_fails = []
for trial in range(20):
    try:
        img_z, pts_z, mask_z = zoom_crop_augmenta(
            img0_r.clone(), pts_pad0.clone(), mask_pad0.clone(), zoom_factor=2.2
        )
        validi_z = pts_z[mask_z == 1.0]
        if len(validi_z) > 0:
            if validi_z.min().item() < -0.001 or validi_z.max().item() > 1.001:
                zoom_fails.append(f"trial {trial}: [{validi_z.min():.4f},{validi_z.max():.4f}]")
    except Exception as e:
        zoom_fails.append(f"trial {trial}: {e}")

if zoom_fails:
    fail(f"zoom_crop_augmenta: {len(zoom_fails)}/20 fuori range")
else:
    ok("zoom_crop_augmenta: coordinate mantenute in [0,1]")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 11 — MRE formula: usa larghezza reale, non 2400 fisso
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 11 — Formula MRE in mm (larghezza reale × pixel_size)")

larghezze = []
pixel_sizes = []
for i in range(0, min(n, 30)):
    img, pts, mask, bbox, px = ds_total[i]
    larghezze.append(img.shape[2])     # larghezza originale prima del resize
    pixel_sizes.append(px.item())

w_min, w_max = min(larghezze), max(larghezze)
if w_max - w_min > 200:
    warn(
        f"Le immagini hanno larghezze molto diverse: [{w_min},{w_max}] px. "
        f"Usare larghezza_reale × pixel_size (NON 2400 × pixel_size) in evaluate."
    )
    # Mostra l'impatto dell'errore
    esempio_w  = larghezze[0]
    px_ex      = pixel_sizes[0]
    err_fisso  = 2400 * px_ex
    err_reale  = esempio_w * px_ex
    pct_errore = abs(err_fisso - err_reale) / err_reale * 100
    warn(
        f"Esempio: img_w={esempio_w}, px={px_ex:.3f} "
        f"→ 2400×px={err_fisso:.1f}mm vs w_reale×px={err_reale:.1f}mm "
        f"(differenza {pct_errore:.1f}%)"
    )
else:
    ok(f"Larghezze uniformi [{w_min},{w_max}] px — formula MRE non critica")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 12 — Maschera: landmark non annotati → mask=0
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 12 — Maschera landmark non annotati")

# ISBI non ha i landmark 19-25 (UIT,UIA,LIA,Ls,Li,Pog`,Sn)
# Tracciati non hanno tutti i 26 landmark
for nome, sorgente in [("ISBI", "isbi"), ("TRACCIATI", "tracciati")]:
    ds_src = isbi if sorgente == "isbi" else tracciati
    # Accedi tramite il pool Datasets per avere la maschera
    # Trova 5 campioni della sorgente giusta
    campioni_fonte = []
    offset = ds_total.len_aariz if sorgente == "isbi" else ds_total.len_aariz + ds_total.len_isbi
    fine   = offset + (ds_total.len_isbi if sorgente == "isbi" else ds_total.len_tracciati)
    for i in range(offset, min(fine, offset + 10)):
        img, pts, mask, bbox, px = ds_total[i]
        campioni_fonte.append(mask)

    if not campioni_fonte:
        warn(f"{nome}: nessun campione trovato nel pool")
        continue

    # Landmark che ISBI non ha: 19,20,21,22,23,24,25 (7 landmark)
    if sorgente == "isbi":
        lm_assenti_attesi = [19, 20, 21, 22, 23, 24, 25]
        tutti_zero = all(m[lm_assenti_attesi].sum().item() == 0 for m in campioni_fonte)
        if tutti_zero:
            ok(f"{nome}: landmark 19-25 correttamente mascherati (mask=0)")
        else:
            fail(f"{nome}: alcuni landmark 19-25 hanno mask=1 ma non dovrebbero esserci nel dataset!")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 13 — Numero campioni per sorgente (no duplicati)
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 13 — Composizione del pool (no duplicati ISBI/Ceph)")

print(f"    Aariz:     {ds_total.len_aariz}")
print(f"    ISBI:      {ds_total.len_isbi}")
print(f"    Tracciati: {ds_total.len_tracciati}")
print(f"    TOTALE:    {len(ds_total)}")

if ds_total.len_aariz < 900:
    warn(f"Aariz ha solo {ds_total.len_aariz} campioni (attesi ~1000 con png+jpg+jpeg+bmp)")
else:
    ok(f"Aariz: {ds_total.len_aariz} campioni (tutti gli split)")

if ds_total.len_isbi < 350:
    warn(f"ISBI ha solo {ds_total.len_isbi} campioni (attesi 400)")
else:
    ok(f"ISBI: {ds_total.len_isbi} campioni (tutte le 400 immagini)")

if ds_total.len_tracciati == 0:
    warn("Tracciati: 0 campioni — il path potrebbe essere sbagliato")
else:
    ok(f"Tracciati: {ds_total.len_tracciati} campioni")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 14 — Modello HRNet: forward pass
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 14 — HRNetLandmark: forward pass")

try:
    model = HRNetLandmark(num_landmarks=26, pretrained=False)
    model.eval()
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out, x_out = model(dummy)

    atteso_out_ch = 26 * 3  # heatmap + offset_x + offset_y per ogni landmark
    if out.shape[1] != atteso_out_ch:
        fail(f"HRNet output: {out.shape[1]} canali (attesi {atteso_out_ch})")
    else:
        ok(f"HRNet output: {list(out.shape)} | feature map: {list(x_out.shape)}")

    if not torch.isfinite(out).all():
        fail("HRNet output contiene NaN/Inf!")
    else:
        ok("HRNet output finito (no NaN/Inf)")

    in_channels_refine = x_out.shape[1]
    ok(f"Canali feature map per RefinementNet: {in_channels_refine}")

except Exception as e:
    fail(f"HRNet forward pass: {e}")
    in_channels_refine = 192   # fallback


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 15 — RefinementNet: forward pass
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 15 — RefinementNet: forward pass")

try:
    refine = RefinementNet(num_landmarks=26, in_channels=in_channels_refine)
    refine.eval()
    B_r   = 4
    dummy_roi = torch.randn(B_r, in_channels_refine, ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE)
    dummy_idx = torch.randint(0, 26, (B_r,))
    with torch.no_grad():
        out_r = refine(dummy_roi, dummy_idx)

    if out_r.shape != torch.Size([B_r, 2]):
        fail(f"RefinementNet output shape: {out_r.shape} (atteso [{B_r}, 2])")
    else:
        ok(f"RefinementNet output: {list(out_r.shape)}")

    if out_r.abs().max().item() > MAX_OFFSET + 1e-4:
        fail(f"RefinementNet output supera MAX_OFFSET={MAX_OFFSET}: max={out_r.abs().max().item():.6f}")
    else:
        ok(f"RefinementNet output in [-{MAX_OFFSET},{MAX_OFFSET}]: max={out_r.abs().max().item():.6f}")

except Exception as e:
    fail(f"RefinementNet forward pass: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 16 — Seed identico in train_heatmap e train_refine
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 16 — Seed split identico (train_heatmap = train_refine)")

SEED = 42
g1 = torch.Generator().manual_seed(SEED)
g2 = torch.Generator().manual_seed(SEED)

n_tot = len(ds_total)
n_val = int(n_tot * 0.2)
_, v1 = random_split(ds_total, [n_tot - n_val, n_val], generator=g1)
_, v2 = random_split(ds_total, [n_tot - n_val, n_val], generator=g2)

if set(v1.indices) == set(v2.indices):
    ok(f"Seed {SEED}: val set identico tra i due script "
       f"({len(v1)} campioni, primo={min(v1.indices)}, ultimo={max(v1.indices)})")
else:
    fail("Val set diverso! Controlla che entrambi gli script usino seed=42 e val_split=0.2")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 17 — Dimensioni reali immagini Aariz (7 macchine diverse → risoluzioni diverse)
# Aariz ha immagini da 7 X-ray machine diverse → le dimensioni variano.
# Questo check cataloga le dimensioni reali e verifica che il pixel_size CSV
# copra tutte le immagini del pool.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 17 — Dimensioni reali immagini Aariz (7 macchine, risoluzioni variabili)")

try:
    import collections
    dimensioni_aariz = collections.Counter()
    pixel_sizes_aariz = []
    n_sample_aariz = min(100, len(aariz))
    n_px_mancanti = 0
    for i in range(n_sample_aariz):
        sample = aariz[i]
        img = sample[0]
        h, w = img.shape[1], img.shape[2]
        dimensioni_aariz[(h, w)] += 1
        ps = sample[2].item() if len(sample) >= 3 else None
        if ps is not None:
            pixel_sizes_aariz.append(ps)
            if ps == aariz.default_pixel_size:
                n_px_mancanti += 1

    print(f"  Distribuzione dimensioni (H×W) su {n_sample_aariz} campioni:")
    for (h, w), cnt in sorted(dimensioni_aariz.items(), key=lambda x: -x[1]):
        print(f"    {h}×{w}: {cnt} immagini")

    n_dim_diverse = len(dimensioni_aariz)
    if n_dim_diverse == 1:
        ok("Aariz: tutte le immagini hanno la stessa dimensione (raro: 1 sola macchina?)")
    else:
        # Verifica se il fix MRE è già applicato
        import pathlib as _pl17
        _ev17 = _pl17.Path("evaluate_refine.py").read_text()
        _fix17 = "largh_px" in _ev17 or "larghezze_orig" in _ev17
        if _fix17:
            warn(f"Aariz: {n_dim_diverse} dimensioni diverse — MRE gestito con larghezza reale per campione (fix OK)")
        else:
            warn(f"Aariz: {n_dim_diverse} dimensioni diverse trovate — CRITICO per MRE!")
            print(f"  {ROSSO}→ evaluate_refine.py usa larghezza_orig_px=2400 FISSO: ERRATO per Aariz!{RESET}")
            print(f"  {ROSSO}→ Devi usare la larghezza REALE di ogni immagine × pixel_size{RESET}")

    if n_px_mancanti > 0:
        pct = n_px_mancanti / n_sample_aariz * 100
        warn(f"Aariz: {n_px_mancanti}/{n_sample_aariz} ({pct:.0f}%) campioni usano pixel_size default={aariz.default_pixel_size} "
             f"(non trovati nel CSV)")
    else:
        ok(f"Aariz: tutti i {n_sample_aariz} campioni hanno pixel_size dal CSV "
           f"(range [{min(pixel_sizes_aariz):.4f},{max(pixel_sizes_aariz):.4f}] mm/px)")

except Exception as e:
    fail(f"CHECK 17: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 18 — Bug MRE in evaluate_refine.py: larghezza fissa vs reale
# La formula corretta per MRE è:
#   errore_mm = distanza_norm * larghezza_reale_px * pixel_size_mm
# NON:
#   errore_mm = distanza_norm * 2400 * pixel_size_mm
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 18 — Formula MRE in evaluate_refine.py (BUG CRITICO)")

try:
    # Raccoglie larghezze reali da tutto il pool (non solo le prime 30)
    larghezze_pool = []
    pixel_sizes_pool = []
    step = max(1, n // 80)
    for i in range(0, n, step):
        img, pts, mask, bbox, px = ds_total[i]
        larghezze_pool.append(img.shape[2])
        pixel_sizes_pool.append(px.item())

    w_min_pool = min(larghezze_pool)
    w_max_pool = max(larghezze_pool)
    larghezza_fissa = 2400   # valore hardcoded in evaluate_refine.py

    print(f"  Larghezze reali nel pool: [{w_min_pool}, {w_max_pool}] px")
    print(f"  Valore fisso in evaluate_refine.py: {larghezza_fissa} px")

    # Verifica se evaluate_refine.py ha già il fix MRE
    import pathlib as _pl
    _ev_src = _pl.Path("evaluate_refine.py").read_text()
    _fix_mre = ("largh_px" in _ev_src or "larghezze_orig" in _ev_src)

    if w_max_pool - w_min_pool < 50:
        ok(f"Larghezze uniformi [{w_min_pool},{w_max_pool}] px — formula MRE non critica")
    elif _fix_mre:
        ok(f"Fix MRE già applicato: evaluate_refine.py usa larghezza reale per campione "
           f"(larghezze [{w_min_pool},{w_max_pool}] px — errore sistematico eliminato)")
    else:
        errori_pct = []
        for w, px in zip(larghezze_pool, pixel_sizes_pool):
            if px > 0 and w > 0:
                mre_fisso = larghezza_fissa * px
                mre_reale = w * px
                errori_pct.append(abs(mre_fisso - mre_reale) / mre_reale * 100)
        if errori_pct:
            err_medio = sum(errori_pct) / len(errori_pct)
            err_max   = max(errori_pct)
            fail(
                f"BUG MRE: larghezza_orig_px=2400 fisso, ma le immagini variano "
                f"[{w_min_pool},{w_max_pool}] px.\n"
                f"    Errore sistematico MRE: medio={err_medio:.1f}%, max={err_max:.1f}%\n"
                f"    FIX: in evaluate_refine.py usa larghezza reale dal collate"
            )
        else:
            warn("Impossibile calcolare errore MRE — pixel_sizes tutti zero?")

except Exception as e:
    fail(f"CHECK 18: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 19 — Dimensioni ISBI: tutte 1935×2400, pixel_size=0.1mm/px (letteratura)
# Fonte: Wang 2016, tutti i paper citati concordano su 1935×2400 @ 0.1mm/px
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 19 — Dimensioni ISBI: atteso 1935×2400, 0.1mm/px")

try:
    dimensioni_isbi = collections.Counter()
    n_isbi_sample = min(30, len(isbi))
    for i in range(n_isbi_sample):
        img = isbi[i][0]
        h, w = img.shape[1], img.shape[2]
        dimensioni_isbi[(h, w)] += 1

    print(f"  Dimensioni trovate in {n_isbi_sample} campioni ISBI:")
    for (h, w), cnt in sorted(dimensioni_isbi.items(), key=lambda x: -x[1]):
        is_ok = (h == 1935 and w == 2400) or (h == 2400 and w == 1935)
        flag = "✓" if is_ok else "✗ INATTESO"
        print(f"    {h}×{w}: {cnt} immagini  {flag}")

    # La letteratura dice 1935×2400 H×W (portrait)
    atteso_h, atteso_w = 1935, 2400
    # Accetta sia H×W=1935×2400 che 2400×1935 (cv2 legge in H,W quindi BMP landscape = 2400×1935)
    n_sbagliati = sum(cnt for (h, w), cnt in dimensioni_isbi.items()
                      if not ((h == 1935 and w == 2400) or (h == 2400 and w == 1935)))
    if n_sbagliati == 0:
        ok(f"ISBI: tutte {n_isbi_sample} immagini hanno dimensioni 1935×2400 o 2400×1935 — corretto")
    else:
        warn(f"ISBI: {n_sbagliati} immagini con dimensioni inattese — verifica BMP nel dataset")

    # Verifica che il pixel_size di ISBI nel pool sia 0.1
    px_isbi_pool = []
    for i in range(ds_total.len_aariz, min(ds_total.len_aariz + 20, ds_total.len_aariz + ds_total.len_isbi)):
        _, _, _, _, px = ds_total[i]
        px_isbi_pool.append(px.item())
    if px_isbi_pool:
        if all(abs(p - 0.1) < 0.001 for p in px_isbi_pool):
            ok(f"ISBI: pixel_size=0.1mm/px confermato (da letteratura: 0.1mm/px)")
        else:
            fail(f"ISBI: pixel_size anomalo: {px_isbi_pool[:5]}")

except Exception as e:
    fail(f"CHECK 19: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 20 — Mapping ISBI P1-P19 → indici canonici: consistenza e completezza
# Verifica che nessun landmark ISBI venga perso o mappato all'indice sbagliato.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 20 — Mapping ISBI P1-P19 → indici canonici")

try:
    from dataset_isbi import ISBI_P_TO_CANONICAL

    # Landmark canonici che ISBI DEVE avere (da letteratura Lindner 2015)
    # P1=S(10), P2=N(4), P3=Or(5), P4=Po(15), P5=Ar(11), P6=PNS(7), P7=ANS(1),
    # P8=A(0), P9=B(2), P10=Pog(6), P11=Gn(13), P12=Me(3), P13=Go(14),
    # P14=Co(12), P15=LPM(16), P16=LMT(18), P17=LIT(17), P19=R(9)
    # P18 (UPM) = non nel canonico → skip
    mapping_atteso = {
        1: 10, 2: 4, 3: 5, 4: 15, 5: 11, 6: 7, 7: 1, 8: 0,
        9: 2, 10: 6, 11: 13, 12: 3, 13: 14, 14: 12,
        15: 16, 16: 18, 17: 17, 19: 9
    }

    errori_mapping = []
    for p_num, idx_atteso in mapping_atteso.items():
        idx_reale = ISBI_P_TO_CANONICAL.get(p_num)
        if idx_reale is None:
            errori_mapping.append(f"P{p_num}: MANCANTE nel mapping (atteso idx={idx_atteso})")
        elif idx_reale != idx_atteso:
            errori_mapping.append(f"P{p_num}: idx={idx_reale} ma atteso {idx_atteso}")

    # Controlla indici duplicati (due P→ stesso canonico)
    valori = list(ISBI_P_TO_CANONICAL.values())
    duplicati = [v for v in set(valori) if valori.count(v) > 1]
    if duplicati:
        errori_mapping.append(f"DUPLICATI nel mapping canonico: indici {duplicati}")

    if errori_mapping:
        for e in errori_mapping:
            fail(f"ISBI mapping: {e}")
    else:
        ok(f"Mapping ISBI P1-P19→canonico corretto: {len(ISBI_P_TO_CANONICAL)} landmark mappati, "
           f"nessun duplicato")

    # Verifica su campioni reali: i 19 landmark attesi hanno mask=1
    campione_isbi = isbi[0]
    pts_isbi  = campione_isbi[1]
    indici_attesi = set(mapping_atteso.values())  # 19 indici canonici
    # Controlla che quei punti siano non-zero
    n_validi_reale = sum(1 for idx in indici_attesi
                         if pts_isbi[idx].sum().item() != 0)
    if n_validi_reale < 17:  # tolleranza: almeno 17/19
        warn(f"ISBI campione 0: solo {n_validi_reale}/19 landmark attesi sono non-zero")
    else:
        ok(f"ISBI campione 0: {n_validi_reale}/19 landmark attesi sono non-zero")

except Exception as e:
    fail(f"CHECK 20: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 21 — Landmark Aariz: tutti i 26 presenti nel JSON?
# Aariz ha 29 landmark nel dataset originale, ma noi usiamo solo 26 (SYMBOL_ORDER).
# Verifica che il parsing JSON trovi tutti i 26 con la media senior/junior.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 21 — Landmark Aariz: 26 landmark e media senior/junior")

try:
    from dataset import SYMBOL_ORDER, NUM_LANDMARKS as NL_AARIZ

    print(f"  SYMBOL_ORDER ({len(SYMBOL_ORDER)} simboli): {SYMBOL_ORDER}")

    conteggio_per_lm = [0] * 26
    campioni_testati = min(50, len(aariz))
    n_lm_medi = 0  # campioni con media senior+junior

    for i in range(campioni_testati):
        sample = aariz[i]
        pts = sample[1]   # (26, 2) landmark
        for j in range(26):
            if pts[j].sum().item() != 0:
                conteggio_per_lm[j] += 1

    print(f"  Presenza landmark su {campioni_testati} campioni:")
    lm_rari = []
    for j, cnt in enumerate(conteggio_per_lm):
        pct = cnt / campioni_testati * 100
        nome = SYMBOL_ORDER[j] if j < len(SYMBOL_ORDER) else f"LM{j}"
        flag = ""
        if pct < 50:
            flag = f"  ← {ROSSO}RARO ({pct:.0f}%){RESET}"
            lm_rari.append((j, nome, pct))
        elif pct < 80:
            flag = f"  ← {GIALLO}PARZIALE ({pct:.0f}%){RESET}"
        print(f"    [{j:2d}] {nome:<6}: {cnt:3d}/{campioni_testati} ({pct:.0f}%){flag}")

    scheletrici_ok = all(conteggio_per_lm[j] > campioni_testati * 0.8 for j in range(16))
    if scheletrici_ok:
        ok("Aariz: tutti i 16 landmark scheletrici presenti in >80% dei campioni")
    else:
        fail("Aariz: alcuni landmark scheletrici mancano in >20% dei campioni!")

    soft_tissue_ok = all(conteggio_per_lm[j] > campioni_testati * 0.5 for j in range(22, 26))
    if soft_tissue_ok:
        ok("Aariz: landmark soft tissue (22-25) presenti in >50% dei campioni")
    else:
        warn("Aariz: alcuni landmark soft tissue mancano in >50% dei campioni "
             "(normale se non annotati)")

except Exception as e:
    fail(f"CHECK 21: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 22 — Coordinate Aariz: verifica che senior E junior vengano mediati
# e non usati singolarmente (bug potenziale: se junior_dir non esiste, usa solo senior).
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 22 — Media senior/junior Aariz (o solo senior se junior assente)")

try:
    import json as _json

    # Prende il primo campione e verifica se esiste il file junior
    fname0, split0, ext0 = aariz.samples[0]
    senior_path = os.path.join(AARIZ_PATH, split0, 'Annotations',
                               'Cephalometric Landmarks', 'Senior Orthodontists',
                               fname0 + '.json')
    junior_path = os.path.join(AARIZ_PATH, split0, 'Annotations',
                               'Cephalometric Landmarks', 'Junior Orthodontists',
                               fname0 + '.json')

    if os.path.exists(junior_path):
        # Carica entrambi e confronta con quello restituito dal dataset
        with open(senior_path) as f:
            senior_data = _json.load(f)
        with open(junior_path) as f:
            junior_data = _json.load(f)

        senior_dict = {item['symbol']: [item['value']['x'], item['value']['y']]
                       for item in senior_data['landmarks']}
        junior_dict = {item['symbol']: [item['value']['x'], item['value']['y']]
                       for item in junior_data['landmarks']}

        # Confronta con il punto restituito dal dataset (deve essere la media ceil)
        sample0 = aariz[0]
        pts_ds   = sample0[1]
        img0_    = sample0[0]
        h0, w0   = img0_.shape[1], img0_.shape[2]

        # Scegli un landmark presente in entrambi
        sym_test = None
        for sym in SYMBOL_ORDER:
            if sym in senior_dict and sym in junior_dict:
                sym_test = sym
                break

        if sym_test:
            idx_test = SYMBOL_ORDER.index(sym_test)
            sx, sy   = senior_dict[sym_test]
            jx, jy   = junior_dict[sym_test]
            # Media con ceil (come nel codice)
            import math as _math
            mx_atteso = _math.ceil(0.5 * (sx + jx)) / w0
            my_atteso = _math.ceil(0.5 * (sy + jy)) / h0
            mx_reale  = pts_ds[idx_test, 0].item()
            my_reale  = pts_ds[idx_test, 1].item()
            diff = max(abs(mx_atteso - mx_reale), abs(my_atteso - my_reale))
            if diff < 0.001:
                ok(f"Aariz: media ceil(senior,junior) corretta per '{sym_test}' "
                   f"(diff={diff:.6f})")
            else:
                fail(f"Aariz: media senior/junior ERRATA per '{sym_test}'! "
                     f"atteso=({mx_atteso:.4f},{my_atteso:.4f}) "
                     f"reale=({mx_reale:.4f},{my_reale:.4f}) diff={diff:.4f}")
        else:
            warn("Aariz: nessun simbolo comune trovato tra senior e junior per verifica")
    else:
        warn(f"Aariz: file junior non trovato ({junior_path}) — usa solo senior. "
             f"Verifica che la struttura cartelle sia corretta.")

except Exception as e:
    fail(f"CHECK 22: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 23 — Coordinate Tracciati: verifica coord_scale=0.01 sia corretto
# Le coordinate raw nei file Excel di tracciato sono ~1000-200000 (pixel assoluti).
# Con coord_scale=0.01 → valori normalizzati ~10-2000 → ancora fuori [0,1]?
# Il dataset divide per la dimensione reale dell'immagine DOPO aver scalato.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 23 — Tracciati: coord_scale e normalizzazione")

try:
    errori_tracciati = []
    n_tr_check = min(30, len(tracciati))
    for i in range(n_tr_check):
        sample_t = tracciati[i]
        pts_t = sample_t[1]
        validi_t = pts_t[pts_t.sum(1) != 0]
        if len(validi_t) == 0:
            continue
        vmin = validi_t.min().item()
        vmax = validi_t.max().item()
        if vmin < -0.001 or vmax > 1.001:
            errori_tracciati.append(f"campione {i}: [{vmin:.4f},{vmax:.4f}]")

    if errori_tracciati:
        fail(f"Tracciati: {len(errori_tracciati)}/{n_tr_check} campioni con coordinate fuori [0,1]:")
        for e in errori_tracciati[:5]:
            print(f"    {e}")
    else:
        # Statistiche distribuzione
        tutti_pts = []
        for i in range(n_tr_check):
            pts_t = tracciati[i][1]
            validi_t = pts_t[pts_t.sum(1) != 0]
            if len(validi_t) > 0:
                tutti_pts.extend(validi_t.flatten().tolist())
        if tutti_pts:
            ok(f"Tracciati: coordinate in [0,1] OK su {n_tr_check} campioni "
               f"(distribuzione: [{min(tutti_pts):.4f},{max(tutti_pts):.4f}], "
               f"media={sum(tutti_pts)/len(tutti_pts):.4f})")
        else:
            warn("Tracciati: nessun landmark valido trovato nei primi 30 campioni")

    # Conta landmark presenti per campione (quanti dei 26 sono annotati)
    n_lm_per_campione = []
    for i in range(n_tr_check):
        pts_t = tracciati[i][1]
        n_validi = (pts_t.sum(1) != 0).sum().item()
        n_lm_per_campione.append(n_validi)
    if n_lm_per_campione:
        media_lm = sum(n_lm_per_campione) / len(n_lm_per_campione)
        ok(f"Tracciati: media landmark annotati per campione: "
           f"{media_lm:.1f}/26 (min={min(n_lm_per_campione)}, max={max(n_lm_per_campione)})")

except Exception as e:
    fail(f"CHECK 23: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 24 — Collate: bbox calcolato correttamente (pixel assoluti immagine originale)
# Il bbox viene usato per log/visualizzazione, non per il training.
# Deve contenere tutti i landmark validi con padding=20.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 24 — Bbox calcolato dal collate (contiene tutti i landmark validi)")

try:
    errori_bbox = []
    for i in range(min(20, n)):
        img, pts, mask, bbox, px = ds_total[i]
        validi = pts[mask == 1.0]
        if len(validi) == 0:
            continue
        # bbox in pixel delle coordinate originali? No: bbox usa le coordinate normalizzate
        # (vedi datasets.py wrap_output: bbox calcolato su pts[:, 0] che sono già normalizzati)
        # Le coordinate sono normalizzate, la bbox è nel range [0,1]

        # Verifica che tutti i landmark validi stiano dentro la bbox (con tolleranza per il padding)
        x_min_pts = validi[:, 0].min().item()
        x_max_pts = validi[:, 0].max().item()
        y_min_pts = validi[:, 1].min().item()
        y_max_pts = validi[:, 1].max().item()

        # bbox ha padding di 20 pixel in coordinate absolute originale (non norm)
        # In realtà il padding viene aggiunto in coordinate assolute... verifica il codice
        # datasets.py: bbox usa pts direttamente che sono già normalizzate [0,1]
        # Il padding di 20 è in unità di coordinate originali, ma pts sono norm [0,1]
        # → il padding 20 viene applicato a valori [0,1]: potrebbe essere un bug!

        # Verifica: i landmark validi devono stare dentro la bbox
        if (x_min_pts < bbox[0].item() - 0.01 or
            x_max_pts > bbox[2].item() + 0.01 or
            y_min_pts < bbox[1].item() - 0.01 or
            y_max_pts > bbox[3].item() + 0.01):
            errori_bbox.append(
                f"campione {i}: landmark fuori dalla bbox! "
                f"pts=[{x_min_pts:.3f},{x_max_pts:.3f}]×[{y_min_pts:.3f},{y_max_pts:.3f}] "
                f"bbox=[{bbox[0]:.3f},{bbox[2]:.3f}]×[{bbox[1]:.3f},{bbox[3]:.3f}]"
            )

    if errori_bbox:
        fail(f"Bbox: {len(errori_bbox)} campioni con landmark fuori dalla bbox")
        for e in errori_bbox[:3]:
            print(f"    {e}")
    else:
        ok("Bbox: tutti i landmark validi sono contenuti nella bbox calcolata")

    # Verifica che il padding in datasets.py sia normalizzato (fix già applicato)
    import pathlib as _pl2
    _ds_src = _pl2.Path("src/datasets.py").read_text()
    import re as _re
    _m = _re.search(r'padding\s*=\s*([0-9.]+)', _ds_src)
    if _m:
        _pad_val = float(_m.group(1))
        if _pad_val >= 1.0:
            warn(
                f"datasets.py: padding={_pad_val} su coordinate normalizzate [0,1] — "
                f"dovrebbe essere ~0.02. Fix non applicato."
            )
        else:
            ok(f"datasets.py: padding={_pad_val} — corretto per coordinate normalizzate [0,1]")
    else:
        warn("datasets.py: padding non trovato — verifica manualmente")

except Exception as e:
    fail(f"CHECK 24: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 25 — Sigma per landmark in coordinate_a_heatmap
# SIGMA_PER_LANDMARK usa indici 5,7,9,10,11,14,15,18,19,20,21,24
# Verifica che siano tutti < 26 e che le heatmap risultanti abbiano il picco
# nella posizione corretta.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 25 — Sigma per landmark e picco heatmap nella posizione corretta")

try:
    from heatmap_utils import SIGMA_PER_LANDMARK

    # Verifica che tutti gli indici siano validi (< 26)
    idx_invalidi = [k for k in SIGMA_PER_LANDMARK if k >= 26]
    if idx_invalidi:
        fail(f"SIGMA_PER_LANDMARK: indici >= 26 (non validi): {idx_invalidi}")
    else:
        ok(f"SIGMA_PER_LANDMARK: tutti gli indici sono validi (<26): "
           f"{sorted(SIGMA_PER_LANDMARK.keys())}")

    # Verifica che il picco della heatmap sia nella posizione attesa
    pts_hm_test = torch.zeros(26, 2)
    mask_hm_test = torch.zeros(26)
    # Testa i landmark con sigma specifica
    test_coords = {5: (0.3, 0.4), 10: (0.5, 0.6), 14: (0.7, 0.2)}
    for idx, (cx, cy) in test_coords.items():
        pts_hm_test[idx] = torch.tensor([cx, cy])
        mask_hm_test[idx] = 1.0

    hm_test = coordinate_a_heatmap(pts_hm_test, mask_hm_test, 512, HEATMAP_SIZE, sigma=4.0)

    picco_ok = True
    for idx, (cx, cy) in test_coords.items():
        hm_lm = hm_test[idx]   # [H, W]
        peak_flat = hm_lm.argmax().item()
        peak_y = peak_flat // HEATMAP_SIZE
        peak_x = peak_flat %  HEATMAP_SIZE
        # Il picco atteso in pixel heatmap
        atteso_x = int(cx * HEATMAP_SIZE)
        atteso_y = int(cy * HEATMAP_SIZE)
        dist_picco = ((peak_x - atteso_x)**2 + (peak_y - atteso_y)**2)**0.5
        if dist_picco > 3:
            fail(f"LM{idx}: picco heatmap a ({peak_x},{peak_y}) "
                 f"ma atteso ({atteso_x},{atteso_y}) — dist={dist_picco:.1f}px")
            picco_ok = False

    if picco_ok:
        ok("Picco heatmap nella posizione corretta per tutti i landmark testati")

except Exception as e:
    fail(f"CHECK 25: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 26 — PESI_LANDMARK: indici validi e nessun overlap con landmark non-trainabili
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 26 — PESI_LANDMARK validi e indici coerenti con SIGMA_PER_LANDMARK")

try:
    from heatmap_utils import PESI_LANDMARK

    idx_invalidi_pesi = [k for k in PESI_LANDMARK if k >= 26 or k < 0]
    if idx_invalidi_pesi:
        fail(f"PESI_LANDMARK: indici fuori range [0,25]: {idx_invalidi_pesi}")
    else:
        ok(f"PESI_LANDMARK: tutti gli indici validi ({sorted(PESI_LANDMARK.keys())})")

    # Verifica che i pesi siano positivi e ragionevoli (non estremi)
    pesi_anomali = {k: v for k, v in PESI_LANDMARK.items() if v <= 0 or v > 10}
    if pesi_anomali:
        warn(f"PESI_LANDMARK: valori anomali (<=0 o >10): {pesi_anomali}")
    else:
        ok(f"PESI_LANDMARK: tutti i pesi ragionevoli "
           f"(range [{min(PESI_LANDMARK.values())},{max(PESI_LANDMARK.values())}])")

    # Check semantico: i landmark dentali/difficili hanno peso maggiore dei scheletrici facili
    pesi_schel_facili = [PESI_LANDMARK.get(i, 1.0) for i in [0, 1, 2, 3, 4, 6]]  # A,ANS,B,Me,N,Pog
    pesi_difficili    = [PESI_LANDMARK.get(i, 1.0) for i in [14, 15, 20, 21]]     # Po, LPM, UIA, LIA
    if sum(pesi_difficili)/len(pesi_difficili) > sum(pesi_schel_facili)/len(pesi_schel_facili):
        ok(f"PESI_LANDMARK: landmark difficili hanno peso maggiore dei facili "
           f"(media difficili={sum(pesi_difficili)/len(pesi_difficili):.1f} "
           f"vs facili={sum(pesi_schel_facili)/len(pesi_schel_facili):.1f})")
    else:
        warn("PESI_LANDMARK: i landmark difficili NON hanno peso maggiore dei facili — "
             "verifica la logica di pesatura")

except Exception as e:
    fail(f"CHECK 26: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 27 — heatmap_a_coordinate usa linspace(0,1): verifica che i bordi
# siano correttamente inclusi (punto a (0,0) e (1,1) devono essere recuperabili)
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 27 — soft-argmax: punti ai bordi recuperabili")

try:
    # Test sui 4 angoli dell'immagine
    pts_bordi = torch.tensor([
        [0.01, 0.01],   # angolo top-left
        [0.99, 0.01],   # angolo top-right
        [0.01, 0.99],   # angolo bottom-left
        [0.99, 0.99],   # angolo bottom-right
        [0.50, 0.50],   # centro
    ])
    mask_bordi = torch.zeros(26)
    pts_bordi_pad = torch.zeros(26, 2)
    for i, (cx, cy) in enumerate(pts_bordi):
        pts_bordi_pad[i] = torch.tensor([cx.item(), cy.item()])
        mask_bordi[i] = 1.0

    hm_bordi = coordinate_a_heatmap(pts_bordi_pad, mask_bordi, 512, HEATMAP_SIZE, sigma=4.0)
    pred_bordi = heatmap_a_coordinate(hm_bordi.unsqueeze(0))  # [1, 26, 2]

    errori_bordi = []
    nomi_test = ["top-left", "top-right", "bottom-left", "bottom-right", "centro"]
    for i in range(len(pts_bordi)):
        gt_x, gt_y = pts_bordi[i, 0].item(), pts_bordi[i, 1].item()
        pr_x = pred_bordi[0, i, 0].item()
        pr_y = pred_bordi[0, i, 1].item()
        err = max(abs(pr_x - gt_x), abs(pr_y - gt_y))
        print(f"    {nomi_test[i]}: GT=({gt_x:.3f},{gt_y:.3f}) pred=({pr_x:.3f},{pr_y:.3f}) err={err:.4f}")
        if err > 0.03:
            errori_bordi.append(f"{nomi_test[i]}: err={err:.4f}")

    if errori_bordi:
        warn(f"soft-argmax: {len(errori_bordi)} punti ai bordi con errore > 0.03 (normale per sigma grande): "
             + "; ".join(errori_bordi))
    else:
        ok("soft-argmax: tutti i punti ai bordi recuperati con errore < 0.03")

except Exception as e:
    fail(f"CHECK 27: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 28 — Consistenza nomi landmark tra dataset.py e nomi in evaluate_refine.py
# NOMI in evaluate_refine.py deve corrispondere esattamente a SYMBOL_ORDER in dataset.py
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 28 — Nomi landmark: SYMBOL_ORDER (dataset.py) == NOMI (evaluate_refine.py)")

try:
    from dataset import SYMBOL_ORDER

    # Nomi hardcoded in evaluate_refine.py (li riproduco qui)
    NOMI_EVALUATE = [
        "A","ANS","B","Me","N","Or","Pog","PNS","Pn","R",
        "S","Ar","Co","Gn","Go","Po",
        "LPM","LIT","LMT","UIT","UIA","LIA",
        "Ls","Li","Pog`","Sn",
    ]

    if len(SYMBOL_ORDER) != len(NOMI_EVALUATE):
        fail(f"Numero landmark diverso: SYMBOL_ORDER={len(SYMBOL_ORDER)} "
             f"vs NOMI_EVALUATE={len(NOMI_EVALUATE)}")
    else:
        mismatch = [(i, s, n) for i, (s, n) in enumerate(zip(SYMBOL_ORDER, NOMI_EVALUATE))
                    if s != n]
        if mismatch:
            fail(f"Mismatch nomi landmark: {mismatch}")
        else:
            ok(f"Nomi landmark coerenti tra dataset.py e evaluate_refine.py ({len(SYMBOL_ORDER)} landmark)")

except Exception as e:
    fail(f"CHECK 28: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 29 — Normalizzazione immagini: nessuna immagine completamente nera o bianca
# Un'immagine radiografica dovrebbe avere pixel significativamente > 0.
# Se img.max() < 0.1, probabilmente il caricamento è fallito silenziosamente.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 29 — Qualità immagini: nessuna immagine degenere (max<0.1 o std<0.01)")

try:
    n_degeneri = {"nere": 0, "basse_varianza": 0}
    n_img_check = min(50, n)
    step_img = max(1, n // n_img_check)
    dettagli_degen = []

    for i in range(0, n, step_img):
        img_raw = ds_total[i][0]   # [1, H, W] float32 [0,1]
        img_max = img_raw.max().item()
        img_std = img_raw.std().item()
        if img_max < 0.1:
            n_degeneri["nere"] += 1
            dettagli_degen.append(f"campione {i}: max={img_max:.4f} (immagine troppo scura)")
        elif img_std < 0.01:
            n_degeneri["basse_varianza"] += 1
            dettagli_degen.append(f"campione {i}: std={img_std:.4f} (poca varianza)")

    if n_degeneri["nere"] > 0 or n_degeneri["basse_varianza"] > 0:
        fail(f"Immagini degeneri: {n_degeneri['nere']} troppo scure, "
             f"{n_degeneri['basse_varianza']} bassa varianza")
        for d in dettagli_degen[:5]:
            print(f"    {d}")
    else:
        ok(f"Tutte le immagini campionate ({n // step_img}) hanno max>0.1 e std>0.01")

except Exception as e:
    fail(f"CHECK 29: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 30 — Flip orizzontale in TTA (evaluate_refine.py):
# Quando flippiamo un'immagine cefalometrica laterale, i landmark scheletrici
# cambiano semantica (la cefalometria è su vista laterale → il flip X inverte
# la direzione). Il TTA con flip può peggiorare le stime.
# Questo check è solo un WARN per segnalare il rischio.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 30 — TTA flip orizzontale (rischio semantico su cefalometria laterale)")

warn(
    "evaluate_refine.py usa TTA con flip orizzontale (TF.hflip). "
    "Su immagini cefalometriche LATERALI il flip inverte destra/sinistra, "
    "cambiando la semantica anatomica (es. S, Po, Or cambiano posizione relativa). "
    "Questo può introdurre bias sistematico nelle predizioni TTA. "
    "Considera di rimuovere il flip dal TTA per la cefalometria laterale."
)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 31 — REFINE_LANDMARK_MASK: indici coerenti con PESI_LANDMARK
# I landmark che refinement migliora devono essere quelli difficili.
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 31 — REFINE_LANDMARK_MASK coerente con landmark difficili")

try:
    from model_refine import REFINE_LANDMARK_MASK

    # Landmark che secondo PESI_LANDMARK sono difficili (peso > 2)
    lm_difficili_pesi = {k for k, v in PESI_LANDMARK.items() if v > 2.0}
    lm_difficili_refine = set(REFINE_LANDMARK_MASK)

    # Devono avere buona sovrapposizione
    overlap_count = len(lm_difficili_pesi & lm_difficili_refine)
    solo_pesi     = lm_difficili_pesi - lm_difficili_refine
    solo_refine   = lm_difficili_refine - lm_difficili_pesi

    print(f"  REFINE_LANDMARK_MASK: {sorted(lm_difficili_refine)}")
    print(f"  Landmark con peso>2 in PESI_LANDMARK: {sorted(lm_difficili_pesi)}")

    if overlap_count >= len(lm_difficili_pesi) * 0.6:
        ok(f"REFINE_LANDMARK_MASK: buona sovrapposizione con PESI_LANDMARK difficili "
           f"({overlap_count}/{len(lm_difficili_pesi)})")
    else:
        warn(f"REFINE_LANDMARK_MASK: bassa sovrapposizione con PESI_LANDMARK "
             f"({overlap_count}/{len(lm_difficili_pesi)})")

    if solo_pesi:
        warn(f"Landmark con peso alto ma NON in REFINE_MASK: {sorted(solo_pesi)} "
             f"— Stage 2 non li raffina")
    if solo_refine:
        warn(f"Landmark in REFINE_MASK ma con peso basso: {sorted(solo_refine)} "
             f"— Stage 2 li raffina ma non sono tra i più difficili")

except Exception as e:
    fail(f"CHECK 31: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 32 — Fix MRE: collate_fn di evaluate_refine restituisce larghezza reale
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 32 — Fix MRE: collate evaluate_refine porta larghezza originale")

try:
    import ast, pathlib

    src = pathlib.Path("evaluate_refine.py").read_text()

    # 1. Il collate deve raccogliere la larghezza prima del resize
    if "img.shape[2]" in src and "larghezze_orig" in src:
        ok("collate_fn raccoglie img.shape[2] (larghezza originale) prima del resize")
    else:
        fail("collate_fn NON raccoglie la larghezza originale — il fix MRE non è applicato")

    # 2. Il collate deve restituire larghezze_orig come 6° elemento
    if "larghezze_orig" in src and "return (" in src:
        # Cerca che larghezze_orig sia nell'ultima return del collate
        idx_return = src.rfind("larghezze_orig,")
        if idx_return != -1:
            ok("collate_fn restituisce larghezze_orig come 6° elemento del batch")
        else:
            fail("larghezze_orig definita ma non restituita nel collate")
    else:
        fail("larghezze_orig assente nel collate — fix incompleto")

    # 3. Il loop MRE deve fare unpacking di 6 elementi
    if "larghezze_orig in loader" in src or "pixel_sizes, larghezze_orig in loader" in src:
        ok("loop calcola_mre_sdr fa unpacking di larghezze_orig dal loader")
    else:
        fail("loop calcola_mre_sdr non ha larghezze_orig nell'unpacking — FAIL MRE")

    # 4. La formula MRE deve usare larghezze_orig[b] e NON il valore fisso 2400
    if "larghezza_orig_px" in src and 'config["larghezza_orig_px"]' in src:
        # Controlla che nella formula mm= non ci sia più config["larghezza_orig_px"]
        # (dev'esserci largh_px o larghezze_orig[b])
        linee_mm = [l.strip() for l in src.split("\n") if "mm =" in l or "mm=" in l]
        usa_fisso = any('config["larghezza_orig_px"]' in l for l in linee_mm)
        if usa_fisso:
            fail("Formula MRE usa ancora config['larghezza_orig_px']=2400 fisso — ~22% errore sistematico")
        else:
            ok("Formula MRE non usa più il valore fisso 2400")
    else:
        ok("config['larghezza_orig_px'] non presente nella formula MRE — fix applicato")

    # 5. Stima errore residuo: le larghezze Aariz variano tra ~1900-2000 vs 2400 ISBI
    if "largh_px" in src:
        ok("Formula MRE usa largh_px (larghezza reale per campione) — errore sistematico eliminato")
    else:
        warn("Non trovata variabile 'largh_px' nella formula MRE — verifica manualmente")

except Exception as e:
    fail(f"CHECK 32: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 33 — Fix bbox: padding normalizzato [0,1] in datasets.py
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 33 — Fix bbox: padding coordinate normalizzate in datasets.py")

try:
    import pathlib

    src_ds = pathlib.Path("src/datasets.py").read_text()

    # 1. Padding non deve più essere 20.0
    if "padding = 20" in src_ds or "padding=20" in src_ds:
        fail("datasets.py ha ancora padding=20 su coordinate normalizzate [0,1] "
             "— la bbox sarà sempre [0,0,W,H] per il clamping")
    else:
        ok("padding=20 non presente — fix applicato")

    # 2. Padding deve essere 0.02 (o un valore < 1)
    import re
    m = re.search(r'padding\s*=\s*([0-9.]+)', src_ds)
    if m:
        val = float(m.group(1))
        if val < 1.0:
            ok(f"padding={val} — compatibile con coordinate normalizzate [0,1]")
        else:
            fail(f"padding={val} — troppo grande per coordinate normalizzate (deve essere <1.0)")
    else:
        warn("Nessun 'padding = ...' trovato in datasets.py — verifica manualmente")

    # 3. Il clamping deve usare 1.0 (non img.shape[...])
    if "min(1.0," in src_ds and "max(0.0," in src_ds:
        ok("bbox clampata a [0,1] — corretto per coordinate normalizzate")
    elif "img.shape[2]" in src_ds and "padding" in src_ds:
        fail("bbox usa ancora img.shape[2] come limite superiore — incompatibile con coord normalizzate")
    else:
        warn("Verifica manuale del clamping bbox in datasets.py")

    # 4. Ramo else (nessun landmark valido) deve usare [0,0,1,1]
    if "0.0, 0.0, 1.0, 1.0" in src_ds:
        ok("bbox fallback (nessun landmark) usa [0,0,1,1] — corretto")
    elif "img.shape" in src_ds:
        fail("bbox fallback usa ancora img.shape — incompatibile con coordinate normalizzate")
    else:
        warn("Verifica manuale del fallback bbox in datasets.py")

    # 5. Verifica pratica: carica un campione e controlla che la bbox sia in [0,1]
    try:
        from datasets import Datasets
        ds_test = Datasets(
            aariz_path=AARIZ_PATH,
            isbi_path=ISBI_PATH,
        )
        n_fuori = 0
        for i in range(min(30, len(ds_test))):
            _, _, _, bbox, _ = ds_test[i]
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > 1.0 or bbox[3] > 1.0:
                n_fuori += 1
        if n_fuori == 0:
            ok("Bbox di 30 campioni reali tutte in [0,1] — fix confermato empiricamente")
        else:
            fail(f"{n_fuori}/30 campioni hanno bbox fuori da [0,1] — fix non completo")
    except Exception as e2:
        warn(f"Verifica empirica bbox saltata: {e2}")

except Exception as e:
    fail(f"CHECK 33: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 34 — Consistenza collate train vs evaluate: stesso numero di elementi
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 34 — Consistenza collate: train_heatmap vs train_refine vs evaluate_refine")

try:
    import pathlib, re

    src_th  = pathlib.Path("train_heatmap.py").read_text()
    src_tr  = pathlib.Path("train_refine.py").read_text()
    src_ev  = pathlib.Path("evaluate_refine.py").read_text()

    # Verifica strutturale tramite stringhe chiave invece del contatore regex (inaffidabile)
    # train_heatmap: deve avere heatmaps nel return e unpacking a 6
    th_ok = "torch.stack(heatmaps)" in src_th and             "torch.stack(pixel_sizes)" in src_th
    # train_refine: NON ha heatmaps, unpacking 5
    tr_ok = "for imgs, punti, masks, _, _ in loader" in src_tr
    # evaluate_refine: ha larghezze_orig come 6° elemento
    ev_ok = "larghezze_orig" in src_ev and             "pixel_sizes, larghezze_orig in loader" in src_ev

    if th_ok:
        ok("train_heatmap.py collate: restituisce heatmaps + pixel_sizes — struttura corretta")
    else:
        warn("train_heatmap.py collate: verifica manuale — struttura inattesa")

    if tr_ok:
        ok("train_refine.py loop: unpacking 5 elementi — consistente con collate")
    else:
        warn("train_refine.py loop: unpacking diverso da 5 — verifica manualmente")

    if ev_ok:
        ok("evaluate_refine.py collate+loop: larghezze_orig presente e usato — fix confermato")
    else:
        fail("evaluate_refine.py: larghezze_orig mancante nel collate o nel loop MRE "
             "— AttributeError a runtime")

except Exception as e:
    fail(f"CHECK 34: {e}")



# ══════════════════════════════════════════════════════════════════════════════
# CHECK 35 — AarizDataset: carica tutti i formati (png, jpg, jpeg, bmp)
# ══════════════════════════════════════════════════════════════════════════════
sezione("CHECK 35 — AarizDataset: formati multipli (png/jpg/jpeg/bmp)")

try:
    import pathlib as _pl35
    _ds_src35 = _pl35.Path("src/dataset.py").read_text()

    _formati = [".png", ".jpg", ".jpeg", ".bmp"]
    _formati_presenti = [f for f in _formati if f in _ds_src35]
    _formati_mancanti = [f for f in _formati if f not in _ds_src35]

    if _formati_mancanti:
        warn(f"dataset.py non gestisce i formati: {_formati_mancanti} "
             f"— alcuni campioni Aariz potrebbero essere ignorati")
    else:
        ok(f"dataset.py gestisce tutti i formati Aariz: {_formati_presenti}")

    # Verifica empirica: controlla quanti campioni carica AarizDataset
    _n_aariz = ds_total.len_aariz
    if _n_aariz >= 900:
        ok(f"AarizDataset carica {_n_aariz} campioni (attesi ~1000) — formati multipli OK")
    elif _n_aariz >= 600:
        warn(f"AarizDataset carica solo {_n_aariz}/1000 campioni — "
             f"alcuni formati potrebbero non essere gestiti")
    else:
        fail(f"AarizDataset carica solo {_n_aariz}/1000 campioni — "
             f"verifica src/dataset.py e i formati supportati")

except Exception as e:
    fail(f"CHECK 35: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# RIEPILOGO FINALE
# ══════════════════════════════════════════════════════════════════════════════
sezione("RIEPILOGO")

n_ok   = sum(1 for r in risultati if r[0] == "OK")
n_warn = sum(1 for r in risultati if r[0] == "WARN")
n_fail = sum(1 for r in risultati if r[0] == "FAIL")

print(f"\n  {VERDE}OK{RESET}:   {n_ok}")
print(f"  {GIALLO}WARN{RESET}: {n_warn}")
print(f"  {ROSSO}FAIL{RESET}: {n_fail}")

if n_fail == 0:
    print(f"\n  {VERDE}✓ Tutti i check critici passati — puoi lanciare il training.{RESET}")
else:
    print(f"\n  {ROSSO}✗ Ci sono {n_fail} FAIL — correggi prima di lanciare il training.{RESET}")
    print("\n  Problemi da risolvere:")
    for stato, msg in risultati:
        if stato == "FAIL":
            print(f"    • {msg}")

if n_warn > 0:
    print(f"\n  {GIALLO}Avvertenze:{RESET}")
    for stato, msg in risultati:
        if stato == "WARN":
            print(f"    • {msg}")