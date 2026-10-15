"""
analisi_stage2.py — does Stage 2 actually fix Stage 1 errors, and would it help
to apply it only where the uncertainty is high?

Uses the checkpoints already trained (no training). On the validation split:
  1. Stage 1 (with TTA) for every image, plus TTA and heatmap uncertainty.
  2. Stage 2 applied to ALL 26 landmarks (with the usual guard), so its effect
     can be measured everywhere, not only on the fixed mask.
Reports:
  A. MRE: Stage 1 / Stage 2 on the current fixed mask / on all landmarks /
     ideal case (Stage 2 kept only where it helps = upper bound of any selection rule)
  B. Stage 2 effect per uncertainty quintile: does it help on the uncertain points?
  C. Threshold sweep: apply Stage 2 only where sigma > threshold. The threshold
     is chosen on the even images and checked on the odd ones (honest estimate).
  D. Per landmark effect.
  E. How big the Stage 2 corrections are, how often the guard rejects them,
     and how many errors are larger than the maximum correction.
Writes one row per (image, landmark) to <dir>/analisi_stage2.csv.

    python analisi_stage2.py --dir esperimenti/probabilistic
    python analisi_stage2.py --dir esperimenti/fixed --batch-size 4
    python analisi_stage2.py --da-csv esperimenti/probabilistic/analisi_stage2.csv   # report only
"""
import argparse
import csv
import os
import runpy
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import Datasets, crea_split
from src.heatmap_utils import HEATMAP_TRAIN_SIZE
from src.inferenza import (carica_stage1, carica_stage2, collate_base, predici_con_tta,
                           applica_stage2, errori_mm, stage2_predizioni)
from src.incertezza import momenti_heatmap, cov_in_mm, sigma_radiale

NOMI = ["A","ANS","B","Me","N","Or","Pog","PNS","Pn","R","S","Ar","Co","Gn","Go","Po",
        "LPM","LIT","LMT","UIT","UIA","LIA","Ls","Li","Pog`","Sn"]

CONFIG = {
    "aariz_path":    "data/Aariz",
    "isbi_path":     "data/Cepha400",
    "num_landmarks": 26,
    "batch_size":    2,
    "val_split":     0.2,
    "img_size":      512,
    "heatmap_size":  HEATMAP_TRAIN_SIZE,
    "hrnet_channels": [64, 128, 256, 512, 1024],
    "usa_tta":       True,
    "tta_angoli":    [-5.0, -2.5, 0.0, 2.5, 5.0],
    "fattore_guardia": 1.2,
    "tta_stage2":      True,    # also run Stage 2 with test-time augmentation (for strategie_stage2.py)
}


def stage2_su_tutti(stage2, coords, config, x_out=None, originali=None, dims=None, ps=None):
    """Stage 2 applied to all landmarks of a batch (either kind of Stage 2).
    Returns (corrected coords [B,N,2], proposed offsets [B,N,2] normalized,
    guard_ok [B,N])."""
    B, N, _ = coords.shape
    grezzo = applica_stage2(stage2, coords, x_out=x_out, originali=originali, dims=dims, pixel_size=ps,
                            landmark_da_correggere=range(N), fattore_guardia=1e9)
    offset = grezzo - coords
    g = config["fattore_guardia"]
    if getattr(stage2, "tipo_input", "feature") == "immagine":
        off_mm = offset * dims.view(-1, 1, 2) * ps.view(-1, 1, 1)
        ok = (off_mm.abs() <= g * stage2.max_offset_mm).all(dim=2)
    else:
        ok = (offset.abs() <= g * stage2.max_offset).all(dim=2)
    return torch.where(ok.unsqueeze(-1), grezzo, coords), offset, ok


def raccogli(stage1, stage2, loader, device, config):
    """One record per valid (image, landmark)."""
    # Extra columns for strategie_stage2.py (image-based Stage 2 only; NaN otherwise):
    #   e1x, e1y    Stage 1 error vector (prediction - truth), mm
    #   d2x, d2y    Stage 2 proposed shift (single crop, no guard), mm
    #   d2tx, d2ty  Stage 2 proposed shift with test-time augmentation, mm
    #   s2_hm       spread of the Stage 2 crop distribution, mm
    #   s2_tta      spread of the Stage 2 TTA answers, mm
    extra = ("e1x", "e1y", "d2x", "d2y", "d2tx", "d2ty", "s2_hm", "s2_tta")
    r = {k: [] for k in ("img", "lm", "err1", "err2", "sigma_tta", "sigma_heatmap",
                         "corr_mm", "guardia_ok") + extra}
    n_img = 0
    for immagini, punti, mask, _, ps, dims, originali in loader:
        immagini, punti, mask = immagini.to(device), punti.to(device), mask.to(device)
        ps, dims = ps.to(device), dims.to(device)
        out = predici_con_tta(stage1, immagini, config)
        c1 = out["coords"][:, :mask.shape[1]]
        c2, offset, ok = stage2_su_tutti(stage2, c1, config, x_out=out["x_out"], originali=originali,
                                         dims=dims, ps=ps)

        e1 = errori_mm(c1, punti, dims, ps)
        e2 = errori_mm(c2, punti, dims, ps)
        s_tta = sigma_radiale(cov_in_mm(out["cov_tta"], dims, ps))
        _, cov_hm, _ = momenti_heatmap(out["pred_raw"][:, 0::3], stage1.heatmap_mode, stage1.griglia)
        s_hm = sigma_radiale(cov_in_mm(cov_hm, dims, ps))
        corr_mm = torch.sqrt(((offset * dims.view(-1, 1, 2)) ** 2).sum(-1)) * ps.view(-1, 1)

        B, N = mask.shape
        nan = torch.full((B, N), float("nan"), device=device)
        col = {k: nan for k in extra}
        e1v = (c1 - punti) * dims.view(-1, 1, 2) * ps.view(-1, 1, 1)
        col["e1x"], col["e1y"] = e1v[..., 0], e1v[..., 1]
        if getattr(stage2, "tipo_input", "feature") == "immagine":
            b_idx = torch.arange(B, device=device).repeat_interleave(N)
            l_idx = torch.arange(N, device=device).repeat(B)
            det = stage2_predizioni(stage2, c1.reshape(-1, 2), b_idx, l_idx, originali, ps,
                                    tta=config.get("tta_stage2", True))
            col["d2x"], col["d2y"] = det["offset"][:, 0].view(B, N), det["offset"][:, 1].view(B, N)
            col["d2tx"], col["d2ty"] = det["offset_tta"][:, 0].view(B, N), det["offset_tta"][:, 1].view(B, N)
            col["s2_hm"] = det["var_hm"].sqrt().view(B, N)
            col["s2_tta"] = det["var_tta"].sqrt().view(B, N)
        for b in range(B):
            for i in range(N):
                if mask[b, i].item() != 1.0:
                    continue
                r["img"].append(n_img + b); r["lm"].append(i)
                r["err1"].append(e1[b, i].item()); r["err2"].append(e2[b, i].item())
                r["sigma_tta"].append(s_tta[b, i].item()); r["sigma_heatmap"].append(s_hm[b, i].item())
                r["corr_mm"].append(corr_mm[b, i].item()); r["guardia_ok"].append(bool(ok[b, i].item()))
                for k in extra:
                    r[k].append(col[k][b, i].item())
        n_img += B
        print(f"\r  images processed: {n_img}", end="", flush=True)
    print()
    return {k: np.asarray(v) for k, v in r.items()}


SDR_MM = [2.0, 2.5, 3.0, 4.0]   # accuracy thresholds: % of points within X mm


def errori_con_selezione(r, sel):
    """Per-point errors when Stage 2 is applied only where sel is True."""
    return np.where(sel, r["err2"], r["err1"])


def mre_con_selezione(r, sel):
    """MRE when Stage 2 is applied only where sel is True."""
    return float(errori_con_selezione(r, sel).mean())


def intestazione_sdr():
    return " ".join(f"{'<' + format(t, 'g') + 'mm':>7}" for t in SDR_MM)


def colonne_sdr(err):
    """Accuracy (SDR): share of points with error below each threshold of SDR_MM."""
    return " ".join(f"{(err < t).mean():7.1%}" for t in SDR_MM)


def carica_csv(path):
    """Rebuilds the per-point records from a CSV written by this script."""
    with open(path) as f:
        righe = list(csv.DictReader(f))
    r = {
        "img": np.array([int(x["img"]) for x in righe]),
        "lm": np.array([NOMI.index(x["landmark"]) for x in righe]),
        "guardia_ok": np.array([x["guardia_ok"] == "True" for x in righe]),
    }
    for k in ("err1", "err2", "sigma_tta", "sigma_heatmap", "corr_mm"):
        r[k] = np.array([float(x[k]) for x in righe])
    return r


def report(r, max_corr_mm, maschera_fissa):
    e1, e2, lm = r["err1"], r["err2"], r["lm"]
    maschera = np.isin(lm, sorted(maschera_fissa))

    print(f"\n{'='*64}\nA. MRE (mm) and accuracy — {len(e1)} points, {len(np.unique(r['img']))} images\n{'='*64}")
    print(f"  {'':36s} {'MRE':>6}  {intestazione_sdr()}")
    for nome, err in (("Stage 1 only", e1),
                      ("Stage 2 on the current fixed mask", errori_con_selezione(r, maschera)),
                      ("Stage 2 on all landmarks", e2),
                      ("Ideal (Stage 2 only where it helps)", np.minimum(e1, e2))):
        print(f"  {nome:36s} {err.mean():6.3f}  {colonne_sdr(err)}")
    print(f"  fixed mask = {[NOMI[i] for i in sorted(maschera_fissa)]} | "
          f"'Ideal' = upper bound for ANY selection rule")
    migliora = e2 < e1 - 1e-6
    peggiora = e2 > e1 + 1e-6
    print(f"  Stage 2 improves {migliora.mean():.1%} of the points, worsens {peggiora.mean():.1%}, "
          f"leaves {1 - migliora.mean() - peggiora.mean():.1%} unchanged")

    for fonte in ("sigma_tta", "sigma_heatmap"):
        s = r[fonte]
        print(f"\n{'='*64}\nB. Stage 2 effect per quintile of {fonte}\n{'='*64}")
        print(f"  {'quintile':>8} {'sigma mm':>14} {'err S1':>7} {'err S2':>7} {'delta':>7} "
              f"{'improves':>8} {'err>maxcorr':>11}")
        bordi = np.quantile(s, [0, .2, .4, .6, .8, 1])
        for k in range(5):
            sel = (s >= bordi[k]) & ((s <= bordi[k + 1]) if k == 4 else (s < bordi[k + 1]))
            print(f"  {k+1:>8} {bordi[k]:6.2f}-{bordi[k+1]:<6.2f} {e1[sel].mean():7.3f} {e2[sel].mean():7.3f} "
                  f"{(e2[sel] - e1[sel]).mean():+7.3f} {migliora[sel].mean():8.1%} "
                  f"{(e1[sel] > max_corr_mm).mean():11.1%}")

        print(f"\nC. Apply Stage 2 only where {fonte} > threshold")
        print(f"  {'threshold':>9} {'% corrected':>11} {'MRE':>6}  {intestazione_sdr()}")
        print(f"  {'(none)':>9} {0:11.1%} {e1.mean():6.3f}  {colonne_sdr(e1)}")
        soglie = np.quantile(s, [0, .5, .6, .7, .8, .9, .95, .99])
        for t in soglie:
            err = errori_con_selezione(r, s > t)
            print(f"  {t:9.2f} {(s > t).mean():11.1%} {err.mean():6.3f}  {colonne_sdr(err)}")
        # honest estimate: threshold chosen on even images, measured on odd ones
        pari = (r["img"] % 2) == 0
        candidati = np.quantile(s[pari], np.linspace(0, 0.99, 100))
        rp = {k: v[pari] for k, v in r.items()}
        migliore = min(candidati, key=lambda t: mre_con_selezione(rp, rp[fonte] > t))
        rd = {k: v[~pari] for k, v in r.items()}
        print(f"  threshold chosen on even images: {migliore:.2f} mm -> on odd images "
              f"MRE {rd['err1'].mean():.3f} (Stage 1) vs {mre_con_selezione(rd, rd[fonte] > migliore):.3f} "
              f"(selective Stage 2), {(rd[fonte] > migliore).mean():.1%} corrected")
        print(f"  {'odd images':>21} {'MRE':>6}  {intestazione_sdr()}")
        print(f"  {'Stage 1 only':>21} {rd['err1'].mean():6.3f}  {colonne_sdr(rd['err1'])}")
        err_d = errori_con_selezione(rd, rd[fonte] > migliore)
        print(f"  {'selective Stage 2':>21} {err_d.mean():6.3f}  {colonne_sdr(err_d)}")

    print(f"\n{'='*64}\nD. Per landmark (Stage 2 applied to every point)\n{'='*64}")
    print(f"  {'':5s} {'err S1':>7} {'err S2':>7} {'delta':>7} {'improves':>8} {'in mask':>7}")
    for i in range(len(NOMI)):
        sel = lm == i
        if not sel.any():
            continue
        print(f"  {NOMI[i]:5s} {e1[sel].mean():7.3f} {e2[sel].mean():7.3f} {(e2[sel] - e1[sel]).mean():+7.3f} "
              f"{migliora[sel].mean():8.1%} {'yes' if i in maschera_fissa else '':>7}")

    print(f"\n{'='*64}\nE. Size of the Stage 2 corrections\n{'='*64}")
    c = r["corr_mm"]
    print(f"  max correction allowed (MAX_OFFSET): ~{max_corr_mm:.2f} mm on this data")
    print(f"  proposed correction: median {np.median(c):.3f} mm | 90th pct {np.quantile(c, .9):.3f} mm | "
          f"max {c.max():.3f} mm")
    print(f"  rejected by the guard: {(~r['guardia_ok']).mean():.1%}")
    print(f"  Stage 1 error: median {np.median(e1):.2f} mm | errors > max correction: "
          f"{(e1 > max_corr_mm).mean():.1%}")
    print(f"  Spearman(correction size, Stage 1 error): "
          f"{np.corrcoef(np.argsort(np.argsort(c)), np.argsort(np.argsort(e1)))[0, 1]:+.3f}"
          f"   (near 0 = Stage 2 does not 'see' where Stage 1 is wrong)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="experiment folder with modello_heatmap.pth and modello_refine.pth")
    ap.add_argument("--da-csv", help="only redo the report from a CSV written by this script (no models, no GPU)")
    ap.add_argument("--max-offset-mm", type=float, default=5.0,
                    help="with --da-csv: Stage 2 max correction per axis, for section E")
    ap.add_argument("--stage2", help="Stage 2 checkpoint (default: <dir>/modello_refine.pth)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()
    if a.da_csv:
        r = carica_csv(a.da_csv)
        stats = os.path.join(a.dir or os.path.dirname(a.da_csv), "landmark_stats.py")
        maschera_fissa = set(runpy.run_path(stats)["REFINE_LANDMARK_MASK"]) if os.path.exists(stats) else set()
        report(r, a.max_offset_mm * np.sqrt(2), maschera_fissa)
        return
    if not a.dir:
        ap.error("--dir is required (unless --da-csv is given)")
    if a.batch_size:
        CONFIG["batch_size"] = a.batch_size
    if a.no_tta:
        CONFIG["usa_tta"] = False
    CONFIG["stage1_path"] = os.path.join(a.dir, "modello_heatmap.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | dir: {a.dir} | TTA: {CONFIG['usa_tta']}")
    stage1 = carica_stage1(CONFIG, device)
    percorso_s2 = a.stage2 or os.path.join(a.dir, "modello_refine.pth")
    stage2 = carica_stage2(percorso_s2, CONFIG["num_landmarks"], device)
    if stage2 is None:
        raise SystemExit(f"{percorso_s2} not found")

    dataset = Datasets(aariz_path=CONFIG["aariz_path"], isbi_path=CONFIG["isbi_path"])
    _, val_ds = crea_split(dataset, CONFIG["val_split"], seed=42)
    loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0,
                        collate_fn=partial(collate_base, img_size=CONFIG["img_size"], con_originali=True))

    # Fixed Stage 2 mask of THIS experiment (run_tutto.slurm saves it in --dir and
    # removes src/landmark_stats.py at the end)
    stats = os.path.join(a.dir, "landmark_stats.py")
    if os.path.exists(stats):
        maschera_fissa = set(runpy.run_path(stats)["REFINE_LANDMARK_MASK"])
    else:
        from src.model_refine import REFINE_LANDMARK_MASK as maschera_fissa
        print(f"[WARN] {stats} not found: using the mask of src/landmark_stats.py or the fallback one")
    print(f"Fixed Stage 2 mask: {[NOMI[i] for i in sorted(maschera_fissa)]}")

    r = raccogli(stage1, stage2, loader, device, CONFIG)

    # Maximum correction (radial, mm)
    if getattr(stage2, "tipo_input", "feature") == "immagine":
        max_corr_mm = stage2.max_offset_mm * np.sqrt(2)
    else:
        # normalized MAX_OFFSET x mean image side x pixel size, on a typical image of this set
        dims = np.array([val_ds[k][0].shape[1:] for k in range(min(len(val_ds), 50))], float)
        ps = np.array([float(val_ds[k][4]) for k in range(min(len(val_ds), 50))])
        max_corr_mm = float(stage2.max_offset * np.sqrt(2) * np.mean(dims.mean(axis=1) * ps))

    report(r, max_corr_mm, maschera_fissa)

    nome = "analisi_stage2.csv" if not a.stage2 else \
        f"analisi_stage2_{os.path.splitext(os.path.basename(a.stage2))[0]}.csv"
    out = os.path.join(a.dir, nome)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img", "landmark"] + [k for k in r if k not in ("img", "lm")])
        for k in range(len(r["img"])):
            w.writerow([r["img"][k], NOMI[r["lm"][k]]] + [r[c][k] for c in r if c not in ("img", "lm")])
    print(f"\nPer-point data saved to {out}")
    print(f"Compare the ways of combining Stage 1 and Stage 2 with:  python strategie_stage2.py {out}")


if __name__ == "__main__":
    main()
