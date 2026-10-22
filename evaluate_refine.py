import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

from src.datasets import Datasets, crea_insieme
from src.heatmap_utils import HEATMAP_TRAIN_SIZE
from src.inferenza import (carica_stage1 as _carica_stage1, carica_stage2, collate_base,
                           decodifica as _decodifica, predici_con_tta as _predici_con_tta,
                           applica_stage2 as _applica_stage2, errori_mm, seleziona_punti)
from src.incertezza import (momenti_heatmap, cov_in_mm, sigma_radiale, analisi_incertezza,
                            stampa_analisi_incertezza, FONTI_INCERTEZZA)

try:
    from src.parametri_stage2 import SELEZIONE_STAGE2, FUSIONE_STAGE2, SOGLIA_SIGMA_TTA_MM
except ImportError:   # fallback if the file is missing
    from src.inferenza import FUSIONE_DEFAULT as FUSIONE_STAGE2
    SELEZIONE_STAGE2, SOGLIA_SIGMA_TTA_MM = "fusione", 0.25

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
    "stage1_path":       "esperimenti/fixed/modello_heatmap.pth",
    "stage2_path":       "esperimenti/fixed/modello_refine.pth",
    "sdr_thresholds_mm": [2.0, 2.5, 3.0, 4.0],
    "hrnet_channels":    [192, 384, 768, 1536],
    "usa_tta":           True,
    "tta_angoli":        [-5.0, -2.5, 0.0, 2.5, 5.0],
    # How Stage 2 is used:
    #   "fusione"    (default) every point = weighted average of Stage 1 and Stage 2,
    #                weights from their uncertainties (fusione_stage2, needs usa_tta).
    #                Best in strategie_stage2.py: accuracy <2mm 86.6% -> 88.4%.
    #   "incertezza" Stage 2 only where sigma_tta > soglia_sigma_tta_mm (needs usa_tta)
    #   "tutti" | "maschera" (REFINE_LANDMARK_MASK)
    # For "incertezza" the threshold
    # depends on the Stage 1 model: analisi_stage2.py prints the sigma_tta
    # quantiles and the best threshold. For "probabilistic" + image-based Stage 2,
    # 0.25 mm (~ the most uncertain 50% of the points) was chosen on the even
    # validation images and confirmed on the odd ones (MRE 1.099 -> 1.060 mm).
    # Values from src/parametri_stage2.py (measured on the first run, kept fixed)
    "selezione_stage2":    SELEZIONE_STAGE2,
    "fusione_stage2":      FUSIONE_STAGE2,
    "soglia_sigma_tta_mm": SOGLIA_SIGMA_TTA_MM,
    # Uncertainty analysis + calibration file used by predict_heatmap.py
    "analizza_incertezza": True,
    "copertura":           0.9,
    "calibrazione_path":   "calibrazione_incertezza.json",
    # Images to evaluate on: "validation" (also writes the calibration) or a test
    # set of the official split: "test", "test_aariz", "test_isbi1", "test_isbi2"
    # (uses the calibration computed on validation, never rewrites it)
    "insieme":             "validation",
}

NOMI = ["A","ANS","B","Me","N","Or","Pog","PNS","Pn","R","S","Ar","Co","Gn","Go","Po","LPM","LIT","LMT","UIT","UIA","LIA","Ls","Li","Pog`","Sn"]

# Anatomical category of each landmark index (0-based)
CATEGORIE = {
    "skeletal":    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],   # incl. Po (15)
    "dental":      [16, 17, 18, 19, 20, 21],                              # incl. LIA (21)
    "soft tissue": [8, 22, 23, 24, 25],                                   # Pn (pronasale), lips, Pog`, Sn
}
CATEGORIA_DI = {i: cat for cat, idx in CATEGORIE.items() for i in idx}


def collate_fn(batch, img_size=512):
    """Resizes images to img_size, normalizes landmark coordinates to
    [0, 1] and keeps the original size and the original images, needed by
    the image-based Stage 2 (see src/inferenza.collate_base)."""
    return collate_base(batch, img_size, con_originali=True)


def carica_stage1(config, device):
    """Loads the Stage 1 checkpoint (see src/inferenza.py)."""
    return _carica_stage1(config, device)


def decodifica(pred_raw, config, stage1=None):
    """Decodes raw Stage 1 heatmaps into normalized (x, y) coordinates."""
    return _decodifica(pred_raw, stage1, config["heatmap_size"])


def calcola_mre_sdr(stage1, stage2, loader, device, config, raccogli_incertezza=False):
    """
    Runs inference over the whole validation loader and computes, per
    landmark and globally: Mean Radial Error (MRE, in mm) and Successful
    Detection Rate (SDR) at each threshold in sdr_thresholds_mm.
    Errors are converted to mm with the original width AND height of each
    image (x and y scaled separately) and its own pixel size.

    With raccogli_incertezza=True also returns, as a 5th value, one record
    per (image, landmark) with the error and the uncertainty of every source
    (heatmap spread, TTA spread, total), for analisi_incertezza().
    """
    stage1.eval()
    if stage2 is not None:
        stage2.eval()
    errori_per_landmark = [[] for _ in range(config["num_landmarks"])]
    record = {"img": [], "lm": [], "err": [], **{f"sigma_{f}": [] for f in FONTI_INCERTEZZA}}
    n_img = 0
    with torch.no_grad():
        for immagini, punti, mask, _, pixel_sizes, dims, originali in loader:
            immagini, punti, mask = immagini.to(device), punti.to(device), mask.to(device)
            pixel_sizes, dims = pixel_sizes.to(device), dims.to(device)

            out = _predici_con_tta(stage1, immagini, config)
            B, N = mask.shape
            pred_coords = out["coords"][:, :N, :]
            if stage2 is not None:
                sel = seleziona_punti(config.get("selezione_stage2", "tutti"), B, N, device,
                                      cov_tta=out["cov_tta"] if config.get("usa_tta", True) else None,
                                      dims=dims, pixel_size=pixel_sizes,
                                      soglia_mm=config.get("soglia_sigma_tta_mm"))
                fusione = config.get("fusione_stage2") if config.get("selezione_stage2") == "fusione" else None
                sigma1 = sigma_radiale(cov_in_mm(out["cov_tta"], dims, pixel_sizes))[:, :N] if fusione else None
                pred_coords = _applica_stage2(stage2, pred_coords, x_out=out["x_out"], originali=originali,
                                              dims=dims, pixel_size=pixel_sizes, seleziona=sel,
                                              fusione=fusione, sigma1_mm=sigma1)
            err = errori_mm(pred_coords, punti, dims, pixel_sizes)

            if raccogli_incertezza:
                _, cov_hm, _ = momenti_heatmap(out["pred_raw"][:, 0::3], stage1.heatmap_mode,
                                               stage1.griglia, temperatura=1.0)
                var_hm  = sigma_radiale(cov_in_mm(cov_hm, dims, pixel_sizes)) ** 2
                var_tta = sigma_radiale(cov_in_mm(out["cov_tta"], dims, pixel_sizes)) ** 2

            for b in range(B):
                for i in range(N):
                    if mask[b, i].item() != 1.0:
                        continue
                    errori_per_landmark[i].append(err[b, i].item())
                    if raccogli_incertezza:
                        record["img"].append(n_img + b)
                        record["lm"].append(i)
                        record["err"].append(err[b, i].item())
                        record["sigma_heatmap"].append(var_hm[b, i].sqrt().item())
                        record["sigma_tta"].append(var_tta[b, i].sqrt().item())
                        record["sigma_totale"].append((var_hm[b, i] + var_tta[b, i]).sqrt().item())
            n_img += B

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
    if raccogli_incertezza:
        return mre, mre_lm, sdr, sdr_lm, record
    return mre, mre_lm, sdr, sdr_lm


def stampa_risultati(titolo, mre, mre_lm, sdr, sdr_lm):
    """Prints a formatted MRE/SDR report: global metrics, metrics broken
    down by anatomical category (skeletal/dental/soft tissue), and a
    per-landmark table."""
    print(f"\n{'='*55}")
    print(titolo)
    print(f"{'='*55}")
    print(f"Global MRE: {mre:.2f} mm")
    for cat, idx in CATEGORIE.items():
        vals = [mre_lm[i] for i in idx if i < len(mre_lm) and mre_lm[i] is not None]
        if vals:
            nomi = ",".join(NOMI[i] for i in idx)
            print(f"MRE {cat:12s}: {np.mean(vals):.2f} mm  ({nomi})")
    print("\nSDR per threshold (mm):")
    for s, v in sdr.items():
        print(f"  within {s}mm: {v:.1f}%")

    soglie = list(sdr.keys())
    # Per-landmark table header
    hdr_sdr = "  ".join(f"SDR{s:g}" for s in soglie)
    print(f"\n{'Landmark':<6} {'Cat':12}  {'MRE':>6}  {hdr_sdr}")
    print("-" * (6 + 14 + 8 + len(soglie) * 10))
    for i, err in enumerate(mre_lm):
        cat  = CATEGORIA_DI.get(i, "")
        nome = NOMI[i] if i < len(NOMI) else str(i+1)
        if err is not None:
            sdr_vals = "  ".join(
                f"{sdr_lm[i][s]:>6.1f}%" if sdr_lm[i] else " " * 7
                for s in soglie
            )
            print(f"  {nome:<5} [{cat:12s}]  {err:>5.2f}mm  {sdr_vals}")
        else:
            print(f"  {nome:<5} [{cat:12s}]  — no valid samples")


def copertura_calibrazione_validation(record, path):
    """On a test set: coverage of the radii calibrated on validation (radius =
    q[landmark] x sigma), without re-calibrating on the test images."""
    import os
    import numpy as np
    if not path or not os.path.exists(path):
        print(f"\nNo calibration file ({path}): evaluate on validation first to create it.")
        return
    with open(path) as f:
        cal = json.load(f)
    lm, err = np.asarray(record["lm"]), np.asarray(record["err"], float)
    print(f"\n{'='*55}\nUNCERTAINTY — radii calibrated on validation (target {cal['copertura']:.0%})\n{'='*55}")
    for fonte, dati in cal["fonti"].items():
        if f"sigma_{fonte}" not in record:
            continue
        q = np.array([np.nan if v is None else v for v in dati["q"]], float)
        raggio = q[lm] * np.asarray(record[f"sigma_{fonte}"], float)
        ok = ~np.isnan(raggio)
        print(f"[{fonte}]  coverage on this set: {(err[ok] <= raggio[ok]).mean():.1%} | mean radius {raggio[ok].mean():.2f} mm")


def salva_calibrazione(path, risultati, stage1, config):
    """Writes the per-landmark conformal factors used by predict_heatmap.py."""
    dati = {
        "copertura": risultati["copertura"],
        "heatmap_mode": stage1.heatmap_mode,
        "stage1_path": config["stage1_path"],
        "usa_tta": config.get("usa_tta", True),
        "nomi": NOMI,
        "fonti": {f: {"q": r["q"], "spearman": r["spearman"],
                      "copertura_test": r["copertura_test"]}
                  for f, r in risultati["fonti"].items()},
    }
    with open(path, "w") as f:
        json.dump(dati, f, indent=2)
    print(f"\nCalibration saved to {path} (used by predict_heatmap.py)")


def main():
    """CLI entry point: evaluates Stage 1 alone and Stage 1+Stage 2 on the
    Aariz+ISBI validation split, printing MRE/SDR reports, the improvement
    from adding Stage 2 and the analysis of the uncertainty estimates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"TTA: {CONFIG['tta_angoli'] if CONFIG['usa_tta'] else 'off'}"
          + (" + flip + photometric" if CONFIG["usa_tta"] else ""))

    stage1 = carica_stage1(CONFIG, device)
    stage2 = carica_stage2(CONFIG["stage2_path"], CONFIG["num_landmarks"], device)
    if stage2 is None:
        print("Stage 2 not found — evaluating Stage 1 only")
    else:
        modo = CONFIG["selezione_stage2"]
        dettaglio = {"incertezza": f" (sigma_tta > {CONFIG['soglia_sigma_tta_mm']} mm)",
                     "fusione": f" (weighted average of Stage 1 and Stage 2, {CONFIG.get('fusione_stage2')})"}
        print(f"Stage 2 mode: {modo}{dettaglio.get(modo, '')}")

    dataset = Datasets(aariz_path=CONFIG["aariz_path"], isbi_path=CONFIG["isbi_path"])
    val_ds = crea_insieme(dataset, CONFIG.get("insieme", "validation"), CONFIG["val_split"], seed=42)
    print(f"Evaluated on: {CONFIG.get('insieme', 'validation')} ({len(val_ds)} images)")
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, collate_fn=partial(collate_fn, img_size=CONFIG["img_size"]))

    incert = CONFIG.get("analizza_incertezza", True)
    ris1 = calcola_mre_sdr(stage1, None, val_loader, device, CONFIG, raccogli_incertezza=incert)
    mre1, mre_lm1, sdr1, sdr_lm1 = ris1[:4]
    stampa_risultati("STAGE 1 — heatmap only", mre1, mre_lm1, sdr1, sdr_lm1)
    record = ris1[4] if incert else None

    if stage2 is not None:
        ris2 = calcola_mre_sdr(stage1, stage2, val_loader, device, CONFIG, raccogli_incertezza=incert)
        mre2, mre_lm2, sdr2, sdr_lm2 = ris2[:4]
        stampa_risultati("STAGE 1 + STAGE 2", mre2, mre_lm2, sdr2, sdr_lm2)
        print(f"\n{'='*55}\nIMPROVEMENT\n{'='*55}")
        print(f"MRE: {mre1:.2f} → {mre2:.2f} mm  ({mre1-mre2:+.2f} mm)")
        for s in CONFIG["sdr_thresholds_mm"]:
            d = sdr2[s] - sdr1[s]
            print(f"SDR {s}mm: {sdr1[s]:.1f}% → {sdr2[s]:.1f}%  ({d:+.1f}%)")
        if incert:
            record = ris2[4]   # calibrate on the final pipeline output

    if incert and CONFIG.get("insieme", "validation") == "validation":
        risultati = analisi_incertezza(record, CONFIG["num_landmarks"], copertura=CONFIG["copertura"])
        stampa_analisi_incertezza(risultati, NOMI)
        salva_calibrazione(CONFIG["calibrazione_path"], risultati, stage1, CONFIG)
    elif incert:
        copertura_calibrazione_validation(record, CONFIG["calibrazione_path"])


if __name__ == "__main__":
    main()
