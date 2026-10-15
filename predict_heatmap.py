import torch, cv2, numpy as np, sys, os, json

from src.heatmap_utils import HEATMAP_TRAIN_SIZE
from src.inferenza import carica_stage1, carica_stage2, predici_con_tta, applica_stage2, seleziona_punti
from src.incertezza import momenti_heatmap, cov_in_mm, sigma_radiale

USE_VOTING = False   # False = soft-argmax (stable), True = regression voting

CONFIG = {
    "num_landmarks":  26,   # 19 original + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
    "img_size":       512,
    "heatmap_size":   HEATMAP_TRAIN_SIZE,
    "stage1_path":    "modello_heatmap.pth",
    "stage2_path":    "modello_refine.pth",
    "usa_stage2":     True,
    # Which points Stage 2 corrects (see evaluate_refine.py): "tutti" | "maschera" | "incertezza"
    "selezione_stage2":    "incertezza",
    "soglia_sigma_tta_mm": 0.45,
    "usa_tta":        True,
    "tta_angoli":     [-5.0, -2.5, 0.0, 2.5, 5.0],
    "hrnet_channels": [192, 384, 768, 1536],
    # Pixel size of the input radiograph (mm/px), to express the uncertainty in mm.
    # Can also be given as the 2nd command-line argument.
    "pixel_size_mm":  0.1,
    # Written by evaluate_refine.py; turns the uncertainty into a radius that
    # contains the true point with the calibrated probability (e.g. 90%)
    "calibrazione_path": "calibrazione_incertezza.json",
    "fonte_incertezza":  "totale",    # "heatmap" | "tta" | "totale"
    "mostra_finestra":   True,        # False on servers without a display
    "salva_overlay":     True,        # saves <image>_pred.png next to the image
}

NOMI_LANDMARK = [
    "A","ANS","B","Me","N","Or","Pog","PNS","Pn","R",
    "S","Ar","Co","Gn","Go","Po","LPM","LIT","LMT",
    "UIT","UIA","LIA",
    "Ls","Li","Pog`","Sn",
]


def carica_modelli(config, device):
    """Loads both Stage 1 (heatmap) and, if available and enabled, Stage 2
    (refinement) models and returns them ready for inference (eval mode)."""
    stage1 = carica_stage1(config, device)
    stage2 = carica_stage2(config["stage2_path"], config["num_landmarks"], device) if config["usa_stage2"] else None
    if stage2 is None:
        print("Stage 2 not available.")
    return stage1, stage2


def carica_calibrazione(config, stage1):
    """Per-landmark conformal factors from evaluate_refine.py, or None."""
    path = config.get("calibrazione_path")
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        cal = json.load(f)
    if cal.get("heatmap_mode") != stage1.heatmap_mode or cal.get("stage1_path") != config["stage1_path"]:
        print("[WARN] the calibration file was computed for another Stage 1 checkpoint: ignored. "
              "Re-run evaluate_refine.py.")
        return None
    if cal.get("usa_tta", True) != config["usa_tta"]:
        print("[WARN] calibration computed with a different TTA setting: radii may be off.")
    fonte = config.get("fonte_incertezza", "totale")
    if fonte not in cal["fonti"]:
        return None
    return {"copertura": cal["copertura"], "q": cal["fonti"][fonte]["q"], "fonte": fonte}


def predici(stage1, stage2, img_path, device, config):
    """
    Full single-image inference pipeline: loads the image, resizes it to
    the training resolution, runs Stage 1 (with or without TTA), optionally
    refines the result with Stage 2, and rescales the coordinates back to
    the original image size.
    Returns (original_grayscale_image, landmarks_in_pixels [N, 2],
    uncertainty dict with the per-landmark sigmas in mm and heatmap peaks).
    """
    img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h_orig, w_orig = img_orig.shape
    img_resized = cv2.resize(img_orig, (config["img_size"], config["img_size"]))
    img_tensor  = (torch.from_numpy(img_resized).float()
                   .unsqueeze(0).unsqueeze(0) / 255.0).to(device)

    out = predici_con_tta(stage1, img_tensor, config, usa_voting=USE_VOTING)
    coords_norm = out["coords"]
    dims = torch.tensor([[w_orig, h_orig]], dtype=torch.float32, device=device)
    ps   = torch.tensor([config["pixel_size_mm"]], dtype=torch.float32, device=device)
    if stage2 is not None:
        sel = seleziona_punti(config["selezione_stage2"], 1, coords_norm.shape[1], device,
                              cov_tta=out["cov_tta"] if config["usa_tta"] else None,
                              dims=dims, pixel_size=ps, soglia_mm=config["soglia_sigma_tta_mm"])
        originale = torch.from_numpy(img_orig).float().unsqueeze(0) / 255.0
        coords_norm = applica_stage2(stage2, coords_norm, x_out=out["x_out"], originali=[originale],
                                     dims=dims, pixel_size=ps, seleziona=sel)
    _, cov_hm, picco = momenti_heatmap(out["pred_raw"][:, 0::3], stage1.heatmap_mode, stage1.griglia)
    incertezza = {
        "heatmap": sigma_radiale(cov_in_mm(cov_hm, dims, ps))[0].cpu().numpy(),
        "tta":     sigma_radiale(cov_in_mm(out["cov_tta"], dims, ps))[0].cpu().numpy(),
        "picco":   picco[0].cpu().numpy(),
    }
    incertezza["totale"] = np.sqrt(incertezza["heatmap"] ** 2 + incertezza["tta"] ** 2)

    coords = coords_norm[0].cpu().numpy()
    coords[:, 0] *= w_orig; coords[:, 1] *= h_orig
    return img_orig, coords, incertezza


def disegna_landmark(img_gray, landmarks, raggi_px=None, titolo="Landmark prediction",
                     salva_in=None, mostra=True):
    """Draws the predicted landmarks (with their names) over the image and,
    if given, a circle of radius raggi_px[i] around each one (calibrated
    uncertainty). Optionally saves the result and/or displays it in an
    OpenCV window, scaled down if larger than 1000px."""
    img_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w    = img_vis.shape[:2]
    raggio  = max(6, int(min(h, w) / 150))
    font_scale = max(0.4, min(h, w) / 1500)
    spessore   = max(1, int(min(h, w) / 600))
    for i, pt in enumerate(landmarks):
        x, y = int(pt[0]), int(pt[1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        if raggi_px is not None and np.isfinite(raggi_px[i]):
            cv2.circle(img_vis, (x, y), int(round(raggi_px[i])), (60, 60, 255), spessore, cv2.LINE_AA)
        cv2.circle(img_vis, (x, y), raggio, (0, 220, 255), -1)
        cv2.circle(img_vis, (x, y), raggio, (0, 0, 0), 1)
        nome = NOMI_LANDMARK[i] if i < len(NOMI_LANDMARK) else str(i + 1)
        cv2.putText(img_vis, nome, (x + raggio + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255),
                    spessore, cv2.LINE_AA)
    if salva_in:
        cv2.imwrite(salva_in, img_vis)
        print(f"Overlay saved to {salva_in}")
    if mostra:
        scala     = min(1.0, 1000 / max(h, w))
        img_show  = cv2.resize(img_vis, (int(w * scala), int(h * scala)))
        cv2.imshow(titolo, img_show)
        print("Press any key to close..."); cv2.waitKey(0); cv2.destroyAllWindows()


def main():
    """CLI entry point: predicts landmarks for a single image (path given
    as the first argument, or a default sample; optional 2nd argument: pixel
    size in mm/px) and shows/prints the result with its uncertainty."""
    img_path = "data/Aariz/test/Cephalograms/cks2ip8fp29yl0yuf6ry9266i.png"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG["pixel_size_mm"] = float(sys.argv[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modo   = "regression_voting" if USE_VOTING else "soft-argmax"
    print(f"Device: {device} | Backbone: HRNet-W48 | Mode: {modo}")
    stage1, stage2 = carica_modelli(CONFIG, device)
    cal = carica_calibrazione(CONFIG, stage1)
    img_orig, landmarks, inc = predici(stage1, stage2, img_path, device, CONFIG)

    sigma = inc[cal["fonte"] if cal else CONFIG["fonte_incertezza"]]
    raggi_mm = None
    if cal:
        raggi_mm = np.array([q * s if q is not None else np.nan for q, s in zip(cal["q"], sigma)])

    print(f"\nPredicted coordinates (pixels) — pixel size {CONFIG['pixel_size_mm']} mm/px")
    intest = f"  {'':5s}  {'x':>8s} {'y':>8s}  {'sigma_hm':>8s} {'sigma_tta':>9s}"
    if cal:
        intest += f"  {'r' + format(cal['copertura'], '.0%'):>7s}"
    print(intest + "   (mm)")
    for i, pt in enumerate(landmarks):
        nome = NOMI_LANDMARK[i] if i < len(NOMI_LANDMARK) else str(i + 1)
        riga = f"  {nome:5s}  {pt[0]:8.1f} {pt[1]:8.1f}  {inc['heatmap'][i]:8.2f} {inc['tta'][i]:9.2f}"
        if cal:
            riga += f"  {raggi_mm[i]:7.2f}"
        print(riga)
    if cal:
        print(f"\nr{cal['copertura']:.0%}: radius that contains the true point with probability "
              f"~{cal['copertura']:.0%} (calibrated on the validation set, source '{cal['fonte']}').")
    else:
        print("\nNo calibration file: sigma columns are raw spreads, not calibrated radii. "
              "Run evaluate_refine.py to create it.")

    raggi_px = raggi_mm / CONFIG["pixel_size_mm"] if raggi_mm is not None else None
    salva = os.path.splitext(img_path)[0] + "_pred.png" if CONFIG["salva_overlay"] else None
    disegna_landmark(img_orig, landmarks, raggi_px, salva_in=salva, mostra=CONFIG["mostra_finestra"])


if __name__ == "__main__":
    main()
