"""
accuratezza_sd.py — accuracy (share of points within 2 / 2.5 / 3 / 4 mm) as
mean ± standard deviation over the images, overall and per landmark, for all
points and, optionally, only for the points with sigma_tta <= threshold.

For every image the accuracy is computed on its own landmarks; the table gives
the mean and the SD of those per-image values. Per landmark each image has one
point (inside or outside the threshold), so there the SD is sqrt(p(1-p)) and
only says how close p is to 50%.

Final pipeline errors (fusion with src/parametri_stage2.py); --stage1 for Stage 1.
Reads the CSV written by analisi_stage2.py (no models, no GPU):

    python accuratezza_sd.py esperimenti_leo/probabilistic/analisi_stage2.csv
    python accuratezza_sd.py <csv> --soglia 0.6 --salva accuratezza_sd.csv
"""
import argparse
import csv

import numpy as np

from src.parametri_stage2 import FUSIONE_STAGE2
from strategie_stage2 import SDR_MM, carica, errori, peso_fusione


def per_immagine(e, img, sel):
    """[n_images, len(SDR_MM)] accuracy (%) of each image on the selected points."""
    righe = []
    for i in np.unique(img[sel]):
        ei = e[sel & (img == i)]
        righe.append([(ei < t).mean() * 100 for t in SDR_MM])
    return np.array(righe).reshape(-1, len(SDR_MM))


def media_sd(a):
    if len(a) == 0:
        return [np.nan] * (2 * len(SDR_MM))
    sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])
    return [v for k in range(a.shape[1]) for v in (a[:, k].mean(), sd[k])]


def fmt(v):
    return "".join(f"{v[2 * k]:7.1f} ± {v[2 * k + 1]:4.1f}" for k in range(len(SDR_MM)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="analisi_stage2.csv")
    ap.add_argument("--soglia", type=float, help="also: only points with sigma_tta <= threshold (mm)")
    ap.add_argument("--stage1", action="store_true", help="errors of Stage 1 alone instead of the final ones")
    ap.add_argument("--salva", help="write the table to this CSV")
    a = ap.parse_args()

    d = carica(a.csv)
    p = FUSIONE_STAGE2
    e = np.linalg.norm(d["e1"], axis=1) if a.stage1 else \
        errori(d, peso_fusione(d, "d", p["c"], p["p"], p["floor_mm"]), "d")
    img, lm = d["img"], d["landmark"]
    nomi = list(dict.fromkeys(lm))
    gruppi = [(n, lm == n) for n in nomi] + [("TOTALE", np.ones(len(e), bool))]
    insiemi = [("all points", np.ones(len(e), bool))]
    if a.soglia is not None:
        insiemi.append((f"sigma_tta <= {a.soglia:.2f} mm", d["sigma_tta"] <= a.soglia))

    intest = "".join(f"{'<' + format(t, 'g') + ' mm (%)':>14s}" for t in SDR_MM)
    righe_csv = []
    for nome_ins, tieni in insiemi:
        print("=" * 76)
        print(f"Accuracy, mean ± SD over images — {nome_ins} — "
              f"{'Stage 1 only' if a.stage1 else 'final (fusion)'}")
        print("=" * 76)
        print(f"{'landmark':8s} {'images':>6s}{intest}")
        for nome, sel in gruppi:
            pi = per_immagine(e, img, sel & tieni)
            v = media_sd(pi)
            if nome == "TOTALE":
                print("-" * 76)
            print(f"{nome:8s} {len(pi):6d}{fmt(v)}")
            righe_csv.append([nome_ins, nome, len(pi)] + [f"{x:.2f}" for x in v])
        print()

    if a.salva:
        with open(a.salva, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["punti", "landmark", "n_immagini"] +
                       [c for t in SDR_MM for c in (f"acc{t:g}_media", f"acc{t:g}_sd")])
            w.writerows(righe_csv)
        print(f"Saved to {a.salva}")


if __name__ == "__main__":
    main()
