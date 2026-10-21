"""
risultati_esclusione.py — results on all points vs results keeping only the
points whose Stage 1 uncertainty is low (sigma_tta <= threshold), overall and
per landmark, with how many points are excluded.

Errors are those of the final pipeline (Stage 1 + Stage 2 fused with
src/parametri_stage2.py); with --stage1, those of Stage 1 alone.
Reads the CSV written by analisi_stage2.py (no models, no GPU):

    python risultati_esclusione.py esperimenti_run2/probabilistic/analisi_stage2.csv --soglia 0.3
    python risultati_esclusione.py <csv> --soglia 0.3 0.57 --salva esclusione.csv

The excluded points are the ones a clinician would check by hand: the table
says how good the automatic result is on the points that are kept.
"""
import argparse
import csv

import numpy as np

from src.parametri_stage2 import FUSIONE_STAGE2
from strategie_stage2 import SDR_MM, carica, errori, peso_fusione


def metriche(e):
    """n, mean, SD, SDR at each threshold (NaN if no points)."""
    if len(e) == 0:
        return [0, np.nan, np.nan] + [np.nan] * len(SDR_MM)
    sd = e.std(ddof=1) if len(e) > 1 else 0.0
    return [len(e), e.mean(), sd] + [(e < t).mean() * 100 for t in SDR_MM]


def fmt(m):
    n, media, sd, *sdr = m
    if n == 0:
        return f"{'—':>14s}" + "".join(f"{'—':>7s}" for _ in sdr)
    return f"{media:6.2f} ± {sd:5.2f}" + "".join(f"{s:6.1f}%" for s in sdr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="analisi_stage2.csv")
    ap.add_argument("--soglia", type=float, nargs="+", default=[0.3],
                    help="keep only points with sigma_tta <= threshold (mm); several thresholds allowed")
    ap.add_argument("--stage1", action="store_true", help="errors of Stage 1 alone instead of the final ones")
    ap.add_argument("--salva", help="write all the tables to this CSV")
    a = ap.parse_args()

    d = carica(a.csv)
    p = FUSIONE_STAGE2
    e = np.linalg.norm(d["e1"], axis=1) if a.stage1 else \
        errori(d, peso_fusione(d, "d", p["c"], p["p"], p["floor_mm"]), "d")
    sig = d["sigma_tta"]
    quale = "Stage 1 only" if a.stage1 else f"final (fusion {p})"
    nomi = list(dict.fromkeys(d["landmark"]))
    gruppi = [(n, d["landmark"] == n) for n in nomi] + [("TOTALE", np.ones(len(e), bool))]
    sdr_int = "".join(f"{'<' + format(t, 'g') + 'mm':>7s}" for t in SDR_MM)
    righe_csv = []

    # ── overview: how much each threshold excludes and what it gains ──
    print("=" * 96)
    print(f"Overview — {quale} — {len(e)} points")
    print("=" * 96)
    print(f"{'kept':22s} {'excluded':>15s}   {'MRE ± SD':>14s}{sdr_int}")
    print(f"{'all points':22s} {0:7d} ({0:5.1%})   {fmt(metriche(e))}")
    for s in sorted(set(a.soglia) | {0.2, 0.3, 0.4, 0.57, 0.8, 1.0}):
        tieni = sig <= s
        print(f"{'sigma_tta <= ' + format(s, '.2f') + ' mm':22s} {(~tieni).sum():7d} ({(~tieni).mean():5.1%})   "
              f"{fmt(metriche(e[tieni]))}")

    # ── per landmark, for each requested threshold ──
    for s in a.soglia:
        tieni = sig <= s
        print()
        print("=" * 128)
        print(f"All points vs only sigma_tta <= {s:.2f} mm — {quale}")
        print("=" * 128)
        print(f"{'':8s} {'':5s} {'ALL POINTS':^42s}   {'EXCLUDED':^14s}   {'KEPT (sigma_tta <= ' + format(s, '.2f') + ')':^48s}")
        print(f"{'landmark':8s} {'n':>5s} {'MRE ± SD':>14s}{sdr_int}   {'n':>5s} {'%':>7s}   "
              f"{'n':>5s} {'MRE ± SD':>14s}{sdr_int}")
        for nome, sel in gruppi:
            tutti, tenuti = metriche(e[sel]), metriche(e[sel & tieni])
            escl = int((sel & ~tieni).sum())
            if nome == "TOTALE":
                print("-" * 128)
            print(f"{nome:8s} {tutti[0]:5d} {fmt(tutti)}   {escl:5d} {escl / tutti[0]:7.1%}   "
                  f"{tenuti[0]:5d} {fmt(tenuti)}")
            righe_csv.append([s, nome, tutti[0], escl, tenuti[0]] +
                             [f"{v:.4f}" for v in tutti[1:]] + [f"{v:.4f}" for v in tenuti[1:]])

    if a.salva:
        sdr_col = [f"sdr{t:g}" for t in SDR_MM]
        with open(a.salva, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["soglia_sigma_tta_mm", "landmark", "n_tutti", "n_esclusi", "n_tenuti",
                        "mre_tutti", "sd_tutti"] + [c + "_tutti" for c in sdr_col] +
                       ["mre_tenuti", "sd_tenuti"] + [c + "_tenuti" for c in sdr_col])
            w.writerows(righe_csv)
        print(f"\nSaved to {a.salva}")


if __name__ == "__main__":
    main()
