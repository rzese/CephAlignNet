"""
deviazione_errori.py — mean radial error ± standard deviation, overall and per
landmark, for Stage 1 only and for the final pipeline (Stage 1 + Stage 2 with
the fusion parameters of src/parametri_stage2.py).

Reads the CSV written by analisi_stage2.py (no models, no GPU):
    python deviazione_errori.py esperimenti_run2/probabilistic/analisi_stage2.csv
    python deviazione_errori.py <csv> --salva errori_sd.csv     # also writes the table
"""
import argparse
import csv

import numpy as np

from src.parametri_stage2 import FUSIONE_STAGE2
from strategie_stage2 import carica, errori, peso_fusione


def riga(nome, e1, ef):
    return [nome, len(e1), e1.mean(), e1.std(ddof=1), ef.mean(), ef.std(ddof=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="analisi_stage2.csv")
    ap.add_argument("--salva", help="write the table to this CSV")
    a = ap.parse_args()

    d = carica(a.csv)
    p = FUSIONE_STAGE2
    w = peso_fusione(d, "d", p["c"], p["p"], p["floor_mm"])
    e1 = np.linalg.norm(d["e1"], axis=1)       # Stage 1 only
    ef = errori(d, w, "d")                     # final (fusion)

    # landmarks in the order they appear in the CSV
    nomi = list(dict.fromkeys(d["landmark"]))
    tabella = [riga(n, e1[d["landmark"] == n], ef[d["landmark"] == n]) for n in nomi]
    tabella.append(riga("TOTALE", e1, ef))

    print(f"Radial error (mm), mean ± SD | fusion {p}")
    print(f"{'landmark':8s} {'n':>5s}   {'Stage 1':>15s}   {'final (fusion)':>15s}")
    for nome, n, m1, s1, mf, sf in tabella:
        if nome == "TOTALE":
            print("-" * 50)
        print(f"{nome:8s} {n:5d}   {m1:6.2f} ± {s1:5.2f}   {mf:6.2f} ± {sf:5.2f}")

    if a.salva:
        with open(a.salva, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["landmark", "n", "mre_stage1", "sd_stage1", "mre_finale", "sd_finale"])
            for r in tabella:
                w_csv.writerow([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]])
        print(f"\nSaved to {a.salva}")


if __name__ == "__main__":
    main()
