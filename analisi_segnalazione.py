"""
analisi_segnalazione.py — can the uncertainty flag the points to be checked
by a clinician? Reads the CSV written by analisi_stage2.py (no GPU needed).

For each uncertainty score, the points are ranked from most to least
uncertain and the top X% is "flagged". Reported:
  - recall:    share of the large errors (> 2, 3, 4 mm) that end up flagged
  - precision: share of the flagged points that really have a large error
  - the same for a random flag (baseline: recall = X%)
Scores:
  sigma_tta, sigma_heatmap   raw uncertainty in mm
  *_rel                      uncertainty divided by the median of its landmark,
                             so that intrinsically hard landmarks (Co, R, ...)
                             are not always flagged just because they are hard
Also reports the image level: flagging the images with the highest mean
uncertainty, how many of the images with at least one error > 3 mm are caught.

    python analisi_segnalazione.py esperimenti/probabilistic/analisi_stage2.csv
"""
import csv
import sys

import numpy as np

QUOTE = [0.05, 0.10, 0.20]
SOGLIE_MM = [2.0, 3.0, 4.0]


def carica(path):
    with open(path) as f:
        righe = list(csv.DictReader(f))
    d = {k: np.array([r[k] for r in righe]) for k in righe[0]}
    for k in ("err1", "sigma_tta", "sigma_heatmap"):
        d[k] = d[k].astype(float)
    d["img"] = d["img"].astype(int)
    return d


def relativo(sigma, landmark):
    out = np.empty_like(sigma)
    for lm in np.unique(landmark):
        sel = landmark == lm
        out[sel] = sigma[sel] / np.median(sigma[sel])
    return out


def report_punti(err, punteggi):
    print(f"\n{'='*72}\nPOINT LEVEL — {len(err)} points")
    for s in SOGLIE_MM:
        print(f"  errors > {s:.0f} mm: {(err > s).sum()} ({(err > s).mean():.1%})")
    print(f"{'='*72}")
    for nome, p in punteggi.items():
        ordine = np.argsort(-p)
        print(f"\n[{nome}]")
        print(f"  {'flagged':>8} " + " ".join(f"{'recall>'+format(s,'.0f')+'mm':>12} {'prec>'+format(s,'.0f')+'mm':>10}" for s in SOGLIE_MM))
        for q in QUOTE:
            k = int(round(q * len(err)))
            flag = np.zeros(len(err), bool)
            flag[ordine[:k]] = True
            celle = []
            for s in SOGLIE_MM:
                grandi = err > s
                rec = (flag & grandi).sum() / max(grandi.sum(), 1)
                prec = (flag & grandi).sum() / max(k, 1)
                celle.append(f"{rec:12.1%} {prec:10.1%}")
            print(f"  {q:8.0%} " + " ".join(celle))
    print(f"\n  (random flag: recall = share flagged; precision = share of large errors overall)")


def report_immagini(d, punteggi, soglia=3.0):
    imgs = np.unique(d["img"])
    peggiore = np.array([d["err1"][d["img"] == i].max() for i in imgs])
    problematiche = peggiore > soglia
    print(f"\n{'='*72}\nIMAGE LEVEL — {len(imgs)} images, {problematiche.sum()} "
          f"({problematiche.mean():.1%}) with at least one error > {soglia:.0f} mm\n{'='*72}")
    for nome, p in punteggi.items():
        media = np.array([p[d["img"] == i].mean() for i in imgs])
        ordine = np.argsort(-media)
        riga = []
        for q in QUOTE + [0.30]:
            k = int(round(q * len(imgs)))
            flag = np.zeros(len(imgs), bool)
            flag[ordine[:k]] = True
            riga.append(f"top {q:.0%}: {(flag & problematiche).sum() / max(problematiche.sum(), 1):5.1%}")
        print(f"  [{nome:18s}] share of problematic images caught — " + " | ".join(riga))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "esperimenti/probabilistic/analisi_stage2.csv"
    d = carica(path)
    lm = d["landmark"]
    punteggi = {
        "sigma_tta": d["sigma_tta"],
        "sigma_heatmap": d["sigma_heatmap"],
        "sigma_tta_rel": relativo(d["sigma_tta"], lm),
        "sigma_heatmap_rel": relativo(d["sigma_heatmap"], lm),
    }
    report_punti(d["err1"], punteggi)
    report_immagini(d, punteggi)


if __name__ == "__main__":
    main()
