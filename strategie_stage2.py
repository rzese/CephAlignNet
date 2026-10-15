"""
strategie_stage2.py — which way of combining Stage 1 and Stage 2 gives the
highest ACCURACY (share of points within 2 / 2.5 / 3 / 4 mm)?

Reads the CSV written by analisi_stage2.py (needs the columns e1x..s2_tta,
i.e. an image-based Stage 2). Numpy only: no models, no GPU.

Every point has a Stage 1 error vector e1 (mm) and a Stage 2 proposed shift d
(mm). A strategy decides a weight w in [0, 1] per point and the final error is
|e1 + w * d|:  w = 0 keeps Stage 1, w = 1 takes Stage 2, values in between
average the two estimates. Strategies compared (each also with Stage 2 TTA):

  soglia       w = 1 where sigma_tta (Stage 1 uncertainty) > threshold
  fusione      w = s1^p / (s1^p + (c * s2)^p): weight given by how reliable
               each stage is (s1 = Stage 1 uncertainty, s2 = Stage 2
               uncertainty); p = 2 is the classic inverse-variance fusion
  classific.   logistic regression predicting "Stage 2 improves this point"
               from the uncertainties, the size of the shift and the
               landmark; used as a hard switch (w = 1 if p > threshold) or as
               a soft weight (w = p)

Parameters are chosen to maximize the mean accuracy over the four thresholds
on one half of the images (even / odd image index) and measured on the other
half, then the other way round (2-fold cross-validation). The table reports
the mean of the two held-out halves, so it is an honest estimate for new
images. "Ideal" rows (best of Stage 1 / Stage 2 per point, knowing the truth)
are an unreachable upper bound.

    python strategie_stage2.py esperimenti/probabilistic/analisi_stage2.csv
    python strategie_stage2.py <csv> --salva src/parametri_stage2.py   # also write the parameters

With --salva the fusion parameters and the sigma_tta threshold are re-fitted on
ALL the images and written to a module read by evaluate_refine.py and
predict_heatmap.py (src/parametri_stage2.py). Do it only when you want to
replace the measurement used by the following runs.
"""
import argparse
import csv
import datetime
import sys

import numpy as np

SDR_MM = [2.0, 2.5, 3.0, 4.0]
GUARDIA_MM = 6.0     # corrections larger than this (per axis) are discarded, as in applica_stage2
NUM_LANDMARKS = 26


# ─── data ──────────────────────────────────────────────────────────────────────
def carica(path):
    with open(path) as f:
        righe = list(csv.DictReader(f))
    if "d2x" not in righe[0]:
        sys.exit("This CSV has no Stage 2 vectors: re-run analisi_stage2.py (image-based Stage 2).")
    nomi_lm = sorted({r["landmark"] for r in righe})
    d = {"img": np.array([int(r["img"]) for r in righe]),
         "landmark": np.array([r["landmark"] for r in righe])}
    for k in ("e1x", "e1y", "d2x", "d2y", "d2tx", "d2ty", "s2_hm", "s2_tta", "sigma_tta", "sigma_heatmap"):
        d[k] = np.array([float(r[k]) for r in righe])
    d["lm"] = np.array([nomi_lm.index(x) for x in d["landmark"]])
    d["e1"] = np.stack([d["e1x"], d["e1y"]], 1)
    for nome, (kx, ky) in (("d", ("d2x", "d2y")), ("dt", ("d2tx", "d2ty"))):
        v = np.stack([d[kx], d[ky]], 1)
        d[nome] = np.where((np.abs(v) <= GUARDIA_MM).all(1, keepdims=True), v, 0.0)
    return d


def sottoinsieme(d, sel):
    return {k: v[sel] for k, v in d.items()}


# ─── metrics ───────────────────────────────────────────────────────────────────
def errori(d, w, chiave):
    return np.linalg.norm(d["e1"] + w[:, None] * d[chiave], axis=1)


def accuratezza(err):
    """Objective: mean accuracy over the SDR thresholds."""
    return float(np.mean([(err < t).mean() for t in SDR_MM]))


def punteggio(err):
    """What the parameter search maximizes: accuracy first; on ties, the
    lower MRE wins (the MRE term is smaller than one point of accuracy)."""
    return accuratezza(err) - 1e-5 * float(err.mean())


def metriche(err):
    return [err.mean()] + [(err < t).mean() for t in SDR_MM] + [accuratezza(err)]


# ─── strategies: each returns a function (data) -> w, fitted on the fit set ────
def strategia_costante(valore):
    def fit(_, __):
        return (lambda d: np.full(len(d["img"]), valore)), ""
    return fit


def fit_soglia(dfit, chiave):
    s = dfit["sigma_tta"]
    migliore, param = -1, None
    for t in np.quantile(s, np.linspace(0, 0.99, 100)):
        a = punteggio(errori(dfit, (s > t).astype(float), chiave))
        if a > migliore:
            migliore, param = a, t
    return (lambda d: (d["sigma_tta"] > param).astype(float)), f"sigma_tta > {param:.2f} mm"


def sigma2(d, chiave):
    # Stage 2 uncertainty: crop spread, plus TTA spread for the TTA version
    return np.sqrt(d["s2_hm"] ** 2 + (d["s2_tta"] ** 2 if chiave == "dt" else 0.0)) + 1e-6


def peso_fusione(d, chiave, c, p, f):
    s1 = np.sqrt(d["sigma_tta"] ** 2 + f ** 2) + 1e-6
    s2 = sigma2(d, chiave)
    return s1 ** p / (s1 ** p + (c * s2) ** p)


def fit_fusione(dfit, chiave):
    migliore, param = -1, None
    for p in (1.0, 2.0, 4.0):
        for f in (0.0, 0.1, 0.2, 0.4):
            for c in np.logspace(-2, 1, 31):
                a = punteggio(errori(dfit, peso_fusione(dfit, chiave, c, p, f), chiave))
                if a > migliore:
                    migliore, param = a, (c, p, f)
    c, p, f = param
    return (lambda d: peso_fusione(d, chiave, c, p, f)), f"c={c:.2f} p={p:g} floor={f:g} mm"


def caratteristiche(d, chiave, media=None, std=None):
    shift = np.linalg.norm(d[chiave], axis=1)
    cont = np.stack([np.log(d["sigma_tta"] + 1e-3), np.log(d["sigma_heatmap"] + 1e-3),
                     np.log(d["s2_hm"] + 1e-3), np.log(d["s2_tta"] + 1e-3),
                     np.log(shift + 0.05)], 1)
    if media is None:
        media, std = cont.mean(0), cont.std(0) + 1e-6
    onehot = np.eye(NUM_LANDMARKS)[d["lm"]]
    X = np.hstack([np.ones((len(cont), 1)), (cont - media) / std, onehot])
    return X, media, std


def regressione_logistica(X, y, l2=1.0, iterazioni=50):
    """L2-regularized logistic regression fitted with Newton's method."""
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2); reg[0] = 0.0
    for _ in range(iterazioni):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + reg * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg + 1e-6)
        passo = np.linalg.solve(H, g)
        w -= passo
        if np.abs(passo).max() < 1e-6:
            break
    return w


def fit_classificatore(dfit, chiave):
    X, media, std = caratteristiche(dfit, chiave)
    migliora = np.linalg.norm(dfit["e1"] + dfit[chiave], axis=1) < np.linalg.norm(dfit["e1"], axis=1)
    coef = regressione_logistica(X, migliora.astype(float))

    def prob(d):
        Xd, _, _ = caratteristiche(d, chiave, media, std)
        return 1 / (1 + np.exp(-Xd @ coef))

    pf = prob(dfit)
    migliore, scelta = punteggio(errori(dfit, pf, chiave)), ("morbido", None)
    for t in np.linspace(0.05, 0.95, 19):
        a = punteggio(errori(dfit, (pf > t).astype(float), chiave))
        if a > migliore:
            migliore, scelta = a, ("soglia", t)
    if scelta[0] == "morbido":
        return prob, "soft weight w = p"
    t = scelta[1]
    return (lambda d: (prob(d) > t).astype(float)), f"switch if p > {t:.2f}"


# ─── evaluation ────────────────────────────────────────────────────────────────
def valuta(d, fit, chiave):
    """2-fold cross-validation over even/odd images: mean held-out metrics."""
    pari = (d["img"] % 2) == 0
    risultati, descr = [], ""
    for sel_fit in (pari, ~pari):
        dfit, dtest = sottoinsieme(d, sel_fit), sottoinsieme(d, ~sel_fit)
        regola, testo = fit(dfit, chiave)
        risultati.append(metriche(errori(dtest, regola(dtest), chiave)))
        descr = descr or testo            # parameters chosen on the even half
    return np.mean(risultati, axis=0), descr


def scrivi_parametri(path, d, csv_path, tabella):
    """Re-fits fusion and threshold on all images and writes the module."""
    _, testo_fusione = fit_fusione(d, "d")
    # recover the chosen parameters from the description "c=.. p=.. floor=.. mm"
    parti = dict(x.split("=") for x in testo_fusione.replace(" mm", "").split())
    c, p, f = float(parti["c"]), float(parti["p"]), float(parti["floor"])
    _, testo_soglia = fit_soglia(d, "d")
    soglia = float(testo_soglia.split(">")[1].replace("mm", ""))
    corpo = "\n".join("  " + r for r in tabella)
    with open(path, "w") as fh:
        fh.write(f'''"""
How Stage 2 is combined with Stage 1 — fixed parameters, read by
evaluate_refine.py and predict_heatmap.py.

Written by strategie_stage2.py on {datetime.date.today()} from {csv_path}
(parameters re-fitted on all the images; table = 2-fold CV over even/odd images,
objective = mean accuracy within 2 / 2.5 / 3 / 4 mm):

{corpo}

Not re-tuned automatically after a new training. To measure again:
    python analisi_stage2.py --dir <experiment>
    python strategie_stage2.py <experiment>/analisi_stage2.csv --salva src/parametri_stage2.py
"""

# "fusione" | "incertezza" | "tutti" | "maschera"  (see evaluate_refine.py)
SELEZIONE_STAGE2 = "fusione"

# Fusion weight of Stage 2: w = s1^p / (s1^p + (c * s2)^p), s1 = sqrt(sigma_tta^2 + floor^2)
FUSIONE_STAGE2 = {{"c": {c:.3g}, "p": {p:g}, "floor_mm": {f:g}}}

# Used only with SELEZIONE_STAGE2 = "incertezza": Stage 2 where sigma_tta > threshold
SOGLIA_SIGMA_TTA_MM = {soglia:.3g}
''')
    print(f"\nParameters written to {path}: fusion c={c:.3g} p={p:g} floor={f:g} mm | threshold {soglia:.3g} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="esperimenti/probabilistic/analisi_stage2.csv")
    ap.add_argument("--salva", help="write the parameters (fitted on all images) to this module, "
                                    "e.g. src/parametri_stage2.py")
    a = ap.parse_args()
    path = a.csv
    d = carica(path)
    tabella = []
    print(f"{len(d['img'])} points, {len(np.unique(d['img']))} images | objective: mean accuracy "
          f"within {', '.join(format(t, 'g') for t in SDR_MM)} mm | 2-fold CV over even/odd images\n")
    intest = f"{'strategy':34s} {'MRE':>6} " + " ".join(f"{'<' + format(t, 'g') + 'mm':>7}" for t in SDR_MM) \
        + f" {'mean acc':>8}   parameters (chosen on even images)"
    print(intest)
    print("-" * len(intest))

    righe = [("Stage 1 only", strategia_costante(0.0), "d")]
    for chiave, suff in (("d", ""), ("dt", " + S2 TTA")):
        righe += [
            ("Stage 2 everywhere" + suff, strategia_costante(1.0), chiave),
            ("threshold on sigma_tta" + suff, fit_soglia, chiave),
            ("fusion (weighted average)" + suff, fit_fusione, chiave),
            ("classifier" + suff, fit_classificatore, chiave),
        ]
    for nome, fit, chiave in righe:
        m, descr = valuta(d, fit, chiave)
        riga = f"{nome:34s} {m[0]:6.3f} " + " ".join(f"{x:7.1%}" for x in m[1:5]) + f" {m[5]:8.2%}"
        print(f"{riga}   {descr}")
        if chiave == "d":
            tabella.append(riga)

    print("-" * len(intest))
    pari = (d["img"] % 2) == 0
    for chiave, suff in (("d", ""), ("dt", " + S2 TTA")):
        ideale = np.minimum(np.linalg.norm(d["e1"], axis=1), np.linalg.norm(d["e1"] + d[chiave], axis=1))
        m = np.mean([metriche(ideale[pari]), metriche(ideale[~pari])], axis=0)   # same averaging as above
        print(f"{'Ideal (best of the two)' + suff:34s} {m[0]:6.3f} " + " ".join(f"{x:7.1%}" for x in m[1:5])
              + f" {m[5]:8.2%}   unreachable upper bound")


    if a.salva:
        scrivi_parametri(a.salva, d, path, [f"{'strategy':34s} {'MRE':>6} "
                                            + " ".join(f"{'<' + format(t, 'g') + 'mm':>7}" for t in SDR_MM)
                                            + f" {'mean acc':>8}"] + tabella)


if __name__ == "__main__":
    main()
