"""
coda_errori.py — how much of the error standard deviation comes from a few
large errors, and what those errors are.

Reads the CSV written by analisi_stage2.py (no models, no GPU). Errors are those
of the final pipeline (Stage 1 + Stage 2 fused with src/parametri_stage2.py);
with --stage1, those of Stage 1 alone.

    python coda_errori.py esperimenti_run2/probabilistic/analisi_stage2.csv
    python coda_errori.py <csv> --peggiori 50 --revisione 0.10
    python coda_errori.py <csv> --salva-immagini casi_peggiori    # needs the data/ folder

Sections:
  A. per landmark: mean ± SD, median, 90th/95th percentile, max, SD without the
     worst 1% of points, share of the variance due to that 1%
  B. the worst points (image, landmark, error, sigma_tta, flagged or not)
  C. how many large errors the uncertainty flag (review the most uncertain
     X% of points by sigma_tta) would catch
  D. images with the most large errors (anomalous images?)

With --salva-immagini DIR the worst points (as many as --peggiori) are also
drawn, one PNG each: on the left a zoom around the point, on the right the whole
radiograph. Green = annotation (mean of the two orthodontists), red = Stage 1,
yellow = final (fusion). The file name of the radiograph is printed in section B.
"img" in the CSV is the position in the evaluated set (--insieme, the same one
given to analisi_stage2.py), mapped back to the Aariz / ISBI file here.
"""
import argparse
import os
from collections import Counter

import numpy as np

from src.parametri_stage2 import FUSIONE_STAGE2
from strategie_stage2 import carica, errori, peso_fusione


def statistiche(e, quota_peggiori=0.01):
    """Robust statistics of a set of radial errors."""
    k = max(1, int(np.ceil(quota_peggiori * len(e))))
    ordinati = np.sort(e)
    resto = ordinati[:-k]
    scarti2 = (e - e.mean()) ** 2
    peggiori = np.argsort(e)[-k:]
    return {
        "n": len(e), "media": e.mean(), "sd": e.std(ddof=1), "mediana": np.median(e),
        "p90": np.percentile(e, 90), "p95": np.percentile(e, 95), "max": e.max(),
        "sd_senza": resto.std(ddof=1), "media_senza": resto.mean(),
        "quota_var": scarti2[peggiori].sum() / scarti2.sum(),
    }


def carica_validation(aariz_path, isbi_path, insieme="validation"):
    """Evaluated set in the same order as analisi_stage2.py (CSV column 'img' = position in it)."""
    from src.datasets import Datasets, crea_insieme
    dataset = Datasets(aariz_path=aariz_path, isbi_path=isbi_path)
    return dataset, crea_insieme(dataset, insieme, 0.2, seed=42)


def nome_file(dataset, val_ds, img):
    """'aariz/<file>' or 'isbi/<file>' of validation image number img."""
    g = val_ds.indices[img]
    if hasattr(val_ds.dataset, "unife"):                 # external UNIFE test set
        return f"unife/{val_ds.dataset.unife.samples[g][0]}"
    if g < dataset.len_aariz:
        fname, split, ext = dataset.aariz.samples[g]
        return f"aariz/{split}/{fname}{ext}"
    return f"isbi/{os.path.basename(dataset.isbi.samples[g - dataset.len_aariz][1])}"


def disegna_caso(val_ds, img, lm_idx, nome_lm, e1_mm, ef_mm, titolo, path, lato_px=600):
    """Zoom around the point + whole radiograph, side by side."""
    import cv2
    immagine, punti, _, _, ps = val_ds[img]
    ps = float(ps)
    base = (immagine[0].numpy() * 255).clip(0, 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    vero = punti[lm_idx].numpy().astype(float)
    s1 = vero + np.asarray(e1_mm) / ps
    fin = vero + np.asarray(ef_mm) / ps
    VERDE, ROSSO, GIALLO = (0, 200, 0), (0, 0, 255), (0, 220, 255)

    # zoom: at least 30 mm, or large enough to contain all three points
    centro = (vero + s1 + fin) / 3
    meta_mm = max(15.0, 1.4 * max(np.linalg.norm(s1 - centro), np.linalg.norm(fin - centro),
                                  np.linalg.norm(vero - centro)) * ps + 5.0)
    meta = int(round(meta_mm / ps))
    H, W = base.shape[:2]
    x0, y0 = int(centro[0]) - meta, int(centro[1]) - meta
    pad = cv2.copyMakeBorder(base, meta, meta, meta, meta, cv2.BORDER_CONSTANT, value=0)
    zoom = pad[y0 + meta:y0 + 3 * meta, x0 + meta:x0 + 3 * meta].copy()
    scala = lato_px / zoom.shape[0]
    zoom = cv2.resize(zoom, (lato_px, lato_px), interpolation=cv2.INTER_LINEAR)
    in_zoom = lambda p: (int(round((p[0] - x0) * scala)), int(round((p[1] - y0) * scala)))
    cv2.line(zoom, in_zoom(vero), in_zoom(fin), GIALLO, 1, cv2.LINE_AA)
    for p, col in ((s1, ROSSO), (fin, GIALLO), (vero, VERDE)):
        cv2.circle(zoom, in_zoom(p), 6, col, 2, cv2.LINE_AA)
    # 5 mm scale bar
    barra = int(round(5.0 / ps * scala))
    cv2.line(zoom, (15, lato_px - 15), (15 + barra, lato_px - 15), (255, 255, 255), 2)
    cv2.putText(zoom, "5 mm", (15, lato_px - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # whole radiograph
    s = lato_px / H
    intera = cv2.resize(base, (int(round(W * s)), lato_px), interpolation=cv2.INTER_AREA)
    in_int = lambda p: (int(round(p[0] * s)), int(round(p[1] * s)))
    for p, col in ((s1, ROSSO), (fin, GIALLO), (vero, VERDE)):
        cv2.circle(intera, in_int(p), 5, col, 2, cv2.LINE_AA)
    cv2.rectangle(intera, in_int((x0, y0)), in_int((x0 + 2 * meta, y0 + 2 * meta)), (255, 255, 255), 1)

    tela = np.concatenate([zoom, np.full((lato_px, 8, 3), 255, np.uint8), intera], axis=1)
    testa = np.full((56, tela.shape[1], 3), 255, np.uint8)
    cv2.putText(testa, titolo, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(testa, f"{nome_lm}: green = annotation, red = Stage 1 ({np.linalg.norm(e1_mm):.1f} mm), "
                       f"yellow = final ({np.linalg.norm(ef_mm):.1f} mm)",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.concatenate([testa, tela], axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="analisi_stage2.csv")
    ap.add_argument("--stage1", action="store_true", help="errors of Stage 1 alone instead of the final ones")
    ap.add_argument("--peggiori", type=int, default=30, help="how many worst points to list")
    ap.add_argument("--revisione", type=float, default=0.10,
                    help="fraction of the most uncertain points (sigma_tta) sent to manual review")
    ap.add_argument("--salva-immagini", metavar="DIR",
                    help="draw the worst points (zoom + whole radiograph) into DIR; needs the data")
    ap.add_argument("--aariz-path", default="data/Aariz")
    ap.add_argument("--isbi-path", default="data/Cepha400")
    ap.add_argument("--insieme", default="validation", help="the set given to analisi_stage2.py --insieme")
    a = ap.parse_args()

    d = carica(a.csv)
    e1 = np.linalg.norm(d["e1"], axis=1)
    p = FUSIONE_STAGE2
    w = peso_fusione(d, "d", p["c"], p["p"], p["floor_mm"])
    ef = errori(d, w, "d")
    vett_fin = d["e1"] + w[:, None] * d["d"]       # final error vector (mm)
    dataset = val_ds = None
    if a.salva_immagini:
        dataset, val_ds = carica_validation(a.aariz_path, a.isbi_path, a.insieme)
        os.makedirs(a.salva_immagini, exist_ok=True)
    e = e1 if a.stage1 else ef
    quale = "Stage 1 only" if a.stage1 else f"final (fusion {p})"
    nomi = list(dict.fromkeys(d["landmark"]))
    sig = d["sigma_tta"]
    soglia_sig = np.quantile(sig, 1 - a.revisione)
    segnalato = sig >= soglia_sig

    # ── A ──
    print("=" * 104)
    print(f"A. Radial error per landmark (mm) — {quale}")
    print("=" * 104)
    print(f"{'landmark':8s} {'n':>5s} {'mean ± SD':>14s} {'median':>7s} {'p90':>6s} {'p95':>6s} {'max':>6s}"
          f"   {'without worst 1%: mean ± SD':>28s}  {'variance from worst 1%':>22s}")
    righe = [(n, statistiche(e[d["landmark"] == n])) for n in nomi] + [("TOTALE", statistiche(e))]
    for nome, s in righe:
        if nome == "TOTALE":
            print("-" * 104)
        print(f"{nome:8s} {s['n']:5d} {s['media']:6.2f} ± {s['sd']:5.2f} {s['mediana']:7.2f} {s['p90']:6.2f} "
              f"{s['p95']:6.2f} {s['max']:6.2f}   {s['media_senza']:17.2f} ± {s['sd_senza']:5.2f}  {s['quota_var']:21.0%}")
    print("'without worst 1%': per landmark the worst ceil(1% of n) points are removed (3 of 280); "
          "for TOTALE the worst 1% of all points.")

    # ── B ──
    print()
    print("=" * 104)
    print(f"B. The {a.peggiori} worst points — flagged = sigma_tta among the {a.revisione:.0%} most uncertain "
          f"(sigma_tta >= {soglia_sig:.2f} mm)")
    print("=" * 104)
    print(f"{'img':>5s} {'landmark':8s} {'err final':>9s} {'err S1':>7s} {'sigma_tta':>9s} {'sigma pct':>9s} {'flagged':>8s}"
          + ("   file" if val_ds is not None else ""))
    rango_sig = np.argsort(np.argsort(sig)) / (len(sig) - 1)
    for pos, k in enumerate(np.argsort(e)[::-1][:a.peggiori], 1):
        img, nome_lm = int(d["img"][k]), d["landmark"][k]
        riga = (f"{img:5d} {nome_lm:8s} {ef[k]:9.2f} {e1[k]:7.2f} {sig[k]:9.2f} "
                f"{rango_sig[k]:9.0%} {'yes' if segnalato[k] else '-':>8s}")
        if val_ds is not None:
            f = nome_file(dataset, val_ds, img)
            riga += f"   {f}"
            from analisi_stage2 import NOMI
            lm_idx = NOMI.index(nome_lm)
            titolo = (f"#{pos}  img {img}  {f}  |  sigma_tta {sig[k]:.2f} mm "
                      f"({'flagged' if segnalato[k] else 'not flagged'})")
            disegna_caso(val_ds, img, lm_idx, nome_lm, d["e1"][k], vett_fin[k], titolo,
                         os.path.join(a.salva_immagini, f"{pos:02d}_{nome_lm}_img{img}_{ef[k]:.1f}mm.png"))
        print(riga)
    if val_ds is not None:
        print(f"Images saved in {a.salva_immagini}/")

    # ── C ──
    print()
    print("=" * 104)
    print(f"C. Manual review of the {a.revisione:.0%} most uncertain points: how many large errors it catches")
    print("=" * 104)
    print(f"{'error >':>8s} {'points':>7s} {'caught':>7s} {'% caught':>9s} {'random review':>14s}")
    for t in (2.0, 3.0, 4.0, 5.0):
        grandi = e > t
        if grandi.any():
            print(f"{t:6.1f}mm {grandi.sum():7d} {(grandi & segnalato).sum():7d} "
                  f"{(grandi & segnalato).sum() / grandi.sum():9.0%} {a.revisione:14.0%}")
    restanti = e[~segnalato]
    print(f"Points NOT sent to review ({(~segnalato).sum()}): mean ± SD {restanti.mean():.2f} ± {restanti.std(ddof=1):.2f} mm"
          f" | within 2 mm {(restanti < 2).mean():.1%}")

    # ── D ──
    print()
    print("=" * 104)
    print("D. Images with the most errors > 3 mm (many on one image = anomalous image, not a model problem)")
    print("=" * 104)
    conta = Counter(d["img"][e > 3.0].tolist())
    tot_grandi = sum(conta.values())
    print(f"{tot_grandi} errors > 3 mm on {len(conta)} images of {len(set(d['img'].tolist()))}")
    for img, c in conta.most_common(10):
        sel = (d["img"] == img) & (e > 3.0)
        print(f"  img {img:4d}: {c:2d} errors > 3 mm  ({', '.join(f'{l} {x:.1f}' for l, x in zip(d['landmark'][sel], e[sel]))})")


if __name__ == "__main__":
    main()
