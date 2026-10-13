"""
Visual check of the ISBI/Cepha400 landmark mapping (see src/dataset_isbi.py).

Draws the 19 annotated points of one or more ISBI images, each labelled with
its file index (P1..P19) and with the name it gets under the "legacy" and
"standard" mappings, and saves the result as a PNG (no display needed).

    PYTHONPATH=. python Test/verifica_mappatura_isbi.py [data/Cepha400] [001 002 ...]

Look at, e.g., P5: under "legacy" it is Ar (articulare, behind the mandibular
ramus), under "standard" it is A (subspinale, in front of the maxilla). The
mapping whose names land on the right anatomy is the correct one: set it in
MAPPATURA_ISBI_DEFAULT and retrain.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset_isbi import parse_md_txt, MAPPATURE_ISBI, IMG_SUBFOLDERS
from src.dataset import SYMBOL_ORDER


def disegna(base, n, out_dir):
    sub = next(s for s, r in IMG_SUBFOLDERS if n in r)
    img = cv2.imread(os.path.join(base, "RawImage", sub, f"{n:03d}.bmp"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{n:03d}: image not found")
        return
    pts = (parse_md_txt(os.path.join(base, "AnnotationsByMD", "400_senior", f"{n:03d}.txt"))
           + parse_md_txt(os.path.join(base, "AnnotationsByMD", "400_junior", f"{n:03d}.txt"))) / 2
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for p, (x, y) in enumerate(pts, start=1):
        leg = MAPPATURE_ISBI["legacy"].get(p)
        std = MAPPATURE_ISBI["standard"].get(p)
        testo = (f"P{p} L:{SYMBOL_ORDER[leg] if leg is not None else '-'} "
                 f"S:{SYMBOL_ORDER[std] if std is not None else '-'}")
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(vis, testo, (int(x) + 12, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2, cv2.LINE_AA)
    out = os.path.join(out_dir, f"verifica_isbi_{n:03d}.png")
    cv2.imwrite(out, vis)
    print(f"saved {out}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "data/Cepha400"
    casi = [int(c) for c in sys.argv[2:]] or [1, 151, 301]
    for n in casi:
        disegna(base, n, ".")
    print("\nP-number: L = name under the 'legacy' mapping, S = name under the 'standard' mapping")
