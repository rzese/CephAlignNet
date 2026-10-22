"""
dataset_unife.py — UNIFE tracings (data/tracciati), used as an EXTERNAL test set.

Folder layout (one folder per case, inside "Primo Invio", "Secondo Invio", ...):
    <id>.jpg                          the radiograph
    <id>-1.tif                        the same image as saved by ImageJ (fallback)
    <id> Results.csv                  landmark coordinates in mm (ImageJ, origin
                                      top-left): ";X;Y" header and one row per
                                      landmark name; in a few cases the names are
                                      replaced by the row number 1..35 (same order)
    <id> Distance in pixels .docx     the ImageJ scale: "Distance in pixels: 102
                                      Known distance: 10.00 ... Unit of length: mm"
    <id> scala.png                    screenshot of the scale (not needed)

Two scales per case:
  - CSV units -> pixels: the ImageJ calibration saved in the .tif (resolution in
    pixels per unit, "unit=mm", or "unit=inch" in one case), i.e. the one ImageJ
    used to write the coordinates; with no calibration in the .tif the CSV is in
    pixels (one case, written with dots as thousands separators: "1.155.000");
  - mm per pixel of the radiograph (to measure errors in mm): the .tif
    calibration when it is in mm, otherwise the .docx scale (ruler on the
    cephalostat; in a few cases the .docx is empty or has a typo, "110.01.02",
    and the .tif is used).

Only the landmarks with the same definition as in Aariz/ISBI are mapped (22 of
26). Not mapped: R (UNIFE has R3, a different ramus point), Gn (absent), LPM and
LMT (UNIFE marks the first molars, U6/L6 mesial and distal cusps: equivalence
with the Aariz definitions not verified). Unmapped landmarks are [0, 0] and are
masked out downstream, like the ISBI ones.

Returns (image [1, H, W] float in [0, 1], landmarks [26, 2] pixels, pixel_size).
"""
import csv
import os
import re
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets import formatta_campione

# Order of the 35 UNIFE landmarks (used for the CSVs that have row numbers instead of names)
ORDINE_UNIFE = ["Sor", "S", "Ba", "Po", "Na", "Or", "Pt", "ANS", "PNS", "A", "R3", "Co", "Ar", "Go",
                "Me", "Pog", "B", "U6D", "U6M", "L6D", "L6M", "U1Ap", "U1Inc", "L1Ap", "L1Inc",
                "G", "Na'", "Pn", "Sn", "A'", "UL", "LL", "B'", "Pog'", "Me'"]

# UNIFE name -> canonical index (see NOMI_26 in dataset_tracciati.py)
MAPPA_UNIFE = {
    "A": 0, "ANS": 1, "B": 2, "Me": 3, "Na": 4, "Or": 5, "Pog": 6, "PNS": 7, "Pn": 8,
    "S": 10, "Ar": 11, "Co": 12, "Go": 14, "Po": 15,
    "L1Inc": 17, "U1Inc": 19, "U1Ap": 20, "L1Ap": 21,
    "UL": 22, "LL": 23, "Pog'": 24, "Sn": 25,
}
NUM_LANDMARKS = 26


def calibrazione_tif(tif_path):
    """(pixels per unit, unit) of the ImageJ calibration stored in a .tif, or (None, None)."""
    try:
        from PIL import Image
        with Image.open(tif_path) as im:
            tag = im.tag_v2
            risoluzione = tag.get(282)                     # XResolution
            descr = str(tag.get(270, ""))                  # ImageJ description: "...unit=mm..."
    except Exception:
        return None, None
    unita = re.search(r"unit=([A-Za-z]+)", descr)
    if not risoluzione or not unita:
        return None, None
    return float(risoluzione), unita.group(1).lower()


def numero(s):
    """ImageJ number, also with dots as thousands separators ("1.155.000" -> 1155.0)."""
    s = s.strip().replace(",", ".")
    if s.count(".") > 1:
        intero, dec = s.rsplit(".", 1)
        s = intero.replace(".", "") + "." + dec
    return float(s)


def leggi_scala(docx_path):
    """mm per pixel from the ImageJ 'Distance in pixels' document, or None."""
    try:
        xml = zipfile.ZipFile(docx_path).read("word/document.xml").decode("utf8", "ignore")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    testo = re.sub(r"\s+", " ", " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml)))
    px = re.search(r"Distance in pixels:\s*([\d.,]+)", testo)
    mm = re.search(r"Known distance:\s*([\d.,]+)", testo)
    try:
        px, mm = float(px.group(1).replace(",", ".")), float(mm.group(1).replace(",", "."))
    except (AttributeError, ValueError):
        return None
    return mm / px if px > 0 else None


def leggi_coordinate(csv_path):
    """{UNIFE name: (x_mm, y_mm)} from an ImageJ Results.csv (';' or ',' separated)."""
    with open(csv_path, encoding="utf8", errors="ignore") as f:
        testo = f.read()
    sep = ";" if testo.count(";") > testo.count(",") else ","
    righe = [r for r in csv.reader(testo.splitlines(), delimiter=sep) if len(r) >= 3]
    punti = {}
    for r in righe[1:]:
        nome = r[0].strip()
        if nome.isdigit():                                   # numbered rows: same order as ORDINE_UNIFE
            k = int(nome) - 1
            if not 0 <= k < len(ORDINE_UNIFE):
                continue
            nome = ORDINE_UNIFE[k]
        try:
            punti[nome] = (numero(r[1]), numero(r[2]))
        except ValueError:
            continue
    return punti


class UnifeDataset(Dataset):
    """UNIFE tracings as (image, landmarks in pixels, pixel_size)."""

    def __init__(self, base_path="data/tracciati"):
        self.base_path = base_path
        self.samples = []            # (case name, image path, csv path, mm per pixel, pixels per CSV unit)
        saltati = []
        for invio in sorted(os.listdir(base_path)) if os.path.isdir(base_path) else []:
            d_invio = os.path.join(base_path, invio)
            if not os.path.isdir(d_invio) or invio.startswith("."):
                continue
            for caso in sorted(os.listdir(d_invio)):
                d = os.path.join(d_invio, caso)
                if not os.path.isdir(d) or caso.startswith("."):
                    continue
                file = os.listdir(d)
                csv_f = next((f for f in file if f.endswith("Results.csv")), None)
                docx_f = next((f for f in file if f.endswith(".docx") and "Distance" in f), None)
                img_f = next((f for f in file if f.lower() == f"{caso}.jpg"), None) or \
                    next((f for f in file if f.lower().endswith(".tif")), None)
                tif_f = next((f for f in file if f.lower().endswith(".tif")), None)
                ris, unita = calibrazione_tif(os.path.join(d, tif_f)) if tif_f else (None, None)
                px_per_unita = ris if ris else 1.0                 # no calibration: CSV in pixels
                if unita in ("mm", "millimeter", "millimeters"):
                    ps = 1.0 / ris
                else:
                    ps = leggi_scala(os.path.join(d, docx_f)) if docx_f else None
                if not (csv_f and img_f and ps):
                    saltati.append(f"{invio}/{caso}")
                    continue
                self.samples.append((f"{invio}/{caso}", os.path.join(d, img_f), os.path.join(d, csv_f), ps, px_per_unita))
        print(f"[UnifeDataset] {len(self.samples)} cases | {len(saltati)} skipped"
              + (f" ({', '.join(saltati[:5])}{'...' if len(saltati) > 5 else ''})" if saltati else ""))

    def __len__(self):
        return len(self.samples)

    def punti_pixel(self, idx):
        """[26, 2] landmark coordinates in pixels (unmapped ones at [0, 0])."""
        _, _, csv_path, _, px_per_unita = self.samples[idx]
        coords = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
        for nome, (x, y) in leggi_coordinate(csv_path).items():
            if nome in MAPPA_UNIFE:
                coords[MAPPA_UNIFE[nome]] = (x * px_per_unita, y * px_per_unita)
        return coords

    def __getitem__(self, idx):
        _, img_path, _, ps, _ = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        return (torch.from_numpy(img).float().unsqueeze(0) / 255.0,
                torch.from_numpy(self.punti_pixel(idx)),
                torch.tensor(ps, dtype=torch.float32))


class UnifeValutazione(Dataset):
    """UNIFE in the common (image, points, mask, bbox, pixel_size) format of Datasets."""

    def __init__(self, base_path="data/tracciati"):
        self.unife = UnifeDataset(base_path)

    def __len__(self):
        return len(self.unife)

    def __getitem__(self, idx):
        return formatta_campione(self.unife[idx], 0.1, source="UNIFE")
