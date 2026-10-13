import csv
import os
import re
import zipfile

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# ─── Landmarks shared across datasets ─────────────────────────────────────────
NOMI_26 = [
    "A",    # 0   
    "ANS",  # 1   
    "B",    # 2   
    "Me",   # 3  
    "N",    # 4   
    "Or",   # 5   
    "Pog",  # 6   
    "PNS",  # 7   
    "Pn",   # 8   
    "R",    # 9    
    "S",    # 10  
    "Ar",   # 11  
    "Co",   # 12  
    "Gn",   # 13  
    "Go",   # 14  
    "Po",   # 15  
    "LPM",  # 16  
    "LIT",  # 17 
    "LMT",  # 18  
    "UIT",  # 19  
    "UIA",  # 20  
    "LIA",  # 21  
    "Ls",   # 22  
    "Li",   # 23  
    "Pog`", # 24  
    "Sn",   # 25  
]
NUM_LANDMARKS = 26

# ─── Mapping from Excel landmark names → landmark index ──────────────────────
# CANONICAL INDICES (see SYMBOL_ORDER in dataset.py):
# 0=A, 1=ANS, 2=B, 3=Me, 4=N, 5=Or, 6=Pog, 7=PNS, 8=Pn, 9=R,
# 10=S, 11=Ar, 12=Co, 13=Gn, 14=Go, 15=Po, 16=LPM, 17=LIT, 18=LMT,
# 19=UIT, 20=UIA, 21=LIA,
# 22=Ls(UL), 23=Li(LL), 24=Pog`(soft tissue Pog), 25=Sn
#
# The clinics' Excel files are not standardized: the same landmark may be
# labelled with different names/abbreviations/capitalizations depending on
# the software or operator. NAME_MAP collects every variant encountered so
# far and maps it to the canonical index. If a new file uses an unmapped
# name, run this module with DEBUG=True to inspect the raw Excel content
# and add the missing variant here.
NAME_MAP: dict = {
    # ── A (0) ──────────────────────────────────────────
    "A": 0, "a": 0,
    "POINT A": 0, "Point A": 0, "point a": 0,
    "SUBSPINALE": 0, "Subspinale": 0, "Sub-spinale": 0,

    # ── ANS (1) ────────────────────────────────────────
    "ANS": 1, "Ans": 1, "ans": 1,
    "Spina Nasale Ant": 1, "SPINA NASALE ANT": 1,
    "Spina Nasale Anteriore": 1, "SPINA NASALE ANTERIORE": 1,
    "SpNA": 1, "SPNA": 1,

    # ── B (2) ──────────────────────────────────────────
    "B": 2, "b": 2,
    "POINT B": 2, "Point B": 2, "point b": 2,
    "SUPRAMENTALE": 2, "Supramentale": 2,

    # ── Me (3) ─────────────────────────────────────────
    "Me": 3, "ME": 3, "me": 3,
    "Menton": 3, "MENTON": 3,
    "Mento": 3, "MENTO": 3,

    # ── N (4) ──────────────────────────────────────────
    "N": 4, "n": 4,
    "Na": 4, "NA": 4,
    "Nasion": 4, "NASION": 4,
    "Nasionale": 4,

    # ── Or (5) ─────────────────────────────────────────
    "Or": 5, "OR": 5, "or": 5,
    "Orbitale": 5, "ORBITALE": 5,

    # ── Pog (6) ────────────────────────────────────────
    "Pog": 6, "POG": 6, "pog": 6,
    "Pg": 6, "PG": 6,
    "Pogonion": 6, "POGONION": 6,

    # ── PNS (7) ────────────────────────────────────────
    "PNS": 7, "Pns": 7, "pns": 7,
    "Spina Nasale Post": 7, "SPINA NASALE POST": 7,
    "Spina Nasale Posteriore": 7, "SPINA NASALE POSTERIORE": 7,
    "SpNP": 7, "SPNP": 7,

    # ── Pn (8) ─────────────────────────────────────────
    "Pn": 8, "PN": 8, "pn": 8,
    "Pronasale": 8, "PRONASALE": 8,
    "Ptn": 8,

    # ── R/Ramus (9) ────────────────────────────────────
    "R": 9, "r": 9,
    "Rhi": 9, "RHI": 9,
    "Rhinion": 9, "RHINION": 9,

    # ── S (10) ─────────────────────────────────────────
    "S": 10, "s": 10,
    "Se": 10, "SE": 10,
    "Sella": 10, "SELLA": 10,
    "Sella turcica": 10, "SELLA TURCICA": 10,

    # ── Ar (11) ────────────────────────────────────────
    "Ar": 11, "AR": 11, "ar": 11,
    "Articulare": 11, "ARTICULARE": 11,
    "Articolare": 11, "ARTICOLARE": 11,

    # ── Co (12) ────────────────────────────────────────
    "Co": 12, "CO": 12, "co": 12,
    "Con": 12,
    "Condylion": 12, "CONDYLION": 12,
    "Condilion": 12, "CONDILION": 12,
    "Condilo": 12, "CONDILO": 12,

    # ── Gn (13) ────────────────────────────────────────
    "Gn": 13, "GN": 13, "gn": 13,
    "Gnathion": 13, "GNATHION": 13,
    "Gnazione": 13,

    # ── Go (14) ────────────────────────────────────────
    "Go": 14, "GO": 14, "go": 14,
    "Gonion": 14, "GONION": 14,

    # ── Po (15) ────────────────────────────────────────
    "Po": 15, "PO": 15, "po": 15,
    "Por": 15,
    "Porion": 15, "PORION": 15,

    # ── LPM (16) ───────────────────────────────────────
    "LPM": 16, "Lpm": 16, "lpm": 16,
    "L PM": 16, "LP": 16,
    "Lower Premolar": 16, "Lower PM": 16,
    "I Premolare Inf": 16, "I Premolare Inferiore": 16,
    "Premolare Inf": 16, "Premolare Inferiore": 16,

    # ── LIT (17) ───────────────────────────────────────
    "LIT": 17, "Lit": 17, "lit": 17,
    "L IT": 17, "LI": 17,
    "Lower Incisor": 17, "Lower Incisor Tip": 17,
    "Incisivo Inf": 17, "Incisivo Inferiore": 17,
    "I Incisivo Inf": 17, "I Incisivo Inferiore": 17,
    "L1Inc": 17, "l1inc": 17, "L1 Inc": 17, "L1INC": 17,
    "L1": 17, "1IInf": 17, "Inc Inf": 17, "IncInf": 17,

    # ── LMT (18) ───────────────────────────────────────
    "LMT": 18, "Lmt": 18, "lmt": 18,
    "L MT": 18, "LM": 18,
    "Lower Molar": 18, "Lower Molar Tip": 18,
    "Molare Inf": 18, "Molare Inferiore": 18,
    "I Molare Inf": 18, "I Molare Inferiore": 18,
    "L6M": 18, "l6m": 18, "L6 M": 18, "L6M Mes": 18,
    "L6Mes": 18, "L6": 18, "6MolInf": 18, "Mol Inf": 18, "MolInf": 18,

    # ── UIT (19) — Upper Incisor Tip ───────────────────
    "UIT": 19, "Uit": 19, "uit": 19,
    "U IT": 19,
    "Upper Incisor Tip": 19, "UPPER INCISOR TIP": 19,
    "Incisivo Sup Tip": 19, "Punta Incisivo Sup": 19,
    "U1Inc": 19, "u1inc": 19, "U1 Inc": 19, "U1INC": 19,
    "U1": 19, "1ISup": 19, "Inc Sup": 19, "IncSup": 19,
    "Incisivo Superiore Punta": 19, "I Incisivo Sup": 19,

    # ── UIA (20) — Upper Incisor Apex ──────────────────
    "UIA": 20, "Uia": 20, "uia": 20,
    "U IA": 20,
    "Upper Incisor Apex": 20, "UPPER INCISOR APEX": 20,
    "Apice Incisivo Sup": 20, "Apex Incisivo Sup": 20,
    "U1Apex": 20, "u1apex": 20, "U1 Apex": 20,
    "U1Ap": 20, "u1ap": 20,
    "Apice Rad Sup": 20, "ApiceRadSup": 20,
    "Apice I Sup": 20, "Apice Radicolare Sup": 20,

    # ── LIA (21) — Lower Incisor Apex ──────────────────
    "LIA": 21, "Lia": 21, "lia": 21,
    "L IA": 21,
    "Lower Incisor Apex": 21, "LOWER INCISOR APEX": 21,
    "Apice Incisivo Inf": 21, "Apex Incisivo Inf": 21,
    "L1Apex": 21, "l1apex": 21, "L1 Apex": 21,
    "L1Ap": 21, "l1ap": 21,
    "Apice Rad Inf": 21, "ApiceRadInf": 21,
    "Apice I Inf": 21, "Apice Radicolare Inf": 21,

    # ── Ls / UL (22) — Labrale superius ────────────────
    "Ls": 22, "LS": 22, "ls": 22,
    "UL": 22, "ul": 22,
    "Labrale superius": 22, "Labrale Superius": 22, "LABRALE SUPERIUS": 22,
    "Upper Lip": 22, "UPPER LIP": 22, "Labbro Sup": 22, "Labbro Superiore": 22,
    "Labrale Sup": 22,

    # ── Li / LL (23) — Labrale inferius ────────────────
    "Li": 23, "LI_soft": 23,
    "LL": 23, "ll": 23,
    "Labrale inferius": 23, "Labrale Inferius": 23, "LABRALE INFERIUS": 23,
    "Lower Lip": 23, "LOWER LIP": 23, "Labbro Inf": 23, "Labbro Inferiore": 23,
    "Labrale Inf": 23,

    # ── Pog` (24) — Soft Tissue Pogonion ───────────────
    "Pog`": 24, "Pog'": 24, "pog'": 24, "POG'": 24,
    "Soft Tissue Pogonion": 24, "SOFT TISSUE POGONION": 24,
    "ST Pogonion": 24, "Pog ST": 24,
    "Pogonion Tessuti Molli": 24, "Pog TM": 24,

    # ── Sn (25) — Subnasale ────────────────────────────
    "Sn": 25, "SN": 25, "sn": 25,
    "Subnasale": 25, "SUBNASALE": 25,
    "Sub Nasale": 25, "SubNasale": 25,
    "Punto Subnasal": 25,
}

# Set to True to print the raw Excel content of every file being parsed —
# useful for discovering new landmark-name variants to add to NAME_MAP.
DEBUG = False

def _identifica_file(folder: str, case_id: str, use_thumbnail: bool = False):
    """
    Looks for the image file and the Excel annotation file inside a single
    case folder.

    use_thumbnail=True  -> return the "-T.jpg" thumbnail file instead of the
                            full-size image. Used for the TELE folders, where
                            the annotation coordinates are already expressed
                            in thumbnail pixels rather than full-image pixels.
    use_thumbnail=False -> original behaviour: return the full-size image.

    Returns (img_path, excel_path), either of which may be None if not found.
    """
    IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    EXCEL_EXTS = {".xlsx", ".xltx", ".xls"}

    try:
        all_files = [f for f in os.listdir(folder) if not f.startswith(".")]
    except OSError:
        return None, None

    img_thumb  = None   
    img_exact  = None   
    img_fallbk = None   
    excel_path = None

    case_id_l = case_id.lower()

    for fname in sorted(all_files):
        base, ext = os.path.splitext(fname)
        ext_l  = ext.lower()
        base_l = base.lower()

        if ext_l in EXCEL_EXTS and excel_path is None:
            excel_path = os.path.join(folder, fname)

        elif ext_l in IMG_EXTS:
            is_thumbnail = base_l.endswith("-t")
            if is_thumbnail:
                if img_thumb is None:
                    img_thumb = os.path.join(folder, fname)
            elif base_l == case_id_l:
                img_exact = os.path.join(folder, fname)
            elif img_fallbk is None:
                img_fallbk = os.path.join(folder, fname)

    if use_thumbnail:  # some folders require the thumbnail instead of the full image
        img_path = img_thumb or img_exact or img_fallbk
    else:
        img_path = img_exact or img_fallbk

    return img_path, excel_path


# ─── Per-case scale (pixels per mm) ───────────────────────────────────────────
# ImageJ exports landmark coordinates in calibrated units (mm, and in .xlsx/.xltx
# files multiplied by 1000). To turn them into pixels the scale of EACH
# radiograph is needed: it varies a lot between machines and scans (about
# 3.5–11.6 px/mm in the clinic data), so a single global coord_scale gives
# wrong labels for most cases. The scale is looked up, in order, in:
#   1. a table (CSV or Excel) with columns case_id, scale_px_per_mm
#      (e.g. coordinate_radiografie.xlsx);
#   2. a .docx in the case folder with the text of ImageJ's "Set Scale" dialog
#      ("Distance in pixels: ... Known distance: ...");
#   3. otherwise the legacy global coord_scale (with a warning).

def _numero(token):
    # Some exports use '.' also as thousands separator ("1.089.000" = 1089.000):
    # the last group is always the decimal part.
    parts = str(token).strip().replace(" ", "").replace(",", ".").split(".")
    if len(parts) > 1:
        return float("".join(parts[:-1]) + "." + parts[-1])
    return float(parts[0])


def carica_tabella_scale(path):
    """Reads a CSV/Excel table with columns 'case_id' and 'scale_px_per_mm'
    -> {case_id (lowercase): px_per_mm}."""
    righe = []
    if path.lower().endswith((".xlsx", ".xlsm", ".xltx")):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        wb.close()
        header = [str(h).strip() if h is not None else "" for h in rows[0]]
        righe = [dict(zip(header, r)) for r in rows[1:]]
    else:
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            campione = f.read(2048); f.seek(0)
            delim = ";" if campione.count(";") > campione.count(",") else ","
            righe = list(csv.DictReader(f, delimiter=delim))
    tabella = {}
    for r in righe:
        cid, scala = r.get("case_id"), r.get("scale_px_per_mm")
        if cid is None or scala in (None, ""):
            continue
        try:
            tabella[str(cid).strip().lower()] = _numero(scala)
        except ValueError:
            continue
    return tabella


def leggi_scala_docx(path):
    """px/mm from a .docx containing ImageJ's "Set Scale" text, or None."""
    try:
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))
    # Word sometimes splits digits across runs: "11 4 .0 04 4" -> "114.0044"
    dist  = re.search(r"Distance in pixels:\s*([\d .,]+?)\s*Known", text)
    known = re.search(r"Known distance:\s*([\d .,]+?)\s*(Pixel|Unit|$)", text)
    if not dist or not known:
        return None
    try:
        d, k = _numero(dist.group(1).replace(" ", "")), _numero(known.group(1).replace(" ", ""))
    except ValueError:
        return None
    return d / k if k else None


def _trova_docx(folder):
    try:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".docx") and not f.startswith((".", "~$")):
                return os.path.join(folder, f)
    except OSError:
        pass
    return None


# ─── Excel parser ──────────────────────────────────────────────────────────────
def _parse_excel(path: str, coord_scale: float = 1.0,
                 flip_y: bool = False, img_height: int = 0) -> dict:
    """
    Parses a single clinic Excel file and extracts whatever landmarks it can
    recognize. For every cell whose text matches a known name in NAME_MAP,
    it looks for the next two numeric values on the same row and treats them
    as the (x, y) coordinates of that landmark (optionally rescaled and
    y-flipped). Sheets are tried in order and the first one that yields any
    recognized landmark is used.

    Returns a dict {landmark_index: [x, y]} — only the landmarks actually
    found in the file are present.
    """
    
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl not found. Run: pip install openpyxl")

    found = {}
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as e:
        print(f"[TracciatiDataset] Could not open {path}: {e}")
        return found

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_rows = list(ws.iter_rows(values_only=True))
        sheet_found = {}

        if DEBUG:
            print(f"\n  [DEBUG] {os.path.basename(path)} | sheet '{sheet_name}'")
            for ri, row in enumerate(all_rows[:40]):
                non_none = [v for v in (row or []) if v is not None]
                if non_none:
                    print(f"  Row {ri:2d}: {row}")

        for row in all_rows:
            if not row:
                continue
            for col_i, cell in enumerate(row):
                if cell is None:
                    continue
                cell_s = str(cell).strip()

                lm_idx = (NAME_MAP.get(cell_s)
                          or NAME_MAP.get(cell_s.capitalize())
                          or NAME_MAP.get(cell_s.upper()))
                if lm_idx is None or lm_idx in sheet_found:
                    continue

                nums = []
                for other in row[col_i + 1:]:
                    try:
                        nums.append(float(other))
                    except (TypeError, ValueError):
                        pass
                    if len(nums) == 2:
                        break

                if len(nums) == 2:
                    x = nums[0] * coord_scale
                    y = nums[1] * coord_scale
                    if flip_y and img_height > 0:
                        y = img_height - y
                    sheet_found[lm_idx] = [x, y]

        if sheet_found:
            found = sheet_found
            break

    wb.close()
    return found


# ─── Dataset ─────────────────────────────────────────────────────────────────
class TracciatiDataset(Dataset):
    def __init__(self, base_path: str = "data/Tracciati",
                 subfolder_names=None,
                 pixel_size: float = 0.1,
                 coord_scale: float = 1.0,
                 flip_y: bool = False,
                 use_thumbnail: bool = False,
                 tabella_scale=None,
                 unita_excel: str = "auto"):
        """
        use_thumbnail=True  -> load the "-T.jpg" thumbnail file for the TELE
                                folders, where coordinates are already in
                                thumbnail pixels.
        use_thumbnail=False -> original behaviour (full-size image).

        tabella_scale: path of a CSV/Excel table (case_id, scale_px_per_mm) or
                       an already loaded {case_id: px_per_mm} dict. Cases with
                       a known scale (from the table or from a .docx in their
                       folder) are converted mm -> px with their own scale and
                       get pixel_size = 1/scale; the others fall back to
                       coord_scale / pixel_size.
        unita_excel:   unit of the Excel values for cases with a known scale:
                       "mm1000" (ImageJ .xlsx/.xltx, mm x 1000), "mm", or
                       "auto" (mm1000 if the median value is > 3000, else mm).

        Scans base_path/<subfolder>/<case_id>/ for every case, keeping only
        those where both an image and an Excel file were found, and builds
        the self.samples list used by __getitem__.
        """
        self.pixel_size    = pixel_size
        self.coord_scale   = coord_scale
        self.flip_y        = flip_y
        self.use_thumbnail = use_thumbnail
        self.unita_excel   = unita_excel
        self.samples: list = []

        if isinstance(tabella_scale, str):
            tabella_scale = carica_tabella_scale(tabella_scale) if os.path.exists(tabella_scale) else None
            if tabella_scale is None:
                print("[TracciatiDataset] WARNING: scale table not found")
        tabella_scale = {str(k).lower(): v for k, v in (tabella_scale or {}).items()}

        if not os.path.isdir(base_path):
            print(f"[TracciatiDataset] WARNING: folder not found: {base_path}")
            return

        if subfolder_names is None:
            subfolder_names = sorted(
                d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")
            )

        n_ok, n_skip = 0, 0
        origine_scala = {"tabella": 0, "docx": 0, "fallback": 0}
        for sub in subfolder_names:
            sub_path = os.path.join(base_path, sub)
            if not os.path.isdir(sub_path):
                continue

            case_ids = sorted(
                d for d in os.listdir(sub_path)
                if os.path.isdir(os.path.join(sub_path, d)) and not d.startswith(".")
            )

            for cid in case_ids:
                case_path = os.path.join(sub_path, cid)
                img_path, excel_path = _identifica_file(case_path, cid, use_thumbnail=self.use_thumbnail)

                if img_path is None:
                    print(f"[TracciatiDataset] SKIP {case_path}: image not found")
                    n_skip += 1
                    continue
                if excel_path is None:
                    print(f"[TracciatiDataset] SKIP {case_path}: Excel file not found")
                    n_skip += 1
                    continue

                scala = tabella_scale.get(cid.lower())
                if scala is not None:
                    origine_scala["tabella"] += 1
                else:
                    docx = _trova_docx(case_path)
                    scala = leggi_scala_docx(docx) if docx else None
                    origine_scala["docx" if scala else "fallback"] += 1
                self.samples.append((img_path, excel_path, scala))
                n_ok += 1

        print(f"[TracciatiDataset] {n_ok} samples | {n_skip} skipped | {base_path}")
        if n_ok:
            print(f"  scale per case: {origine_scala['tabella']} from table, "
                  f"{origine_scala['docx']} from .docx, {origine_scala['fallback']} with the "
                  f"global coord_scale={coord_scale} (pixel_size={pixel_size}mm/px)")
            if origine_scala["fallback"]:
                print("  WARNING: with the global coord_scale the labels are only right for "
                      "images at the assumed resolution — provide a scale table (tabella_scale).")
            if not DEBUG:
                print("  Set DEBUG=True to inspect the Excel content.")

    def __len__(self):
        """Number of valid (image, Excel) case pairs found on disk."""
        return len(self.samples)

    def _in_pixel(self, found, scala):
        """Raw Excel values -> pixels, and the pixel size of this case."""
        if scala is None:
            conv = {k: [x * self.coord_scale, y * self.coord_scale] for k, (x, y) in found.items()}
            return conv, self.pixel_size
        unita = self.unita_excel
        if unita == "auto":
            valori = [abs(v) for xy in found.values() for v in xy]
            unita = "mm1000" if valori and float(np.median(valori)) > 3000 else "mm"
        div = 1000.0 if unita == "mm1000" else 1.0
        conv = {k: [x / div * scala, y / div * scala] for k, (x, y) in found.items()}
        return conv, 1.0 / scala

    def __getitem__(self, idx):
        """
        Loads one case: reads the grayscale image, parses the matching Excel
        file into recognized landmarks, converts them to pixels (per-case
        scale when known), zero-pads to NUM_LANDMARKS=26, and fills in Gn (13)
        as the midpoint of Me (3) and Pog (6) whenever Gn itself wasn't
        annotated but both of those were (this landmark is rarely marked
        explicitly in the clinics' tracings).
        Returns (image_tensor, landmarks_tensor, pixel_size_tensor); landmarks
        not found stay at [0, 0] and are masked out downstream (see datasets.py).
        """
        img_path, excel_path, scala = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((1935, 2400), dtype=np.uint8)
        h_img, w_img = image.shape[:2]

        found = _parse_excel(excel_path, coord_scale=1.0, flip_y=False, img_height=0)

        if not found and not DEBUG:
            print(f"\n[TracciatiDataset] NO LANDMARKS recognized in:\n"
                  f"  {excel_path}\n"
                  f"  → Set DEBUG=True and re-run to see the names used by the software.\n"
                  f"  → Then add them to NAME_MAP in dataset_tracciati.py")

        found, pixel_size = self._in_pixel(found, scala)
        if self.flip_y:
            found = {k: [x, h_img - y] for k, (x, y) in found.items()}

        fuori = [NOMI_26[k] for k, (x, y) in found.items()
                 if not (-0.05 * w_img <= x <= 1.05 * w_img and -0.05 * h_img <= y <= 1.05 * h_img)]
        if fuori:
            print(f"[TracciatiDataset] WARNING {excel_path}: {len(fuori)} landmarks fall outside "
                  f"the image ({', '.join(fuori)}) — check scale/units")

        # Zero-pad a NUM_LANDMARKS=26
        coords = [[0.0, 0.0]] * NUM_LANDMARKS
        for lm_idx, xy in found.items():
            if 0 <= lm_idx < NUM_LANDMARKS:
                coords[lm_idx] = xy

        # Gn computed as the midpoint of Me (3) and Pog (6), when missing
        GN, ME, POG = 13, 3, 6
        if (coords[GN] == [0.0, 0.0]
                and coords[ME]  != [0.0, 0.0]
                and coords[POG] != [0.0, 0.0]):
            coords[GN] = [
                (coords[ME][0] + coords[POG][0]) / 2.0,
                (coords[ME][1] + coords[POG][1]) / 2.0,
            ]

        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        pts_tensor = torch.tensor(coords, dtype=torch.float32)

        return img_tensor, pts_tensor, torch.tensor(pixel_size, dtype=torch.float32)


# ─── Diagnostic script ─────────────────────────────────────────────────────────
# Run this file directly to inspect a Tracciati folder: prints how many
# landmarks were recognized per sample and which ones are still missing,
# to help extend NAME_MAP for new Excel formats.
if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "data/Tracciati"
    print(f"Testing TracciatiDataset → {base}")
    print("=" * 60)

    DEBUG = True

    ds = TracciatiDataset(base_path=base)
    print(f"\nTotal samples: {len(ds)}")

    if len(ds) > 0:
        img, pts, ps = ds[0]
        img_path, excel_path, scala = ds.samples[0]
        print(f"\nFirst sample: {img_path}")
        print(f"Image: {img.shape}  range [{img.min():.3f}, {img.max():.3f}]")
        print(f"Scale: {scala} px/mm | pixel_size: {ps.item():.4f} mm/px")

        trovati  = [(NOMI_26[i], pts[i].tolist())
                    for i in range(NUM_LANDMARKS)
                    if pts[i, 0] != 0.0 or pts[i, 1] != 0.0]
        mancanti = [NOMI_26[i] for i in range(NUM_LANDMARKS)
                    if pts[i, 0] == 0.0 and pts[i, 1] == 0.0]

        print(f"\nLandmarks found ({len(trovati)}/{NUM_LANDMARKS}):")
        for nome, xy in trovati:
            print(f"  {nome:5s}: x={xy[0]:.1f}  y={xy[1]:.1f}")

        if mancanti:
            print(f"\nMISSING landmarks: {mancanti}")
            print("→ Check the [DEBUG] output above to find the names used by the software")
            print("→ Add them to NAME_MAP in dataset_tracciati.py")