import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

NUM_LANDMARKS = 26

# Mapping from the ISBI/Cepha400 "P1..P19" landmark indices to this
# project's canonical 26-landmark indexing (see dataset.py / SYMBOL_ORDER).
#
# "standard" is the landmark order published for the ISBI 2015 challenge
# (Wang et al. 2015/2016) and has been VERIFIED visually on the data
# (Test/verifica_mappatura_isbi.py, cases 001, 151, 301): every point lands on
# the named anatomy.
# "legacy" is the mapping used before: only P1-P4 (S, N, Or, Po) were right,
# the other 15 points were mislabelled on all 400 ISBI images (e.g. P5 = A was
# trained as Ar, P10 = Go as Pog, P13 = upper lip as Go). It is kept only to
# reproduce old experiments. Models trained with it must be retrained.

ISBI_P_TO_CANONICAL_LEGACY = {
    1:  10,   # P1  = S    → idx 10
    2:   4,   # P2  = N    → idx  4
    3:   5,   # P3  = Or   → idx  5
    4:  15,   # P4  = Po   → idx 15
    5:  11,   # P5  = Ar   → idx 11
    6:   7,   # P6  = PNS  → idx  7
    7:   1,   # P7  = ANS  → idx  1
    8:   0,   # P8  = A    → idx  0
    9:   2,   # P9  = B    → idx  2
    10:  6,   # P10 = Pog  → idx  6
    11: 13,   # P11 = Gn   → idx 13
    12:  3,   # P12 = Me   → idx  3
    13: 14,   # P13 = Go   → idx 14
    14: 12,   # P14 = Co   → idx 12
    15: 16,   # P15 = LPM  → idx 16
    16: 18,   # P16 = LMT  → idx 18
    17: 17,   # P17 = LIT  → idx 17
    19:  9,   # P19 = R    → idx  9
}

ISBI_P_TO_CANONICAL_STANDARD = {
    1:  10,   # L1  sella                   → S
    2:   4,   # L2  nasion                  → N
    3:   5,   # L3  orbitale                → Or
    4:  15,   # L4  porion                  → Po
    5:   0,   # L5  subspinale              → A
    6:   2,   # L6  supramentale            → B
    7:   6,   # L7  pogonion                → Pog
    8:   3,   # L8  menton                  → Me
    9:  13,   # L9  gnathion                → Gn
    10: 14,   # L10 gonion                  → Go
    11: 17,   # L11 lower incisal incision  → LIT
    12: 19,   # L12 upper incisal incision  → UIT
    13: 22,   # L13 upper lip               → Ls
    14: 23,   # L14 lower lip               → Li
    15: 25,   # L15 subnasale               → Sn
    16: 24,   # L16 soft tissue pogonion    → Pog`
    17:  7,   # L17 posterior nasal spine   → PNS
    18:  1,   # L18 anterior nasal spine    → ANS
    19: 11,   # L19 articulare              → Ar
}

MAPPATURE_ISBI = {"legacy": ISBI_P_TO_CANONICAL_LEGACY,
                  "standard": ISBI_P_TO_CANONICAL_STANDARD}

# Mapping used everywhere (training, evaluation, statistics). See the note above.
MAPPATURA_ISBI_DEFAULT = "standard"

# Kept for code that imports the old name
ISBI_P_TO_CANONICAL = MAPPATURE_ISBI[MAPPATURA_ISBI_DEFAULT]

# Image subfolders in order of case ID ranges: 001-150, 151-300, 301-400
IMG_SUBFOLDERS = [
    ("TrainingData", range(1,   151)),
    ("Test1Data",    range(151, 301)),
    ("Test2Data",    range(301, 401)),
]


def parse_md_txt(path):
    """
    Reads one annotation TXT file (as provided by the ISBI/Cepha400 dataset).
    File format: 27 lines total.
      - Lines 1-19  : landmark (x, y) pairs, comma-separated
      - Lines 20-27 : quality scores (1/2/3), ignored here
    Returns a (19, 2) float32 array of raw pixel coordinates.
    """
    with open(path, 'rb') as f:
        lines = f.read().decode('utf-8').strip().split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    coords = np.zeros((19, 2), dtype='float32')
    for i in range(min(19, len(lines))):
        parts = lines[i].split(',')
        if len(parts) == 2:
            coords[i] = [float(parts[0]), float(parts[1])]
    return coords


class ISBIDataset(Dataset):
    """
    Loads the full Cepha400 dataset (all 400 images) with landmark
    annotations from two annotators (junior + senior). The coordinates
    used are the AVERAGE of the two annotators, mirroring the approach
    used for AarizDataset.

    Train/val splitting is handled externally by Datasets via
    random_split, exactly as for Aariz.
    """
    def __init__(self, base_path="data/Cepha400", mappatura=None):
        """
        Scans the dataset folder and builds the list of available samples,
        keeping only cases where the image and both annotation files
        (junior and senior) actually exist on disk.
        mappatura: "legacy" | "standard" (default MAPPATURA_ISBI_DEFAULT).
        """
        self.base_path  = base_path
        self.mappatura  = mappatura or MAPPATURA_ISBI_DEFAULT
        self.p_to_canonical = MAPPATURE_ISBI[self.mappatura]
        self.junior_dir = os.path.join(base_path, "AnnotationsByMD", "400_junior")
        self.senior_dir = os.path.join(base_path, "AnnotationsByMD", "400_senior")

        self.samples = []
        for subfolder, id_range in IMG_SUBFOLDERS:
            img_dir = os.path.join(base_path, "RawImage", subfolder)
            for n in id_range:
                img_path    = os.path.join(img_dir,        f"{n:03d}.bmp")
                junior_path = os.path.join(self.junior_dir, f"{n:03d}.txt")
                senior_path = os.path.join(self.senior_dir, f"{n:03d}.txt")
                if os.path.exists(img_path) and \
                   os.path.exists(junior_path) and \
                   os.path.exists(senior_path):
                    self.samples.append((n, img_path))
                else:
                    print(f"[ISBIDataset] ATTENZIONE: campione {n:03d} mancante, saltato")

        print(f"[ISBIDataset] campioni totali={len(self.samples)} | mappatura={self.mappatura}")

    def __len__(self):
        """Number of valid samples found on disk.""" 
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads one sample: reads the grayscale image, averages the junior
        and senior annotations, remaps the P1-P19 points to this project's
        canonical 26-landmark indices, and returns (image_tensor, landmarks_tensor).
        Landmarks not covered by the P1-P19 mapping (with "standard": Pn, R,
        Co, LPM, LMT, UIA, LIA) stay at [0, 0] and are masked out downstream
        (see datasets.py).
        """
        n, img_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((1935, 2400), dtype=np.uint8)

        junior_coords = parse_md_txt(os.path.join(self.junior_dir, f"{n:03d}.txt"))
        senior_coords = parse_md_txt(os.path.join(self.senior_dir, f"{n:03d}.txt"))
        avg_p = (junior_coords + senior_coords) / 2.0   # shape (19, 2)

        coords = np.zeros((NUM_LANDMARKS, 2), dtype='float32')
        for p_num in range(1, 20):
            canonical_idx = self.p_to_canonical.get(p_num)
            if canonical_idx is None:
                continue
            coords[canonical_idx] = avg_p[p_num - 1]

        image_tensor     = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        landmarks_tensor = torch.from_numpy(coords)
        return image_tensor, landmarks_tensor