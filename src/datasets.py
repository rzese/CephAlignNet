"""
datasets.py — aggregator for all datasets used in this project.

Datasets:
  - AarizDataset           (Aariz)
  - ISBIDataset            (Cepha400)
  - TracciatiDataset       (in-house tracings with coordinates)
"""
import hashlib
import torch
import os
import sys
from torch.utils.data import Dataset, Subset

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from dataset           import AarizDataset
    from dataset_isbi      import ISBIDataset
    from dataset_tracciati import TracciatiDataset
except ImportError:
    from .dataset           import AarizDataset
    from .dataset_isbi      import ISBIDataset
    from .dataset_tracciati import TracciatiDataset

# Default pixel size (mm/px) used when a dataset doesn't provide its own
PIXEL_SIZE_ISBI      = 0.1
PIXEL_SIZE_TRACCIATI = 0.1

# Default scale/orientation for the in-house "Tracciati" Excel coordinates.
# The global scale is only a fallback for cases without a known per-case scale
# (see tracciati_tabella_scale and dataset_tracciati.py).
TRACCIATI_COORD_SCALE_DEFAULT = 0.01
TRACCIATI_FLIP_Y_DEFAULT      = False
TRACCIATI_SUBFOLDERS_DEFAULT  = ["tracciato 50", "tracciato 50_2"]

# 19 original landmarks + UIT(19) + UIA(20) + LIA(21)
# + Ls/UL(22) + Li/LL(23) + Pog`(24) + Sn(25) = 26
NUM_LANDMARKS = 26


class Datasets(Dataset):
    """
    Aggregates AarizDataset, ISBIDataset and (optionally) TracciatiDataset
    into a single indexable dataset with a common 26-landmark output format:
    (image, points, mask, bbox, pixel_size). Indices are routed to the
    underlying dataset transparently — Aariz samples first, then ISBI, then
    Tracciati (see __getitem__ routing below).
    """
    def __init__(self,
                 aariz_path=None,
                 isbi_path=None,
                 tracciati_path=None,
                 tracciati_coord_scale: float = 0.01,
                 tracciati_flip_y: bool = False,
                 tracciati_tabella_scale=None,
                 tracciati_subfolders=None,
                 tracciati_unita: str = "auto",
                 isbi_mappatura=None,
                 isbi_base=None,
                 ceph_path=None,
                 ceph_base=None,
                 transform=None):

        path_aariz = aariz_path or "data/Aariz"
        path_isbi  = isbi_path  or isbi_base or "data/ISBI2015"

        if ceph_path or ceph_base:
            print("[Datasets] NOTE: ceph_path ignored — CephBiometrics removed "
                  "(duplicate images of ISBI).")

        # split='all' loads ALL available Aariz images (train + valid + test)
        # instead of only the 'train' split.
        self.aariz = AarizDataset(base_path=path_aariz, split='all')
        self.isbi  = ISBIDataset(base_path=path_isbi, mappatura=isbi_mappatura)

        if tracciati_path and os.path.isdir(tracciati_path):
            # ── Folders with coordinates in scaled units (from Excel .xltx/.xlsx) ──
            # coord_scale=0.01: raw values (~100000) → pixels of the full-size image
            self.tracciati = TracciatiDataset(
                base_path=tracciati_path,
                subfolder_names=tracciati_subfolders or TRACCIATI_SUBFOLDERS_DEFAULT,
                pixel_size=PIXEL_SIZE_TRACCIATI,
                coord_scale=tracciati_coord_scale,
                flip_y=tracciati_flip_y,
                use_thumbnail=False,
                tabella_scale=tracciati_tabella_scale,
                unita_excel=tracciati_unita,
            )
        else:
            self.tracciati = None
            if tracciati_path:
                print(f"[Datasets] WARNING: tracciati_path not found: {tracciati_path}")

        self.len_aariz     = len(self.aariz)
        self.len_isbi      = len(self.isbi)
        self.len_tracciati = len(self.tracciati) if self.tracciati else 0

        print(f"[Datasets] Aariz={self.len_aariz} | ISBI={self.len_isbi} | "
              f"Tracciati={self.len_tracciati} | "
              f"TOT={len(self)}")

    def __len__(self):
        """Total number of samples across all three underlying datasets."""
        return self.len_aariz + self.len_isbi + self.len_tracciati

    def __getitem__(self, idx):
        """
        Routes idx to the right underlying dataset (Aariz -> ISBI ->
        Tracciati, in that order) and wraps its raw output into the common
        (image, points, mask, bbox, pixel_size) format via wrap_output.
        """
        def wrap_output(data, pixel_size_default, source="?"):
            """
            Normalizes a raw dataset sample into the common output format:
            truncates/pads points to NUM_LANDMARKS, builds a validity mask
            (landmarks at [0, 0] are considered "not annotated" and masked
            out), and computes a padded bounding box around the valid points
            for downstream use (e.g. cropping).
            """
            if data is None:
                raise ValueError(f"Dataset '{source}' returned None!")

            if isinstance(data, (tuple, list)):
                if len(data) == 3:
                    img, pts, pixel_size = data
                elif len(data) == 2:
                    img, pts = data
                    pixel_size = torch.tensor(pixel_size_default, dtype=torch.float32)
                else:
                    raise ValueError(f"Dataset '{source}' unexpected format: {type(data)}")

                pts   = pts[:NUM_LANDMARKS]
                n_pts = pts.shape[0]
                mask  = torch.ones(n_pts, dtype=torch.float32)

                invalid = (pts[:, 0] == 0.0) & (pts[:, 1] == 0.0)
                mask[invalid] = 0.0

                valid = pts[~invalid]
                if len(valid) > 0:
                    padding = 20.0
                    bbox = torch.tensor([
                        max(0.0, valid[:, 0].min().item() - padding),
                        max(0.0, valid[:, 1].min().item() - padding),
                        min(float(img.shape[2]), valid[:, 0].max().item() + padding),
                        min(float(img.shape[1]), valid[:, 1].max().item() + padding),
                    ])
                else:
                    bbox = torch.tensor([0.0, 0.0,
                                         float(img.shape[2]), float(img.shape[1])])

                return img, pts, mask, bbox, pixel_size

            raise ValueError(f"Dataset '{source}' formato inatteso: {type(data)}")

        # ── Routing: Aariz → ISBI → Tracciati ──────────────────────────────
        if idx < self.len_aariz:
            return wrap_output(self.aariz[idx], PIXEL_SIZE_ISBI, source="AARIZ")

        idx -= self.len_aariz
        if idx < self.len_isbi:
            return wrap_output(self.isbi[idx], PIXEL_SIZE_ISBI, source="ISBI")

        idx -= self.len_isbi
        return wrap_output(self.tracciati[idx], PIXEL_SIZE_TRACCIATI, source="TRACCIATI")

# ── Fingerprint of the validation set ─────────────────────────────────────────
# A short hash of the validation images (names, in order). Training scripts save
# it in their checkpoints; loading a checkpoint and building the split both
# register it here, and a mismatch prints a warning: the model would be
# evaluated on images it was trained on (or on a different validation set).
_IMPRONTE = {"split": None, "modelli": {}}


def nomi_campioni(dataset, indici):
    """'aariz/<split>/<file>' or 'isbi/<n>' of the given dataset indices."""
    nomi = []
    if not hasattr(dataset, "aariz"):          # stand-in datasets (tests): indices only
        return [str(g) for g in indici]
    for g in indici:
        if g < dataset.len_aariz:
            fname, split, ext = dataset.aariz.samples[g]
            nomi.append(f"aariz/{split}/{fname}{ext}")
        elif g < dataset.len_aariz + dataset.len_isbi:
            nomi.append(f"isbi/{dataset.isbi.samples[g - dataset.len_aariz][0]:03d}")
        else:
            nomi.append(f"tracciati/{g}")
    return nomi


def impronta_split():
    """Fingerprint of the last validation set built by crea_split (None before)."""
    return _IMPRONTE["split"]


def _controlla_impronte():
    split = _IMPRONTE["split"]
    for nome, imp in _IMPRONTE["modelli"].items():
        if split and imp and imp != split:
            print(f"[WARN] {nome} was trained with validation set {imp}, but this validation set is {split}: "
                  f"the results are NOT valid (training images in the validation set). "
                  f"Same data and code on both machines?")


def registra_impronta_modello(nome, impronta):
    """Called when a checkpoint is loaded (impronta = None for old checkpoints)."""
    _IMPRONTE["modelli"][nome] = impronta
    _controlla_impronte()


def crea_split(dataset, val_split=0.2, seed=42):
    """
    Train/validation split used by every script (training, evaluation,
    statistics), so that they all see the same validation images.

    Only the public samples (Aariz + ISBI) are split; Tracciati samples, if
    present, all go to training (they are excluded from evaluation by design).
    The public split therefore does not depend on whether Tracciati are
    loaded — before, random_split over the whole dataset gave a different
    permutation when they were added, and evaluation scripts (built without
    Tracciati) ended up validating on training images.
    With no Tracciati it reproduces exactly the old
    random_split(dataset, ..., generator=manual_seed(seed)).
    """
    n_pub = dataset.len_aariz + dataset.len_isbi
    n_val = int(n_pub * val_split)
    perm  = torch.randperm(n_pub, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = perm[:n_pub - n_val] + list(range(n_pub, len(dataset)))
    val_idx   = perm[n_pub - n_val:]
    _IMPRONTE["split"] = hashlib.md5("\n".join(nomi_campioni(dataset, val_idx)).encode()).hexdigest()[:10]
    print(f"[crea_split] validation: {len(val_idx)} images, fingerprint {_IMPRONTE['split']}")
    _controlla_impronte()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
