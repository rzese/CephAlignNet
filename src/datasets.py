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


def formatta_campione(data, pixel_size_default, source="?"):
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
        # ── Routing: Aariz → ISBI → Tracciati ──────────────────────────────
        if idx < self.len_aariz:
            return formatta_campione(self.aariz[idx], PIXEL_SIZE_ISBI, source="AARIZ")

        idx -= self.len_aariz
        if idx < self.len_isbi:
            return formatta_campione(self.isbi[idx], PIXEL_SIZE_ISBI, source="ISBI")

        idx -= self.len_isbi
        return formatta_campione(self.tracciati[idx], PIXEL_SIZE_TRACCIATI, source="TRACCIATI")

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


# ── Train / validation / test split ───────────────────────────────────────────
# "ufficiale" (default): the split the datasets come with.
#     Aariz:  train -> training, valid -> validation, test -> test
#     ISBI:   TrainingData (001-150) -> training, except every ISBI_VAL_OGNI-th
#             image (005, 010, ..., 150: 30 images) -> validation;
#             Test1Data (151-300) and Test2Data (301-400) -> test
#     Tracciati (if loaded): all training
# "casuale": the old random 80/20 split of Aariz + ISBI (val_split, seed),
#     only to evaluate models trained with it (e.g. esperimenti_leo).
# Validation is used for early stopping and to tune the fusion, the thresholds
# and the uncertainty calibration; the test sets are only used for the final
# results. Choose the mode with the environment variable CEPH_SPLIT.
SPLIT_DEFAULT = os.environ.get("CEPH_SPLIT", "ufficiale")
ISBI_VAL_OGNI = 5
INSIEMI = ("validation", "test", "test_aariz", "test_isbi1", "test_isbi2", "unife")
UNIFE_PATH_DEFAULT = "data/tracciati"   # external test set (UNIFE tracings), see dataset_unife.py


def gruppo_ufficiale(dataset, g):
    """Official role of dataset index g: 'train', 'valid', 'test_aariz',
    'test_isbi1' or 'test_isbi2'."""
    if g < dataset.len_aariz:
        split = dataset.aariz.samples[g][1]
        return {"train": "train", "valid": "valid", "test": "test_aariz"}[split]
    if g < dataset.len_aariz + dataset.len_isbi:
        n = dataset.isbi.samples[g - dataset.len_aariz][0]
        if n <= 150:
            return "valid" if n % ISBI_VAL_OGNI == 0 else "train"
        return "test_isbi1" if n <= 300 else "test_isbi2"
    return "train"                                   # Tracciati and future in-house data


def _indici_split(dataset, val_split, seed, modo):
    """(train, validation, {test set name: indices}) for the given mode."""
    if modo == "casuale":
        # Only the public samples are split; Tracciati all go to training, so the
        # public split does not depend on whether they are loaded. With no
        # Tracciati it reproduces random_split(dataset, ..., manual_seed(seed)).
        n_pub = dataset.len_aariz + dataset.len_isbi
        n_val = int(n_pub * val_split)
        perm  = torch.randperm(n_pub, generator=torch.Generator().manual_seed(seed)).tolist()
        return perm[:n_pub - n_val] + list(range(n_pub, len(dataset))), perm[n_pub - n_val:], {}
    if modo != "ufficiale":
        raise ValueError(f"unknown split mode: {modo} (use 'ufficiale' or 'casuale')")
    gruppi = [gruppo_ufficiale(dataset, g) for g in range(len(dataset))]
    sel = lambda nome: [g for g, x in enumerate(gruppi) if x == nome]
    test = {n: sel(n) for n in ("test_aariz", "test_isbi1", "test_isbi2")}
    return sel("train"), sel("valid"), test


def crea_split(dataset, val_split=0.2, seed=42, modo=None):
    """
    Train/validation split used by every script (training, evaluation,
    statistics), so that they all see the same validation images.
    modo: "ufficiale" | "casuale" (default SPLIT_DEFAULT, see above);
    val_split and seed are used only by "casuale".
    """
    modo = modo or SPLIT_DEFAULT
    train_idx, val_idx, _ = _indici_split(dataset, val_split, seed, modo)
    _IMPRONTE["split"] = hashlib.md5("\n".join(nomi_campioni(dataset, val_idx)).encode()).hexdigest()[:10]
    print(f"[crea_split] {modo} split: training {len(train_idx)}, validation {len(val_idx)} images, "
          f"fingerprint {_IMPRONTE['split']}")
    _controlla_impronte()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def crea_insieme(dataset, nome="validation", val_split=0.2, seed=42, modo=None, unife_path=None):
    """
    The images an evaluation runs on: "validation" (default) or a test set:
    "test" (all), "test_aariz", "test_isbi1", "test_isbi2" — official split only —
    or "unife", the external UNIFE tracings (data/tracciati, see dataset_unife.py).
    The validation fingerprint is computed anyway, so a model trained with
    another split is still reported.
    """
    if nome == "unife":
        # External test set: never used for training, independent of the split
        from src.dataset_unife import UnifeValutazione
        ds = UnifeValutazione(unife_path or UNIFE_PATH_DEFAULT)
        print(f"[crea_insieme] unife: {len(ds)} images (external test set)")
        return Subset(ds, list(range(len(ds))))
    modo = modo or SPLIT_DEFAULT
    _, val_ds = crea_split(dataset, val_split, seed, modo)
    if nome == "validation":
        return val_ds
    if nome not in INSIEMI:
        raise ValueError(f"unknown set: {nome} (one of {INSIEMI})")
    if modo != "ufficiale":
        raise ValueError("the test sets exist only with the official split (CEPH_SPLIT=ufficiale)")
    _, _, test = _indici_split(dataset, val_split, seed, modo)
    idx = sum(test.values(), []) if nome == "test" else test[nome]
    print(f"[crea_insieme] {nome}: {len(idx)} images")
    return Subset(dataset, idx)
