"""
datasets.py — aggregator for all datasets used in this project.

Datasets:
  - AarizDataset           (Aariz)
  - ISBIDataset            (Cepha400)
  - TracciatiDataset       (in-house tracings with coordinates)
"""
import torch
import os
import sys
from torch.utils.data import Dataset

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

# Default scale/orientation for the in-house "Tracciati" Excel coordinates
TRACCIATI_COORD_SCALE_DEFAULT = 0.01
TRACCIATI_FLIP_Y_DEFAULT      = False

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
        self.isbi  = ISBIDataset(base_path=path_isbi)

        if tracciati_path and os.path.isdir(tracciati_path):
            # ── Folders with coordinates in scaled units (from Excel .xltx/.xlsx) ──
            # coord_scale=0.01: raw values (~100000) → pixels of the full-size image
            self.tracciati = TracciatiDataset(
                base_path=tracciati_path,
                subfolder_names=["tracciato 50", "tracciato 50_2"],
                pixel_size=PIXEL_SIZE_TRACCIATI,
                coord_scale=tracciati_coord_scale,  
                flip_y=tracciati_flip_y,
                use_thumbnail=False,
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