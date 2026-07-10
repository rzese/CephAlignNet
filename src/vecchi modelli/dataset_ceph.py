import os
import json
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

NUM_LANDMARKS = 26   # 19 originali + UIT(19) + UIA(20) + LIA(21) + Ls(22) + Li(23) + Pog`(24) + Sn(25)
# CephBiometrics non ha i landmark soft tissue (22-25) → rimangono [0,0] e la maschera li esclude.

# CephBiometrics usa lo stesso schema P1-P19 di ISBI (Lindner 2015)
# Stesso mapping → stessi indici canonici
CEPH_P_TO_CANONICAL = {
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
    # P18 = UPM → non in SYMBOL_ORDER, skip
    19:  9,   # P19 = R    → idx  9
}


class CephBiometricsDataset(Dataset):
    def __init__(self, base_path="data/CephBiometrics400"):
        self.base_path = base_path
        self.img_base  = os.path.join(base_path, "Immagini")
        self.lnm_base  = os.path.join(base_path, "Landmark")
        if os.path.exists(self.img_base):
            self.sample_ids = sorted([
                d.split('.')[0] for d in os.listdir(self.img_base)
                if os.path.isdir(os.path.join(self.img_base, d))
            ])
        else:
            self.sample_ids = []

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        img_path  = os.path.join(self.img_base, f"{sid}.nii", f"{sid}.nii")
        json_path = os.path.join(self.lnm_base,  f"{sid}.json", f"{sid}.json")

        # Caricamento immagine
        try:
            if os.path.exists(img_path):
                nii_img = nib.load(img_path)
                image   = nii_img.get_fdata()
                image   = np.squeeze(image)
                if image.ndim == 3:
                    image = image[:, :, 0]
                image = image.T
                image = np.flipud(image)
                image = np.ascontiguousarray(image)
                image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            else:
                image = np.zeros((1935, 2400), dtype=np.uint8)
        except Exception:
            image = np.zeros((1935, 2400), dtype=np.uint8)

        # Caricamento landmark: rimappa P1-P19 → indici canonici
        # UIT(19), UIA(20), LIA(21) non presenti → rimangono [0,0]
        coords = np.zeros((NUM_LANDMARKS, 2), dtype='float32')
        try:
            if os.path.exists(json_path) and os.path.isfile(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                lms   = data['slice_landmarks_x'][0]['landmarks']
                h_img = image.shape[0]
                for p_num in range(1, 20):
                    canonical_idx = CEPH_P_TO_CANONICAL.get(p_num)
                    if canonical_idx is None:
                        continue
                    key = f"P{p_num}"
                    if key in lms:
                        pt = lms[key]
                        coords[canonical_idx] = [float(pt[1]), h_img - float(pt[2])]
        except Exception:
            pass

        img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        pts_tensor = torch.tensor(coords, dtype=torch.float32)
        return img_tensor, pts_tensor