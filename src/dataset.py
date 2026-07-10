import json
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

NUM_LANDMARKS = 26

# Canonical order of the 26 landmarks — used to extract points from the
# Aariz JSON files by symbol, so the mapping is robust to whatever order
# the dataset happens to save them in.
SYMBOL_ORDER = [
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


class AarizDataset(Dataset):
    """
    Loads the Aariz cephalometric dataset. Each sample is a grayscale
    cephalogram plus its landmark annotations, built by averaging the
    Senior and Junior orthodontist annotations when both are available
    (falling back to Senior-only otherwise).
    """
    def __init__(self, base_path, split='train', transform=None):
        self.base_path  = base_path
        self.transform  = transform

        # split='all' loads train + valid + test together, so that ALL
        # available Aariz images are used instead of only the 'train' split.
        splits = ['train', 'valid', 'test'] if split == 'all' else [split]

        csv_path = os.path.join(base_path, 'cephalogram_machine_mappings.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            self.pixel_size_map = dict(zip(df['cephalogram_id'], df['pixel_size']))
        else:
            self.pixel_size_map = {}
        self.default_pixel_size = 0.1

        # Build the list of (filename, split, extension) tuples for every
        # requested split, keeping only samples that have a matching
        # Senior-orthodontist annotation file.
        self.samples = []
        for sp in splits:
            img_dir    = os.path.join(base_path, sp, 'Cephalograms')
            senior_dir = os.path.join(base_path, sp, 'Annotations', 'Cephalometric Landmarks', 'Senior Orthodontists')
            if not os.path.exists(img_dir):
                continue
            for f in os.listdir(img_dir):
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.bmp'):
                    ext   = os.path.splitext(f)[1]
                    fname = os.path.splitext(f)[0]
                    if os.path.exists(os.path.join(senior_dir, fname + '.json')):
                        self.samples.append((fname, sp, ext))

        self.filenames = [s[0] for s in self.samples]
        print(f"[AarizDataset] split={splits} | total samples={len(self.samples)} (png+jpg)")
        
        self.img_dir    = os.path.join(base_path, splits[0], 'Cephalograms')
        self.senior_dir = os.path.join(base_path, splits[0], 'Annotations', 'Cephalometric Landmarks', 'Senior Orthodontists')
        self.junior_dir = os.path.join(base_path, splits[0], 'Annotations', 'Cephalometric Landmarks', 'Junior Orthodontists')
        self.usa_junior = os.path.exists(self.junior_dir)

    def __len__(self):
        """Number of valid samples (image + Senior annotation available)."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads one sample: reads the grayscale image and its Senior
        annotation; if a matching Junior annotation also exists, the
        returned landmarks are the average of the two (ceil-rounded),
        otherwise the Senior annotation is used as-is.
        Returns (image_tensor, landmarks_tensor, pixel_size_tensor).
        """

        fname, split, ext = self.samples[idx]
        img_dir    = os.path.join(self.base_path, split, 'Cephalograms')
        senior_dir = os.path.join(self.base_path, split, 'Annotations', 'Cephalometric Landmarks', 'Senior Orthodontists')
        junior_dir = os.path.join(self.base_path, split, 'Annotations', 'Cephalometric Landmarks', 'Junior Orthodontists')

        image = cv2.imread(os.path.join(img_dir, fname + ext), cv2.IMREAD_GRAYSCALE)

        senior = np.array(self._parse_json(os.path.join(senior_dir, fname + '.json')),
                          dtype=np.float32)

        if os.path.exists(junior_dir):
            junior_path = os.path.join(junior_dir, fname + '.json')
            if os.path.exists(junior_path):
                junior = np.array(self._parse_json(junior_path), dtype=np.float32)
                landmarks = np.ceil(0.5 * (senior + junior)).astype(np.float32)
            else:
                landmarks = senior
        else:
            landmarks = senior

        pixel_size = self.pixel_size_map.get(fname, self.default_pixel_size)

        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        return (
            image,
            torch.tensor(landmarks, dtype=torch.float32),
            torch.tensor(pixel_size, dtype=torch.float32),
        )

    def _parse_json(self, path):
        """
        Parses landmarks by symbol name, so this is robust to whatever
        order the dataset saves them in. Returns exactly NUM_LANDMARKS=26
        [x, y] pairs. Landmarks absent from the JSON -> [0.0, 0.0] (to be
        masked out downstream).
        """
        with open(path, 'r') as f:
            data = json.load(f)
        lm_dict = {
            item['symbol']: [item['value']['x'], item['value']['y']]
            for item in data['landmarks']
        }
        return [lm_dict.get(sym, [0.0, 0.0]) for sym in SYMBOL_ORDER]