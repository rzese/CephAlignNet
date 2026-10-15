# Cephalometric Landmark Detection

A two-stage deep learning pipeline for automatic detection of cephalometric
landmarks on lateral skull X-rays (cephalograms), used in orthodontic
diagnosis. The model locates up to **26 anatomical landmarks** (19 classic
skeletal/dental points plus 7 additional dental/soft-tissue points) on a
single grayscale radiograph.

## Overview

The pipeline is split into two stages:

- **Stage 1 — Heatmap regression (`model_hrnet.py`, `train_heatmap.py`)**
  An HRNet-W48 backbone predicts, for every landmark, a Gaussian heatmap
  (its likely location) plus two local offset maps used for sub-pixel
  refinement. Coordinates are decoded from the heatmap via soft-argmax
  (default) or an offset-voting scheme.

- **Stage 2 — Refinement network (`model_refine.py`, `train_refine.py`)**
  `ImageRefinementNet` looks at a 25.6 mm crop of the **original
  full-resolution radiograph** around the Stage 1 prediction and predicts,
  through a small heatmap and soft-argmax, where the landmark is inside the
  crop (correction up to 5 mm). It is trained without Stage 1, on
  ground-truth points moved by synthetic errors, so it cannot simply echo
  Stage 1. It can be applied to all points, to a fixed landmark mask, or
  only to the points with high TTA uncertainty (`selezione_stage2`).

At inference time (`predict_heatmap.py`), Stage 1 runs with test-time
augmentation (rotations, horizontal flip, brightness/contrast jitter) and
its output is optionally corrected by Stage 2, giving the final landmark
coordinates on the original image.

## Heatmap modes and uncertainty

`CONFIG["heatmap_mode"]` in `train_heatmap.py` selects how Stage 1 heatmaps
are trained (the choice is stored in the checkpoint):

- `"fixed"` — fixed-sigma Gaussian targets + Adaptive Wing loss (original).
- `"learned_sigma"` — per-landmark anisotropic target sigma trained together
  with the network (Payer et al., 2020).
- `"probabilistic"` — the heatmap is the logit map of a spatial distribution,
  trained with the negative log-likelihood of the ground truth.

`evaluate_refine.py` measures how well the heatmap spread and the TTA spread
predict the actual error (Spearman, error per quintile) and writes
per-landmark conformal factors to `calibrazione_incertezza.json`;
`predict_heatmap.py` uses them to report a radius that contains the true
point with ~90% probability. Details (in Italian): `DOCUMENTAZIONE.md`.

Unit tests (CPU, synthetic data): `PYTHONPATH=. python Test/test_unita.py`.

## Repository structure

| File | Role |
|---|---|
| `model_hrnet.py` | Stage 1 model: HRNet-W48 backbone + multi-scale fusion head. Outputs a heatmap and two offset channels per landmark, plus the fused feature map used by Stage 2. |
| `model_refine.py` | Stage 2 models: `ImageRefinementNet` (current: crop of the original radiograph → heatmap → offset in mm) and the older `RefinementNet` (crop of the Stage 1 feature map, kept to load old checkpoints). Defines `REFINE_LANDMARK_MASK`. |
| `incertezza.py` | Learnable heatmap sigma (`SigmaAddestrabile`), heatmap/TTA covariance, uncertainty calibration and evaluation. |
| `inferenza.py` | Code shared by the scripts: checkpoint loading, decoding, TTA, Stage 2 application, collate and mm errors. |
| `heatmap_utils.py` | Shared utilities: Gaussian heatmap generation, heatmap/offset losses (Adaptive Wing loss, combined loss), soft-argmax and voting decoders, the training-time augmentation pipeline (`augmenta`), a hard-landmark zoom-crop augmentation, and a geometric-consistency loss that penalizes anatomically implausible landmark configurations. |
| `dataset.py` | `AarizDataset` — loads the public **Aariz** dataset (image + JSON annotations), averaging Senior/Junior orthodontist annotations when both are available. |
| `dataset_isbi.py` | `ISBIDataset` — loads the public **Cepha400 / ISBI 2015** dataset (image + two annotator text files), remapping its 19 landmarks to this project's 26-landmark indexing. |
| `dataset_tracciati.py` | `TracciatiDataset` — loads an in-house dataset of clinic tracings: an image plus an Excel file with landmark coordinates. Includes a large name-matching table (`NAME_MAP`) to cope with inconsistent naming across different clinics/software, and a diagnostic mode (`DEBUG=True`, or running the file directly) to inspect unrecognized landmark names in a new Excel file. |
| `datasets.py` | `Datasets` — aggregates Aariz + ISBI + (optionally) Tracciati into one dataset with a common output format `(image, points, mask, bbox, pixel_size)`, and builds the validity mask/bounding box for each sample. |
| `train_heatmap.py` | Trains Stage 1 (HRNet). Builds the combined loss (heatmap + offset + coordinate Wing loss + geometric consistency), trains with early stopping and an LR scheduler, and saves the best checkpoint. |
| `train_refine.py` | Trains the image-based Stage 2 on crops of the original radiographs centered on ground-truth points moved by synthetic errors (Gaussian ~1.5 mm mixed with uniform up to 5 mm); no Stage 1 needed. |
| `evaluate_refine.py` | Evaluates Stage 1 alone and Stage 1+Stage 2 together on a validation split, computing **MRE** (Mean Radial Error, in mm) and **SDR** (Successful Detection Rate at several mm thresholds), globally, per anatomical category, and per landmark. |
| `predict_heatmap.py` | Command-line inference script: loads both stage checkpoints, runs the full pipeline (with TTA) on a single image, prints the predicted coordinates, and displays them overlaid on the image. |

## The 26 landmarks

```
0  A     1  ANS   2  B     3  Me    4  N     5  Or    6  Pog   7  PNS
8  Pn    9  R      10 S    11 Ar   12 Co    13 Gn    14 Go    15 Po
16 LPM   17 LIT    18 LMT  19 UIT  20 UIA   21 LIA
22 Ls    23 Li     24 Pog` 25 Sn
```

Aariz annotates all 26 landmarks. The 19 points of the public ISBI/Cepha400
dataset (ISBI 2015 challenge order) are remapped into this indexing: S, N, Or,
Po, A, B, Pog, Me, Gn, Go, LIT, UIT, Ls, Li, Sn, Pog`, PNS, ANS, Ar. The
landmarks an image does not have (for ISBI: Pn, R, Co, LPM, LMT, UIA, LIA) are
masked out of the loss and of the metrics.

> The mapping used before (`"legacy"` in `src/dataset_isbi.py`) mislabelled 15
> of the 19 ISBI points; it was checked visually with
> `Test/verifica_mappatura_isbi.py`. Checkpoints trained before this fix must be
> retrained.

## Data

The scripts expect three possible data sources, configured via paths in
each script's `CONFIG` dict:

- **Aariz** (`aariz_path`, default `data/Aariz`) — public dataset, images +
  JSON landmark annotations from Senior/Junior orthodontists.
- **Cepha400 / ISBI 2015** (`isbi_path`, default `data/Cepha400`) — public
  dataset, images + two annotators' text files.
- **Tracciati** (`tracciati_path`, default `data/tracciati_forniti`) — an
  optional, in-house set of 100 additional clinic tracings (image + Excel
  file per case). Not required to run the pipeline; if the folder is
  absent, training simply proceeds with Aariz + ISBI only. It is
  intentionally **excluded** from `evaluate_refine.py` so that MRE/SDR
  metrics stay comparable to public-dataset baselines.

None of the datasets are included in this repository; the paths above must
point to locally downloaded copies.

## Running the pipeline

```bash
# 1. Train Stage 1 (heatmap model)
python train_heatmap.py        # saves modello_heatmap.pth

# 2. Train Stage 2 (refinement network), on top of the frozen Stage 1
python train_refine.py         # saves modello_refine.pth

# 3. Evaluate both stages (MRE / SDR)
python evaluate_refine.py

# 4. Run inference on a single image
python predict_heatmap.py path/to/image.png
```

Each script's `CONFIG` dictionary at the top of the file controls dataset
paths, image/heatmap resolution, batch size, learning rate, number of
epochs, and other hyperparameters.

## Requirements

- Python 3, PyTorch + torchvision
- `timm` (HRNet-W48 backbone)
- `opencv-python`, `numpy`, `pandas`
- `openpyxl` (only needed for `dataset_tracciati.py`, to read Excel files)

A CUDA-capable GPU is strongly recommended for training; inference
(`predict_heatmap.py`) also runs on CPU, only more slowly.
