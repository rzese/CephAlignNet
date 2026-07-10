import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ElasticTransform
from torch.utils.data import WeightedRandomSampler
from functools import partial
import random

from src.datasets import Datasets
from src.model import CephNet

CONFIG = {
    "aariz_path":    "data/Aariz",
    "isbi_path":     "data/Cepha400",
    "ceph_path":     "data/CephBiometrics400",
    "num_landmarks": 19,
    "batch_size":    8,
    "num_epochs":    180,
    "learning_rate": 1e-4,
    "val_split":     0.2,
    "save_path":     "modello_migliore.pth",
    "img_size":      512,
}


def masked_mse_loss(pred, target, mask):
    B, N = mask.shape
    pred = pred.view(B, -1, 2)[:, :N, :]
    errore_per_punto = ((pred - target) ** 2).sum(dim=2)
    num_punti_validi = mask.sum()
    if num_punti_validi == 0:
        return torch.tensor(0.0, requires_grad=True)
    return (errore_per_punto * mask).sum() / num_punti_validi

def prepara_dati(config):
    dataset = Datasets(
        aariz_path=config["aariz_path"],
        isbi_path=config["isbi_path"],
        ceph_path=config["ceph_path"],
    )
    n_val   = int(len(dataset) * config["val_split"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    collate_train = partial(collate_con_resize, size=config["img_size"], training=True)
    collate_val   = partial(collate_con_resize, size=config["img_size"], training=False)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False,collate_fn=collate_train)
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_val)
    return train_loader, val_loader

def augmenta(img, pts, mask):
    # 1. Flip orizzontale (50% di probabilità)
    if random.random() > 0.5:
        img = TF.hflip(img)
        for i in range(pts.shape[0]):
            if mask[i] == 1.0:
                pts[i, 0] = 1.0 - pts[i, 0] 
     # Rotazione leggera ±5 gradi
    angle = random.uniform(-5, 5)
    img = TF.rotate(img, angle)     

    img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
    img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))

    # Rumore gaussiano
    noise = torch.randn_like(img) * 0.01
    img = torch.clamp(img + noise, 0.0, 1.0)

    return img, pts, mask


def collate_con_resize(batch, size=512, training=False):
    immagini, punti, masks, bboxes, pixel_sizes = zip(*batch)
    
    # Resize immagini
    immagini_resize = [
        F.interpolate(img.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
        for img in immagini
    ]
    
    # Padding punti e mask alla dimensione massima del batch
    lista_lunghezze = []
    for p in punti:
        quanti_punti = p.shape[0]
        lista_lunghezze.append(quanti_punti)
        
    max_pts = max(lista_lunghezze)

    punti_pad = []
    masks_pad = []

    for img_orig, p, m in zip(immagini, punti, masks):
        # Scala le coordinate da dimensione originale a 512x512
        h_orig = img_orig.shape[1]
        w_orig = img_orig.shape[2]
        p_scaled = p.clone()
        p_scaled[:, 0] = p[:, 0] / w_orig # X
        p_scaled[:, 1] = p[:, 1] / h_orig # Y

        n = p_scaled.shape[0]
        pad_p = torch.zeros(max_pts, 2)
        pad_p[:n] = p_scaled
        pad_m = torch.zeros(max_pts)
        pad_m[:n] = m
        punti_pad.append(pad_p)
        masks_pad.append(pad_m)

        # Augmentation solo durante il training
    if training:
        immagini_aug, punti_aug, masks_aug = [], [], []
        for img, pts, msk in zip(immagini_resize, punti_pad, masks_pad):
            img_a, pts_a, msk_a = augmenta(img, pts.clone(), msk.clone())
            immagini_aug.append(img_a)
            punti_aug.append(pts_a)
            masks_aug.append(msk_a)
        immagini_resize = immagini_aug
        punti_pad = punti_aug
        masks_pad = masks_aug

    return (torch.stack(immagini_resize), torch.stack(punti_pad), torch.stack(masks_pad), torch.stack(bboxes), torch.stack(pixel_sizes))

def prepara_dati(config):
    dataset = Datasets(aariz_path=config["aariz_path"], isbi_path=config["isbi_path"], ceph_path=config["ceph_path"])
    n_val   = int(len(dataset) * config["val_split"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Peso maggiore alle immagini Aariz (hanno tutti i 29 landmark)
    pesi = []
    for i in train_ds.indices:
        if i < dataset.len_aariz:
            pesi.append(3.0)  # Aariz vista 3 volte più spesso
        else:
            pesi.append(1.0)

    sampler = WeightedRandomSampler(pesi, num_samples=len(pesi), replacement=True)

    collate_train = partial(collate_con_resize, size=config["img_size"], training=True)
    collate_val   = partial(collate_con_resize, size=config["img_size"], training=False)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, num_workers=0, pin_memory=False, collate_fn=collate_train)
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_val)
    return train_loader, val_loader


def train_un_epoca(model, loader, optimizer, device):
    model.train()
    loss_totale = 0.0
    for immagini, punti, mask, _, _  in loader:
        immagini, punti, mask = immagini.to(device), punti.to(device), mask.to(device)
        loss = masked_mse_loss(model(immagini), punti, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_totale += loss.item()
    return loss_totale / len(loader)

def valida(model, loader, device):
    model.eval()
    loss_totale = 0.0
    with torch.no_grad():
        for immagini, punti, mask, _, _ in loader:
            immagini, punti, mask = immagini.to(device), punti.to(device), mask.to(device)
            loss_totale += masked_mse_loss(model(immagini), punti, mask).item()
    return loss_totale / len(loader) 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_loader, val_loader = prepara_dati(CONFIG)
    model     = CephNet(num_landmarks=CONFIG["num_landmarks"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.3, min_lr=1e-6)

    migliore_val_loss = float('inf')
    PATIENCE_EARLY_STOP = 30
    epoche_senza_miglioramento = 0

    for epoca in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_un_epoca(model, train_loader, optimizer, device)
        val_loss   = valida(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Epoca {epoca:3d}/{CONFIG['num_epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < migliore_val_loss:
            migliore_val_loss = val_loss
            epoche_senza_miglioramento = 0
            torch.save({
                'epoca': epoca,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
            }, CONFIG["save_path"])
            print(f"  → salvato (val_loss: {val_loss:.4f})")
        else:
            epoche_senza_miglioramento += 1
            if epoche_senza_miglioramento >= PATIENCE_EARLY_STOP:
                print(f"Early stopping all'epoca {epoca}")
                break


if __name__ == "__main__":
    main()