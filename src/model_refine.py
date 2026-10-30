import torch
import torch.nn as nn
import torch.nn.functional as F

# Maximum correction of Stage 2, in normalized coordinates. Used when TRAINING a
# new Stage 2; when loading a checkpoint, the value it was trained with (saved as
# 'max_offset') is used instead — see src/inferenza.py: carica_stage2.
MAX_OFFSET = 0.008

try:
    from src.landmark_stats import REFINE_LANDMARK_MASK
except ImportError:
    try: 
        from landmark_stats import REFINE_LANDMARK_MASK
    except ImportError:
        print("[WARN] src/landmark_stats.py not found — using fallback "
              "REFINE_LANDMARK_MASK (run `python -m src.analiz_land` to generate it from the data).")
        
        REFINE_LANDMARK_MASK = {5, 9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}

# Output size of the ROI-align crop (square)
ROI_OUTPUT_SIZE = 32

# ROI radius in feature-map space
ROI_HALF_SIZE = 16


class ChannelAttention(nn.Module):
    """CBAM channel attention — tells the model which channels matter."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.fc(self.avg(x).view(B, C))
        mx  = self.fc(self.max(x).view(B, C))
        return x * torch.sigmoid(avg + mx).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial attention — tells the model where to look inside the patch."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        mask = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * mask


class ResBlock(nn.Module):
    """Lightweight residual block with CBAM attention."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.ca    = ChannelAttention(channels)
        self.sa    = SpatialAttention()

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)), inplace=True)
        r = self.bn2(self.conv2(r))
        r = self.ca(r)
        r = self.sa(r)
        return F.relu(x + r, inplace=True)


class RefinementNet(nn.Module):
    """
    Stage 2 model: takes a small ROI crop of the Stage 1 fused feature map
    around a predicted landmark, plus an embedding of which landmark it is,
    and regresses a bounded (x, y) offset correction in normalized image
    coordinates (via a final Tanh scaled by MAX_OFFSET).
    """
    tipo_input = "feature"   # input: crop of the Stage 1 feature map (old Stage 2)

    def __init__(self, num_landmarks=26, in_channels=128, max_offset=None):
        super().__init__()
        self.in_channels = in_channels

        # Initial projection to 256 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with attention
        self.res1 = ResBlock(256)
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),  # 32→16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock(512)
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),  # 16→8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.res3 = ResBlock(512)

        # Global pooling + skip connection from the intermediate (16x16) scale
        self.pool_global = nn.AdaptiveAvgPool2d(1)   # -> [B, 512]
        self.pool_mid    = nn.AdaptiveAvgPool2d(1)   # from intermediate scale -> [B, 512]

        # Fuse: 512 (deep) + 512 (mid) + 64 (landmark embedding) -> 2
        self.landmark_embedding = nn.Embedding(num_landmarks, 64)

        self.regressor = nn.Sequential(
            nn.Linear(512 + 512 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh(),
        )

        self.max_offset = MAX_OFFSET if max_offset is None else float(max_offset)

    def forward(self, x, landmark_idx):
        s = self.stem(x)           # [B, 256, 32, 32]
        r1 = self.res1(s)          # [B, 256, 32, 32]
        d1 = self.down1(r1)        # [B, 512, 16, 16]
        r2 = self.res2(d1)         # [B, 512, 16, 16]
        mid_feat = self.pool_mid(r2).view(x.size(0), -1)   # [B, 512]
        d2 = self.down2(r2)        # [B, 512,  8,  8]
        r3 = self.res3(d2)         # [B, 512,  8,  8]
        deep_feat = self.pool_global(r3).view(x.size(0), -1)  # [B, 512]

        emb = self.landmark_embedding(landmark_idx)            # [B, 64]
        fused = torch.cat([deep_feat, mid_feat, emb], dim=1)  # [B, 1088]
        return self.regressor(fused) * self.max_offset

# ─── Stage 2 on the original radiograph ───────────────────────────────────────
class _BloccoRes(nn.Module):
    """Residual block with optional dilation (wider receptive field)."""
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(x + self.b2(self.c2(r)), inplace=True)


class ImageRefinementNet(nn.Module):
    """
    Stage 2 that looks at the ORIGINAL radiograph instead of the Stage 1
    features. Input: a crop of finestra_mm x finestra_mm centered on the Stage 1
    prediction, resampled to crop_px x crop_px (fixed mm per pixel, so crops
    are comparable across datasets with different pixel sizes).

    The network predicts, for every landmark, a small logit heatmap over the
    crop (crop_px / 4 cells per side); the channel of the requested landmark is
    turned into a distribution (softmax) whose expectation is the landmark
    position inside the crop. Returned offset = that position minus the crop
    center, in mm. Translation-equivariant convolutions + soft-argmax keep the
    spatial information that the old global-pooling regressor threw away.

    forward(crop [K,1,S,S], landmark_idx [K]) -> (offset_mm [K,2], logits [K,S/4,S/4])
    """
    tipo_input = "immagine"

    def __init__(self, num_landmarks=26, finestra_mm=25.6, crop_px=128, max_offset_mm=5.0):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.finestra_mm   = float(finestra_mm)
        self.crop_px       = int(crop_px)
        self.max_offset_mm = float(max_offset_mm)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            _BloccoRes(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            _BloccoRes(128), _BloccoRes(128, dilation=2), _BloccoRes(128, dilation=4), _BloccoRes(128, dilation=8),
        )
        self.head = nn.Conv2d(128, num_landmarks, 1)

    def forward(self, crop, landmark_idx):
        logits_tutti = self.head(self.encoder(crop))                    # [K, N, s, s]
        K, _, s, _ = logits_tutti.shape
        logits = logits_tutti[torch.arange(K, device=crop.device), landmark_idx]   # [K, s, s]
        return _baricentro_crop(logits, self.finestra_mm), logits


def _baricentro_crop(logits, finestra_mm):
    """[K, s, s] logits -> [K, 2] expected position (mm) from the crop center."""
    K, s, _ = logits.shape
    p = torch.softmax(logits.reshape(K, -1), dim=1).view(K, s, s)
    griglia = (torch.arange(s, device=logits.device, dtype=torch.float32) + 0.5) / s - 0.5
    dx = (p.sum(dim=1) * griglia).sum(dim=1)
    dy = (p.sum(dim=2) * griglia).sum(dim=1)
    return torch.stack([dx, dy], dim=1) * finestra_mm


class _BloccoDecoder(nn.Module):
    """U-Net decoder step: upsample x2, concatenate the skip, two 3x3 convs."""
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in + ch_skip, ch_out, 3, padding=1, bias=False), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ImageRefinementUNet(nn.Module):
    """
    Stage 2 on the original radiograph with a pretrained encoder and a U-Net
    decoder: same task and interface as ImageRefinementNet (crop centered on the
    Stage 1 prediction -> logit heatmap of the requested landmark -> expected
    position), but with a much finer output: the heatmap is at half the crop
    resolution (crop_px / 2 cells per side, e.g. 128 x 128 cells of 0.3 mm for a
    38.4 mm / 256 px crop) instead of crop_px / 4.

    Encoder: a timm CNN (default resnet34, ImageNet weights, 1 input channel),
    whose stride-2..32 features are merged back up to stride 2 by the decoder.

    forward(crop [K,1,S,S], landmark_idx [K]) -> (offset_mm [K,2], logits [K,S/2,S/2])
    """
    tipo_input = "immagine"

    def __init__(self, num_landmarks=26, finestra_mm=38.4, crop_px=256, max_offset_mm=15.0,
                 encoder="resnet34", pretrained=True):
        super().__init__()
        import timm
        self.num_landmarks = num_landmarks
        self.finestra_mm   = float(finestra_mm)
        self.crop_px       = int(crop_px)
        self.max_offset_mm = float(max_offset_mm)
        self.encoder_nome  = encoder
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=1)
        canali = self.encoder.feature_info.channels()          # strides 2, 4, 8, 16, 32
        dec = [256, 128, 64, 64]
        self.decoder = nn.ModuleList()
        ch = canali[-1]
        for skip, out in zip(reversed(canali[:-1]), dec):
            self.decoder.append(_BloccoDecoder(ch, skip, out))
            ch = out
        self.head = nn.Conv2d(ch, num_landmarks, 1)

    def forward(self, crop, landmark_idx):
        feats = self.encoder(crop)
        x = feats[-1]
        for blocco, skip in zip(self.decoder, reversed(feats[:-1])):
            x = blocco(x, skip)
        logits_tutti = self.head(x)                                       # [K, N, S/2, S/2]
        K = logits_tutti.shape[0]
        logits = logits_tutti[torch.arange(K, device=crop.device), landmark_idx]
        return _baricentro_crop(logits, self.finestra_mm), logits


def crea_stage2_immagine(arch="base", num_landmarks=26, finestra_mm=25.6, crop_px=128, max_offset_mm=11.0,
                         encoder="resnet34", pretrained=True):
    """Image-based Stage 2 of the given architecture: "base" (ImageRefinementNet)
    or "unet" (ImageRefinementUNet)."""
    if arch == "unet":
        return ImageRefinementUNet(num_landmarks, finestra_mm, crop_px, max_offset_mm,
                                   encoder=encoder, pretrained=pretrained)
    if arch == "base":
        return ImageRefinementNet(num_landmarks, finestra_mm, crop_px, max_offset_mm)
    raise ValueError(f"unknown Stage 2 architecture '{arch}' (expected 'base' or 'unet')")
