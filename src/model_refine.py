import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_OFFSET = 0.008   # reduced from 0.015: max correction ~1.9mm instead of 3.6mm,
                     # prevents Stage 2 from pushing already-good points outside the 2mm threshold

# Landmarks that Stage 2 is applied to: only the ones with historical MRE > 1.7mm.
# The others (A, Pog, Ar, Go, Co, etc.) are already accurate — better not to touch them.
# Indices: 5=Or, 9=S, 12=Gn, 14=Po, 15=LPM, 16=LIT, 17=LMT, 18=UPM
# 19=UIT, 20=UIA, 21=LIA  <- difficult dental landmarks
# 22=Ls(UL), 23=Li(LL), 24=Pog`, 25=Sn  <- soft tissue (Aariz/Tracciati only, all difficult)
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
    def __init__(self, num_landmarks=26, in_channels=128):
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

        self.max_offset = MAX_OFFSET

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