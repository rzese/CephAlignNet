import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_OK = True
except ImportError:
    TIMM_OK = False

# Number of channels each backbone feature branch is projected to
PROJ_CHANNELS = 48


class HRNetLandmark(nn.Module):
    """
    Stage 1 model: HRNet-W48 backbone (via timm) + a lightweight fusion head
    that predicts, for every landmark, a heatmap channel plus two local
    offset channels (x, y). Output shape: [B, num_landmarks * 3, H, W].

    The three channels per landmark are consumed downstream as:
      - channel 0: heatmap (probability map used for soft-argmax decoding)
      - channel 1: x-offset (sub-pixel refinement, used by regression voting)
      - channel 2: y-offset (sub-pixel refinement, used by regression voting)
    """
    def __init__(self, num_landmarks=26, pretrained=True, hrnet_channels=None):
        super().__init__()

        if not TIMM_OK:
            raise ImportError("timm is not installed. Run: pip install timm")

        # in_chans=1: single grayscale channel in, returns the 4 correct
        # feature branches [192, 384, 768, 1536] of HRNet-W48 instead of
        # 5 spurious feature maps.
        self.backbone = timm.create_model(
            "hrnet_w48", pretrained=pretrained,
            features_only=True, in_chans=1,
        )

        if hrnet_channels is None:
            # Auto-detect channel widths of each feature branch with a dummy
            # forward pass, so the head can be built with the right shapes.
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 512, 512)
                feats = self.backbone(dummy)
                hrnet_channels = [f.shape[1] for f in feats]

        print(f"Backbone: HRNet-W48 | Detected channels: {hrnet_channels}")
        self.hrnet_channels = hrnet_channels

        # Project every feature branch to a common PROJ_CHANNELS width so
        # they can be concatenated ("fused") along the channel dimension.
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, PROJ_CHANNELS, kernel_size=1, bias=False),
                nn.BatchNorm2d(PROJ_CHANNELS),
                nn.ReLU(inplace=True),
            )
            for c in hrnet_channels
        ])

        fused_channels = len(hrnet_channels) * PROJ_CHANNELS

        # Prediction head: fused multi-scale features -> heatmap + offsets
        self.head = nn.Sequential(
            nn.Conv2d(fused_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks * 3, kernel_size=1),
        )

    def forward(self, x):
        # x is already [B, 1, H, W] — no replication to 3 channels needed.
        features = self.backbone(x)
        n_proj = len(self.proj)
        if len(features) > n_proj:
            # Keep only the last n_proj branches if the backbone returns more
            features = features[len(features) - n_proj:]

        h, w = features[0].shape[2], features[0].shape[3]

        fused = []
        for i, feat in enumerate(features):
            p = self.proj[i](feat)
            if p.shape[2] != h or p.shape[3] != w:
                p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            fused.append(p)

        x_out   = torch.cat(fused, dim=1)# fused multi-scale feature map, also returned
                                        # for use by the Stage 2 refinement network
        heatmap = self.head(x_out)
        return heatmap, x_out