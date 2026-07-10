
import torch
import torch.nn as nn
import torchvision.models as models

class UNetResNet50(nn.Module):
    def __init__(self, num_landmarks=19, pretrained=True):
        super(UNetResNet50, self).__init__()

        weights  = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # [B, 256, 128, 128]
        self.layer2 = backbone.layer2  # [B, 512,  64,  64]
        self.layer3 = backbone.layer3  # [B,1024,  32,  32]
        self.layer4 = backbone.layer4  # [B,2048,  16,  16]

        # UP1: 16 -> 32
        self.up1  = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # UP2: 32 -> 64
        self.up2  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        # UP3: 64 -> 128
        self.up3  = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        # Output: 128x128 → num_landmarks heatmap
        # UNA SOLA final_conv (bug del doppio final_conv corretto qui)
        self.final_conv = nn.Conv2d(128, num_landmarks, kernel_size=1)

    def forward(self, x):
        x0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s1 = self.layer1(x0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        bn = self.layer4(s3)

        d1 = self.dec1(torch.cat([self.up1(bn), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s1], dim=1))

        return self.final_conv(d3)  # [B, num_landmarks, 128, 128]

class UNetResNet50v2(nn.Module):
    def __init__(self, num_landmarks=19, pretrained=True):
        super(UNetResNet50v2, self).__init__()

        weights  = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # UP1: 16 -> 32
        self.up1  = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),          nn.BatchNorm2d(512), nn.ReLU(True),
        )
        # UP2: 32 -> 64
        self.up2  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),        nn.BatchNorm2d(256), nn.ReLU(True),
        )
        # UP3: 64 -> 128
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),        nn.BatchNorm2d(128), nn.ReLU(True),
        )
        # UP4: 128 -> 256 (NUOVO)
        self.up4  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        )

        self.final_conv = nn.Conv2d(64, num_landmarks, 1)

    def forward(self, x):
        x0 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s1 = self.layer1(x0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        bn = self.layer4(s3)

        d1 = self.dec1(torch.cat([self.up1(bn), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s1], dim=1))
        d4 = self.dec4(self.up4(d3))

        return self.final_conv(d4)  # [B, num_landmarks, 256, 256]