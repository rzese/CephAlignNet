import torch
import torch.nn as nn
import torchvision.models as models


class CephNetHeatmap(nn.Module):
    def __init__(self, num_landmarks=19):
        super(CephNetHeatmap, self).__init__()
        self.num_landmarks = num_landmarks

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        vecchio_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(vecchio_conv1.weight.mean(dim=1, keepdim=True))

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512 + 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.output = nn.Conv2d(128, num_landmarks, kernel_size=1)

    def forward(self, x):
        s1 = self.layer1(self.layer0(x))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        x4 = self.layer4(s3)
        x  = self.conv4(torch.cat([self.up4(x4), s3], dim=1))
        x  = self.conv3(torch.cat([self.up3(x),  s2], dim=1))
        x  = self.conv2(torch.cat([self.up2(x),  s1], dim=1))
        return self.output(x)