import torch
import torch.nn as nn
import torchvision.models as models

class CephNet(nn.Module):
    def __init__(self, num_landmarks=29):
        super(CephNet, self).__init__()
        
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        vecchio_strato = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(vecchio_strato.weight.mean(dim=1, keepdim=True))
        
        numero_di_funzionalita = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(numero_di_funzionalita, num_landmarks * 2)
        )

    def forward(self, x):
        return self.backbone(x)