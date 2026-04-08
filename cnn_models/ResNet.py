import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x