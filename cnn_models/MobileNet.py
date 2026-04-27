import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()
        self.base_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x