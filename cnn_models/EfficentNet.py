import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

        self.base_model.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, model):
        model = self.base_model(model)
        model = self.fc(model)
        return model