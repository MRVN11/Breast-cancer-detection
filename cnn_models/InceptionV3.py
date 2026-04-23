import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

class InceptionV3Model(nn.Module):
    def __init__(self):
        super(InceptionV3Model, self).__init__()
        self.base_model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.base_model.aux_logits = False  # Disable auxiliary output

        in_features = self.base_model.fc.in_features  # 2048 for InceptionV3
        self.base_model.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.base_model(x)  # Returns (batch, 2048) after fc=Identity
        x = self.fc(x)
        return x