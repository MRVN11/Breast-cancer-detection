import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models
from torchvision.models import DenseNet169_Weights


class DenseNet121Model(nn.Module):
    def __init__(self):
        super(DenseNet121Model, self).__init__()
        self.base_model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)

        # Read feature size dynamically
        in_features = self.base_model.classifier.in_features  # 1024
        self.base_model.classifier = nn.Identity()

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
        # Directly access feature extractor for speed
        features = self.base_model.features(x)
        x = f.relu(features, inplace=True)
        x = f.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x