import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        self.base_model.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
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