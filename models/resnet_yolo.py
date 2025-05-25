import torch
import torch.nn as nn
import torchvision.models as models

class ResNetYOLO(nn.Module):
    def __init__(self, num_classes=80):  # COCO has 80 classes
        super(ResNetYOLO, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, (5 + num_classes) * 3, kernel_size=1)  # YOLO-style output
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
