from torchvision.models.segmentation import deeplabv3_resnet50
from base import BaseModel
import torch

class DeeplabV3:
    def __init__(self, num_classes, pretrained=True):
        self.pretrained = pretrained
        self._load_model(num_classes)
        return

    def _load_model(self, num_classes):
        self.model = deeplabv3_resnet50(pretrained=self.pretrained, progress=True)
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        return