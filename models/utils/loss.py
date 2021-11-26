import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class CELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss_ce = self.ce(input, target)
        return loss_ce


class MSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight
        self.loss = nn.MSELoss()
    
    def forward(self, input, target):
        l = self.loss(input, target)
        return l


class DICELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight
        self.dice = DiceLoss('multiclass')

    def forward(self, input, target):
        loss_dice = self.dice(input, target)
        return loss_dice


class CE_DICELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss('multiclass')

    def forward(self, input, target):
        loss_ce = self.ce(input, target)
        loss_dice = self.dice(input, target)
        return self.weight[0] * loss_ce + self.weight[1] * loss_dice

