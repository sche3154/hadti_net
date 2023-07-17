import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L1Loss(nn.Module):
    def __init__(self, weight=1):
        super(L1Loss, self).__init__()
        self.weight = weight
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        return self.weight * self.loss(pred, gt)



class FALoss(nn.Module):

    def __init__(self, weight=1):
        super(FALoss, self).__init__()
        self.weight = weight
        self.loss = torch.nn.L1Loss()

    def forward(self, pred, gt):
        return self.weight * self.loss(pred, gt)

    