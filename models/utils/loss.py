import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentationLosses(object):
    def __init__(self, beta=1, weight=None, cuda=False):
        self.weight = weight
        self.beta = beta
        assert self.weight is None or sum(self.weight) == 2 or sum(self.weight) == 3
        print("Using loss weights: ", self.weight)

        self.cuda = cuda
        self.verbose = True

    def build_loss(self, mode='ce'):
        print("Using {} loss with {} beta".format(mode, self.beta))

        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'ce_dice':
            return self.CE_DICELoss
        elif mode == 'wce_dice':
            return self.WCE_DICELoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
    
        logit = logit.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.reshape(-1)

        loss = torch.nn.functional.cross_entropy(logit, target)

        return loss

    def CrossEntropyLoss_Manual(self, logit, target, weights=None):
        n, c, h, w = logit.size()
        # Calculate log probabilities
        logits_log_softmax = F.log_softmax(logit, dim=1).float()
        logits_log_probs = logits_log_softmax.gather(dim=1, index=target.view(n, 1, h, w).long()) #n, 1, h, w

        # Multiply by exp(weights) [ weights on scale of 0-1, but taking exponent gives 1-e]
        if weights is None:
            weights = torch.zeros_like(logits_log_probs)
        else:
            weights = weights.unsqueeze(1) # weights arrive as n, h, w

        weights_exp = torch.exp(weights) ** 2 # [0 - 1] --> [1 e**3=20]
        # print(torch.unique(weights_exp))
        assert weights_exp.size() == logits_log_probs.size()
        logits_weighted_log_probs = (logits_log_probs * weights_exp).view(n, -1)

        # Rescale the weights so loss is in approximately the same interval (distribution of weights may have a lot of variance)
        weighted_loss = logits_weighted_log_probs.sum(1) / weights_exp.view(n, -1).sum(1)

        # Return mini-batch mean
        return -1 * weighted_loss.mean() # log probs are negative for incorrect predictions and 0 for perfect. need to minimize not maximize

    # https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    def DICELoss(self, logit, target, eps=1e-7):
        logit = logit.exp()

        assert len(logit.shape) == 4
        assert len(target.shape) == 3

        eps = 0.0001
        encoded_target = logit.detach() * 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        return self.calc_dice(logit, encoded_target)

    def calc_dice(self, logit, encoded_target, eps=1e-7):
        intersection = logit * encoded_target
        numerator = (1 + self.beta**2) * intersection.sum(0).sum(1).sum(1)
        denominator = logit + encoded_target

        denominator = (self.beta**2) * denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = (1 - (numerator / denominator))

        return loss_per_channel.sum() / logit.size(1)

    def CE_DICELoss(self, logit, target):
        cross_entropy = self.CrossEntropyLoss(logit, target)
        dice_loss = self.DICELoss(F.log_softmax(logit, dim=1), target)
        
        return self.weight[0]*cross_entropy + self.weight[1]*dice_loss

    def WCE_DICELoss(self, logit, target, weight=None):
        wce = self.CrossEntropyLoss_Manual(logit, target, weight)
        dice_loss = self.DICELoss(F.log_softmax(logit, dim=1), target)

        return self.weight[0]*wce + self.weight[1]*dice_loss


