import torch
from torch.autograd import Function
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target, num): 
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum() 
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=1, reduce_th=0.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce_th = reduce_th
        self.eps = 1e-6

    def forward(self, input, target):
        assert input.size() == target.size()
        input = input[:, 0]
        target = target[:, 0]
        y_pred = input.contiguous().view(-1)
        y_true = target.contiguous().view(-1)
        y_pred = torch.clamp(y_pred, self.eps, 1.0)
        log_pt = -F.binary_cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(log_pt)
        th_pt = torch.where(
            pt < self.reduce_th,
            torch.ones_like(pt),
            (((1 - pt) / (1 - self.reduce_th)) ** self.gamma),
        )
        loss = -self.alpha * th_pt * log_pt
        return torch.sum(loss) / torch.sum(target)


class DiceSensitivityLoss(nn.Module):

    def __init__(self):
        super(DiceSensitivityLoss, self).__init__()

    def forward(self, input, target):
        assert input.size() == target.size() # [2, 1, 512, 512]
        num = target.size(0)
        # input = input[:, 0]
        # target = target[:, 0]
        # y_pred = input.contiguous().view(-1) # [524288]
        # y_true = target.contiguous().view(-1)
        # y_pred = torch.sigmoid(y_pred)
        y_pred = torch.sigmoid(input).view(num, -1)
        y_true = target.view(num, -1)
        TP = (y_pred * y_true).sum(-1)
        TN = ((1-y_pred)*(1- y_true)).sum(-1)
        FP = (y_pred * (1-y_true)).sum(-1)
        FN = ((1-y_pred)* y_true).sum(-1)
        DSC = (2*TP) / (2*TP + FP + FN)
        loss = 1 -DSC.sum()/num
        return loss

class DiceSensitivityLoss2(nn.Module):

    def __init__(self):
        super(DiceSensitivityLoss2, self).__init__()

    def forward(self, input, target):
        assert input.size() == target.size() 
        num = target.size(0)
        y_pred = torch.sigmoid(input).view(num, -1)
        y_true = target.view(num, -1)
        TP = (y_pred * y_true).sum(-1)
        TN = ((1-y_pred)*(1- y_true)).sum(-1)
        FP = (y_pred * (1-y_true)).sum(-1)
        FN = ((1-y_pred)* y_true).sum(-1)
        DSC = (2*TP) / (2*TP + FP + FN)
        SEN = TP / (TP + FN)
        loss = 2 - DSC.sum()/num - SEN.sum()/num
        return loss