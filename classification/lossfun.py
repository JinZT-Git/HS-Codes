import torch
import os
import torch.nn.functional as F
from config import Config as cfg

class_weight = cfg.class_weight
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_device


def cross_entropy_loss(preds, target, reduction):
    logp = F.log_softmax(preds, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


def onehot_encoding(labels, n_classes):
    return torch.zeros(labels.size(0), n_classes).to(labels.device).scatter_(
        dim=1, index=labels.view(-1, 1), value=1)


def label_smoothing(preds, targets, epsilon=0.1):
    # preds为网络最后一层输出的logits
    # targets为未one-hot的真实标签
    n_classes = preds.size(1)
    device = preds.device

    onehot = onehot_encoding(targets, n_classes).float().to(device)
    targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(
        device) * epsilon / n_classes
    loss = cross_entropy_loss(preds, targets, reduction="mean")
    return loss


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        criterion = []
        for i in range(8):
            w = torch.FloatTensor(class_weight[i]).cuda()
            criterion.append(torch.nn.CrossEntropyLoss(weight=w))
        loss = 0
        for _ in (0, len(output)):
            for i in range(0, 8):
                loss += criterion[i](output[i], label[i])
        return loss