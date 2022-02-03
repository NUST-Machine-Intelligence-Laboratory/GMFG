import torch
import numpy as np
import random
import torch.nn.functional as F

def L_loss(logit, logit_l, reduction='mean'):
    label = F.softmax(logit_l, dim=1).detach()
    prob, _ = torch.max(label, 1)
    N = label.size(0)
    log_logit = F.log_softmax(logit, dim=1)
    losses = -torch.sum(log_logit * label, dim=1) * prob # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')