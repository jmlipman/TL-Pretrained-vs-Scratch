# This file contains all the loss functions that I've tested, including the
# proposed "Rectified normalized Region-wise map".
#
# To simplify and for clarity reasons, I've only commented those functions
# that appear in the paper. In any case, there is a lot of repetion because
# the class "BaseData" computes the "weights" (aka Region-wise map) based
# on the name of the loss function; therefore, different loss function names
# provide different weights.
import torch
import numpy as np
from torch import Tensor
from typing import List

def CrossEntropyLoss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Regular Cross Entropy loss function.
    It is possible to use weights with the shape of BWHD (no channel).

    Args:
      `y_pred`: Prediction of the model.
      `y_true`: labels one-hot encoded, BCWHD.

    Returns:
      Loss.
    """
    y_true = y_true[0]
    y_pred = y_pred[0]
    eps = 1e-10
    r = [1-0.017, 0.017]

    if len(y_true.shape) == 2:
        #loss = -torch.mean( r[0]*y_true*torch.log(y_pred +  eps) + r[1]*(1-y_true)*torch.log(1-y_pred + eps) )
        #loss = -torch.mean( y_true*torch.log(y_pred +  eps) + (1-y_true)*torch.log(1-y_pred + eps) )
        ce = torch.sum(y_true * torch.log(y_pred + eps), axis=1)
        #ce = r[0] * (y_true[:, 0] * torch.log(y_pred[:, 0] + eps)) + r[1]*(y_true[:, 1] * torch.log(y_pred[:, 1] + eps))
        return -torch.mean(ce)
    else:
        loss = 0
        for i in range(y_true.shape[1]):
            loss -= torch.mean( y_true[:, i]*torch.log(y_pred[:, i] +  eps) + (1-y_true[:, i])*torch.log(1-y_pred[:, i] + eps) )

    return loss

def CrossEntropyLoss_Distance(y_pred: Tensor, y_true: Tensor) -> Tensor:

    loss = CrossEntropyLoss(y_pred, y_true)

    distances = y_pred[1]
    res = 0
    for distance in distances:
        res += torch.mean(distance) / len(distances)

    lambda_ = 0
    t = loss + lambda_*res
    #print(t.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy())
    #print(t.cpu().detach().numpy())
    return t

