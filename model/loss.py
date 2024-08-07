# # Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# # All rights reserved.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class FocalLoss2d(nn.Module):
#     # output : NxCxHxW float tensor
#     # target :  NxHxW long tensor
#     # weights : C float tensor
#     def __init__(self, gamma=2, weight=None):
#         super(FocalLoss2d, self).__init__()
#         self.gamma = gamma
#         self.nll_loss = nn.NLLLoss(weight)

#     # def forward(self, inputs, targets):

#     #     return self.nll_loss((1 - F.softmax(inputs, dim=1)) ** self.gamma * F.log_softmax(inputs, dim=1), targets)

#     def forward(self, inputs, targets):
#         # Ensure gamma is a float to maintain the float type of tensors
#         #gamma = float(self.gamma)

#         gamma = gamma.type(torch.cuda.FloatTensor)
#         # Calculate the focal loss as per the original formula, ensuring all operations produce float tensors
#         focal_loss = (1 - F.softmax(inputs, dim=1)) ** gamma * F.log_softmax(inputs, dim=1)
#         # Ensure targets are of type Long for nn.NLLLoss
#         targets = targets.long()
#         return self.nll_loss(focal_loss, targets)

# def check_focal_loss2d():
#     num_c = 3
#     weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
#     out_x_np = np.random.randint(0, num_c, size=(16*64*64*num_c)).reshape((16, num_c, 64, 64))
#     target_np = np.random.randint(0, num_c, size=(16*64*64*1)).reshape((16, 64, 64))
#     logits = torch.Tensor(out_x_np)
#     target = torch.Tensor(target_np)
#     loss_val = focal_loss2d(logits, target, weight=weights)
#     print("Focalloss2d: ", loss_val.item())

# def focal_loss2d(output, target, weights=None):
#     if not weights:
#         weights = torch.Tensor([10, 10])
#         weights = weights.cuda() if torch.cuda.is_available() else weights
#     return FocalLoss2d(weight=weights)(output, target)

# if __name__ == '__main__':
#     check_focal_loss2d()


# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss2d(nn.Module):
    # output : NxCxHxW float tensor
    # target : NxHxW long tensor
    # weights : C float tensor
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        # Ensure gamma is a float to maintain the float type of tensors
        gamma = float(self.gamma)
        # Calculate the focal loss as per the original formula, ensuring all operations produce float tensors
        focal_loss = (1 - F.softmax(inputs, dim=1)) ** gamma * F.log_softmax(inputs, dim=1)
        # Ensure targets are of type Long for nn.NLLLoss
        targets = targets.long()
        return self.nll_loss(focal_loss, targets)

def check_focal_loss2d():
    num_c = 3
    weights = torch.Tensor([7, 2, 241, 500, 106, 5, 319, 0.06, 0.58, 0.125, 0.045, 0.18, 0.026, 0.506, 0.99, 0.321])
    out_x_np = np.random.randint(0, num_c, size=(16*64*64*num_c)).reshape((16, num_c, 64, 64))
    target_np = np.random.randint(0, num_c, size=(16*64*64*1)).reshape((16, 64, 64))
    logits = torch.Tensor(out_x_np)
    target = torch.Tensor(target_np)
    loss_val = focal_loss2d(logits, target, weight=weights)
    print("Focalloss2d: ", loss_val.item())

def focal_loss2d(output, target, weights=None):
    if weights is None:
        weights = torch.Tensor([10, 10])
        weights = weights.cuda() if torch.cuda.is_available() else weights
    return FocalLoss2d(weight=weights)(output, target)

if __name__ == '__main__':
    check_focal_loss2d()

def cross_entropy(output, target):
    target = target.long()  # Ensure target is of type Long
    return F.cross_entropy(output, target)