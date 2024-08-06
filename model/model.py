# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    UNet model definition in here
"""

import torch
import torch.nn as nn
from base import BaseModel
from torch.optim import *
from torchvision import models

@torch.no_grad()
def check_model(topology, input_channels, num_classes, input_shape):
    model = CustomSegformer(topology=topology, input_channels=input_channels,
                 num_classes=num_classes)
    model.eval()
    in_tensor = torch.Tensor(*input_shape)
    with torch.no_grad():
        out_tensor, softmaxed = model(in_tensor)
        print(in_tensor.shape, out_tensor.shape)


if __name__ == '__main__':
    # check_model
    check_model(topology="ENC_1_DEC_1", input_channels=7,
                num_classes=2, input_shape=[4, 7, 64, 64])
    
from transformers import SegformerForSemanticSegmentation, SegformerModel, SegformerDecodeHead
from transformers.models.segformer.configuration_segformer import SegformerConfig

class CustomSegformer(nn.Module):
    def __init__(self, input_channels, num_classes, base_model = 'nvidia/mit-b0'):
        super().__init__()
        config = SegformerConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        config.num_channels = input_channels
        self.encoder = SegformerModel(config)
        self.decoder = SegformerDecodeHead(config)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in):
        outputs = self.encoder(x_in,
                               output_attentions = False,
                               output_hidden_states = True,
                               return_dict = True)
        
        logits = self.decoder(outputs.hidden_states)
        x = nn.functional.interpolate(logits, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
        return x, self.softmax(x)