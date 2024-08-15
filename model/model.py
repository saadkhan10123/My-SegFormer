# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    Segformer definition here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.optim import *
from transformers import SegformerModel, SegformerDecodeHead
from transformers.models.segformer.configuration_segformer import SegformerConfig

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
    


class CustomSegformer(nn.Module):
    def __init__(self, input_channels, num_classes, base_model = 'nvidia/mit-b0', load_pretrained = False, 
                 hidden_dropout_prob = None, attention_probs_dropout_prob = None,
                 classifier_dropout_prob = None, drop_path_rate = None):
        super().__init__()
        self.base_model = base_model
        config = SegformerConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        config.num_channels = input_channels
        if hidden_dropout_prob is not None:
            config.hidden_dropout_prob = hidden_dropout_prob
        if attention_probs_dropout_prob is not None:
            config.attention_probs_dropout_prob = attention_probs_dropout_prob
        if classifier_dropout_prob is not None:
            config.classifier_dropout_prob = classifier_dropout_prob
        if drop_path_rate is not None:
            config.drop_path_rate = drop_path_rate
        self.encoder = SegformerModel(config)
        self.decoder = SegformerDecodeHead(config)
        self.softmax = nn.Softmax(dim=1)

        if load_pretrained:
            self.init_weights()

    def forward(self, x_in):
        outputs = self.encoder(x_in,
                               output_attentions = False,
                               output_hidden_states = True,
                               return_dict = True)
        
        logits = self.decoder(outputs.hidden_states)
        x = nn.functional.interpolate(logits, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
        return x, self.softmax(x)
    
    def init_weights(self):
        import torch.nn.functional as F

def init_weights(self):
    base_encoder = SegformerModel.from_pretrained(self.base_model)
    base_decoder = SegformerDecodeHead.from_pretrained(self.base_model)

    # Get the state dictionaries
    encoder_state_dict = base_encoder.state_dict()
    decoder_state_dict = base_decoder.state_dict()

    # Update encoder weights
    for name, param in self.encoder.named_parameters():
        if name in encoder_state_dict:
            if param.shape == encoder_state_dict[name].shape:
                param.data.copy_(encoder_state_dict[name].data)
            else:
                param.data.copy_(F.interpolate(encoder_state_dict[name].data.unsqueeze(0), size=param.shape[1:], mode='bilinear', align_corners=False).squeeze(0))

    # Update decoder weights
    for name, param in self.decoder.named_parameters():
        if name in decoder_state_dict:
            if param.shape == decoder_state_dict[name].shape:
                param.data.copy_(decoder_state_dict[name].data)
            else:
                param.data.copy_(F.interpolate(decoder_state_dict[name].data.unsqueeze(0), size=param.shape[1:], mode='bilinear', align_corners=False).squeeze(0))