"""
Adapted from: https://github.com/MishaLaskin/curl/blob/master/encoder.py
"""

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """
    Convolutional encoder of pixels observations.
    Hard-coded to match implementation details from 
    CURL Appendix B wherever possible, except for
    the output dim which is reduced here since
    Pong is visually non-complex.
    """
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(4, 32, 5, stride=5),
             nn.Conv2d(32, 64, 5, stride=5)]
        )

        self.fc = nn.Linear(384, 128)
        self.ln = nn.LayerNorm(128)

        self.outputs = dict()

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        conv = torch.relu(self.convs[1](conv))
        self.outputs['conv2'] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = h_norm

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
