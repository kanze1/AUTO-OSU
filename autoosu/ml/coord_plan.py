"""Predict a soft physical-exit-to-head spacing distribution from known music/rhythm.

This is a spacing planner, not a supervised direction/style classifier. Clean geometry
is used only by its loss. The geometry model always receives predicted distributions.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SpacingPlanner(nn.Module):
    def __init__(self, width, heads, components=4):
        super().__init__()
        self.components = components
        self.timing = nn.Linear(12, width)
        layer = nn.TransformerEncoderLayer(width, heads, 4*width, dropout=0.,
                                           batch_first=True, norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.parameters_head = nn.Linear(width, components*3)
        self.projection = nn.Linear(components*3, width)

    def forward(self, music, layout):
        nodes = music[:, layout['heads']] + self.timing(layout['features'])[None]
        parameters = self.parameters_head(self.blocks(nodes))
        logits, location, raw_scale = parameters.chunk(3, -1)
        scale = F.softplus(raw_scale) + torch.finfo(parameters.dtype).eps
        # Log probabilities and distribution parameters preserve multiple possible distances.
        distribution = torch.cat((logits.log_softmax(-1), location, scale), -1)
        return self.projection(distribution)[:, layout['owners']], (logits, location, scale)

    @staticmethod
    def loss(parameters, clean, layout):
        logits, location, scale = (p.float() for p in parameters)
        # Convert normalized x/y back to a common unit, 512 pixels.
        positions = (clean.transpose(1, 2)+1)*clean.new_tensor([.5, .375])
        heads, exits = positions[:, layout['heads']], positions[:, layout['exits']]
        distance = (heads[:, 1:]-exits[:, :-1]).norm(dim=-1).log1p()
        if distance.numel() == 0:
            return logits.sum()*0.
        log_density = (-.5*((distance[..., None]-location[:, 1:])/scale[:, 1:]).square()
                       -scale[:, 1:].log()-.5*math.log(2*math.pi))
        return -torch.logsumexp(logits[:, 1:].log_softmax(-1)+log_density, -1).mean()
