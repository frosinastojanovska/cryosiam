import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    def __init__(self, apply_nonlin=None):
        super().__init__()
        self.apply_nonlin = apply_nonlin

    def forward(self, x, dist_maps):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        dist_maps = -dist_maps.to(device=x.device, dtype=x.dtype)
        return (x * dist_maps).mean()
