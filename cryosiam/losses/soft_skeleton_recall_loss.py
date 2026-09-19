import torch
import torch.nn as nn


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, smooth=1., class_weights=None):
        super().__init__()
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, x, y):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, y.shape)]):
                y_onehot = y if shp_x[1] == 1 else y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.float)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]

            y_onehot = y_onehot.to(x.dtype)
            sum_gt = y_onehot.sum(axes)

        if shp_x[1] > 1:
            x = x[:, 1:]

        inter_rec = (x * y_onehot).sum(axes)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        rec = (inter_rec + self.smooth) / torch.clip(sum_gt + self.smooth, 1e-8)

        if self.class_weights is not None:
            w = torch.as_tensor(self.class_weights, device=rec.device, dtype=rec.dtype)
            return -(rec * w).sum() / w.sum()
        return -rec.mean()