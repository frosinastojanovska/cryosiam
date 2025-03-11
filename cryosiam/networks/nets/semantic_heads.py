import torch.nn as nn


class SemanticHeads(nn.Module):
    """
    Build a SemanticHeads model.
    """

    def __init__(self, n_input_channels=1, num_classes=1, distance_pred=True, spatial_dims=3,
                 filters=(32, 64), kernel_size=3, padding=1):
        super().__init__()

        if spatial_dims == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        self.distance_pred = distance_pred
        if distance_pred:
            self.distance_head = nn.Sequential(conv(n_input_channels, filters[0], kernel_size, padding=padding),
                                               norm(filters[0]),
                                               nn.ReLU(inplace=False),
                                               conv(filters[0], filters[1], kernel_size, padding=padding),
                                               norm(filters[1]),
                                               nn.ReLU(inplace=False),
                                               conv(filters[1], num_classes, kernel_size, padding=padding))
        self.semantic_head = nn.Sequential(conv(n_input_channels, filters[0], kernel_size, padding=padding),
                                           norm(filters[0]),
                                           nn.ReLU(inplace=False),
                                           conv(filters[0], filters[1], kernel_size, padding=padding),
                                           norm(filters[1]),
                                           nn.ReLU(inplace=False),
                                           conv(filters[1], num_classes, kernel_size, padding=padding))

    def forward(self, x):
        output = self.semantic_head(x)
        if self.distance_pred:
            distances = self.distance_head(x)
            return output, distances
        return output
