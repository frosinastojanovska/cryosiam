import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import ResNet


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, block_type='bottleneck', n_input_channels=1, num_layers=(1, 1, 1, 1), spatial_dims=3,
                 num_filters=(64, 128, 256, 512), no_max_pool=True, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder = ResNet(block_type, list(num_layers), list(num_filters),
                              n_input_channels=n_input_channels,
                              no_max_pool=no_max_pool,
                              num_classes=dim,
                              spatial_dims=spatial_dims,
                              act=("relu", {"inplace": False}))

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=False),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=False),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=False),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward_one(self, x):
        z = self.encoder(x)
        p = self.predictor(z)
        p = F.normalize(p, p=2, dim=1)
        return z, p

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

    def get_encoder_features(self, x):
        outputs = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act(x)
        if not self.encoder.no_max_pool:
            x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        outputs.append(x)  # res2
        x = self.encoder.layer2(x)
        outputs.append(x)  # res3
        x = self.encoder.layer3(x)
        outputs.append(x)  # res4
        x = self.encoder.layer4(x)
        outputs.append(x)  # res5

        x_pooled = self.encoder.avgpool(x)

        x_pooled = x_pooled.view(x_pooled.size(0), -1)
        if self.encoder.fc is not None:
            x_pooled = self.encoder.fc(x_pooled)

        return outputs, x_pooled
