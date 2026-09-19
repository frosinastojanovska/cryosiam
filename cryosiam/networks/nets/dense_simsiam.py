import torch
import torch.nn as nn
from monai.networks.nets import ResNet

from .fpn_decoder import FPNDecoder
from .bifpn_decoder import BiFPNDecoder


class DenseSimSiam(nn.Module):
    """
    Build a DenseSimSiam model.
    """

    def __init__(self, block_type='bottleneck', spatial_dims=3, n_input_channels=1,
                 num_layers=(1, 1, 1, 1), num_filters=(64, 128, 256, 512), no_max_pool=True, fpn_channels=128,
                 dim=2048, pred_dim=512, dense_dim=32, dense_pred_dim=64, include_levels=False,
                 add_later_conv=False, decoder_type='fpn', decoder_layers=2, decoder=True):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        dense_dim: feature dimension (default: 32) - pixel/voxel level
        dense_pred_dim: hidden dimension of the predictor (default: 64) - pixel/voxel level
        include_levels: use embeddings from every level in the FPN
        """
        super(DenseSimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.spatial_dims = spatial_dims
        self.encoder = ResNet(block_type, list(num_layers), list(num_filters),
                              n_input_channels=n_input_channels,
                              no_max_pool=no_max_pool,
                              num_classes=dim,
                              spatial_dims=self.spatial_dims,
                              act=("relu", {"inplace": False}))
        if block_type == 'bottleneck':
            expansion = 4
        else:
            expansion = 1

        if decoder:
            if decoder_type == 'fpn':
                self.decoder = FPNDecoder(num_filters=num_filters, spatial_dims=self.spatial_dims,
                                          out_channels=fpn_channels, expansion=expansion, add_later_conv=add_later_conv)
            else:
                self.decoder = BiFPNDecoder(num_filters=num_filters, spatial_dims=self.spatial_dims,
                                            out_channels=fpn_channels, expansion=expansion, num_layers=decoder_layers)

            # build a 3-layer projector
            prev_dim = self.encoder.fc.weight.shape[1]
            self.global_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                  nn.BatchNorm1d(prev_dim),
                                                  nn.ReLU(inplace=False),  # first layer
                                                  nn.Linear(prev_dim, prev_dim, bias=False),
                                                  nn.BatchNorm1d(prev_dim),
                                                  nn.ReLU(inplace=False),  # second layer
                                                  self.encoder.fc,
                                                  nn.BatchNorm1d(dim, affine=False))  # output layer
            self.global_projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

            # build a 2-layer predictor
            self.global_predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                                  nn.BatchNorm1d(pred_dim),
                                                  nn.ReLU(inplace=False),  # hidden layer
                                                  nn.Linear(pred_dim, dim))  # output layer

            if self.spatial_dims == 2:
                conv = nn.Conv2d
                norm = nn.BatchNorm2d
            else:
                conv = nn.Conv3d
                norm = nn.BatchNorm3d
            # build a 3-layer local projector
            self.projector = nn.Sequential(conv(fpn_channels, fpn_channels, 1, bias=False),
                                           norm(fpn_channels),
                                           nn.ReLU(inplace=False),
                                           conv(fpn_channels, fpn_channels, 1, bias=False),
                                           norm(fpn_channels),
                                           nn.ReLU(inplace=False),
                                           conv(fpn_channels, dense_dim, 1),
                                           norm(dense_dim, affine=False))

            # build a 2-layer local predictor
            self.predictor = nn.Sequential(conv(dense_dim, dense_pred_dim, 1, bias=False),
                                           norm(dense_pred_dim),
                                           nn.ReLU(inplace=False),
                                           conv(dense_pred_dim, dense_dim, 1))
            self.dense_criterion = nn.CosineSimilarity(dim=1)
            self.global_criterion = nn.CosineSimilarity(dim=1)

            self.include_levels = include_levels
            if self.include_levels:
                self.level_projector_2 = nn.Sequential(conv(fpn_channels, fpn_channels, 1, bias=False),
                                                       norm(fpn_channels),
                                                       nn.ReLU(inplace=False),
                                                       conv(fpn_channels, fpn_channels, 1, bias=False),
                                                       norm(fpn_channels),
                                                       nn.ReLU(inplace=False),
                                                       conv(fpn_channels, dense_dim, 1),
                                                       norm(dense_dim, affine=False))
                self.level_projector_4 = nn.Sequential(conv(fpn_channels, fpn_channels, 1, bias=False),
                                                       norm(fpn_channels),
                                                       nn.ReLU(inplace=False),
                                                       conv(fpn_channels, fpn_channels, 1, bias=False),
                                                       norm(fpn_channels),
                                                       nn.ReLU(inplace=False),
                                                       conv(fpn_channels, dense_dim, 1),
                                                       norm(dense_dim, affine=False))
                self.level_projector_8 = nn.Sequential(conv(fpn_channels, fpn_channels, 1, bias=False),
                                                       norm(fpn_channels),
                                                       nn.ReLU(inplace=False),
                                                       conv(fpn_channels, fpn_channels, 1, bias=False),
                                                       norm(fpn_channels),
                                                       nn.ReLU(inplace=False),
                                                       conv(fpn_channels, dense_dim, 1),
                                                       norm(dense_dim, affine=False))
                # build a 2-layer local predictor
                self.level_predictor_2 = nn.Sequential(conv(dense_dim, dense_pred_dim, 1, bias=False),
                                                       norm(dense_pred_dim),
                                                       nn.ReLU(inplace=False),
                                                       conv(dense_pred_dim, dense_dim, 1))
                self.level_predictor_4 = nn.Sequential(conv(dense_dim, dense_pred_dim, 1, bias=False),
                                                       norm(dense_pred_dim),
                                                       nn.ReLU(inplace=False),
                                                       conv(dense_pred_dim, dense_dim, 1))
                self.level_predictor_8 = nn.Sequential(conv(dense_dim, dense_pred_dim, 1, bias=False),
                                                       norm(dense_pred_dim),
                                                       nn.ReLU(inplace=False),
                                                       conv(dense_pred_dim, dense_dim, 1))
                self.level_criterion = nn.CosineSimilarity(dim=1)

    def forward(self, x1, x2):
        # compute features for one view
        feats1, feats1_global = self.get_encoder_features(x1)
        feats2, feats2_global = self.get_encoder_features(x2)

        z1_global = self.global_projector(feats1_global)  # NxC
        z2_global = self.global_projector(feats2_global)  # NxC
        p1_global = self.global_predictor(z1_global)  # NxC
        p2_global = self.global_predictor(z2_global)  # NxC

        if self.include_levels:
            feats1, o3_1, o2_1, o1_1 = self.decoder(feats1, True)
            feats2, o3_2, o2_2, o1_2 = self.decoder(feats2, True)
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)

            levels_z1_2 = self.level_projector_2(o3_1)
            levels_z1_4 = self.level_projector_4(o2_1)
            levels_z1_8 = self.level_projector_8(o1_1)
            levels_p1_2 = self.level_predictor_2(levels_z1_2)
            levels_p1_4 = self.level_predictor_4(levels_z1_4)
            levels_p1_8 = self.level_predictor_8(levels_z1_8)

            levels_z2_2 = self.level_projector_2(o3_2)
            levels_z2_4 = self.level_projector_4(o2_2)
            levels_z2_8 = self.level_projector_8(o1_2)
            levels_p2_2 = self.level_predictor_2(levels_z2_2)
            levels_p2_4 = self.level_predictor_4(levels_z2_4)
            levels_p2_8 = self.level_predictor_8(levels_z2_8)

            return p1, p2, z1.detach(), z2.detach(), p1_global, p2_global, z1_global.detach(), z2_global.detach(), \
                levels_p1_2, levels_p1_4, levels_p1_8, levels_p2_2, levels_p2_4, levels_p2_8, \
                levels_z1_2.detach(), levels_z1_4.detach(), levels_z1_8.detach(), \
                levels_z2_2.detach(), levels_z2_4.detach(), levels_z2_8.detach()
        else:
            feats1 = self.decoder(feats1)
            feats2 = self.decoder(feats2)
            z1 = self.projector(feats1)
            z2 = self.projector(feats2)
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
            return p1, p2, z1.detach(), z2.detach(), p1_global, p2_global, z1_global.detach(), z2_global.detach()

    def calculate_dense_loss(self, p, z, mask_p, mask_z):
        p = self.select_overlap_pixels(p, mask_p)
        z = self.select_overlap_pixels(z, mask_z)
        return self.dense_criterion(p, z)

    def calculate_levels_loss(self, levels_p, levels_z, mask_p, mask_z, num_levels=None):
        loss = []
        i = 0
        for p, z in zip(levels_p, levels_z):
            if num_levels is not None and i >= num_levels:
                break
            mask_p = torch.div(mask_p, 2).to(torch.int32)
            mask_z = torch.div(mask_z, 2).to(torch.int32)
            p = self.select_overlap_pixels(p, mask_p)
            z = self.select_overlap_pixels(z, mask_z)
            loss.append(self.level_criterion(p, z))
            i += 1
        return loss

    def calculate_global_loss(self, p, z):
        return self.global_criterion(p, z)

    @staticmethod
    def select_overlap_pixels(feats, mask):
        shape = feats.size()
        masked = None
        for i in range(shape[0]):
            current_mask = mask[i]
            if masked is None:
                if len(shape) > 4:
                    shape_mask = (shape[0], shape[1],
                                  current_mask[0][1] - current_mask[0][0] + 1,
                                  current_mask[1][1] - current_mask[1][0] + 1,
                                  current_mask[2][1] - current_mask[2][0] + 1)
                else:
                    shape_mask = (shape[0], shape[1],
                                  current_mask[0][1] - current_mask[0][0] + 1,
                                  current_mask[1][1] - current_mask[1][0] + 1)
                masked = torch.zeros(shape_mask, dtype=torch.float)
                masked = masked.to(feats.get_device())
            if len(shape) > 4:
                masked[i, :, :, :, :] = feats[i, :, current_mask[0][0]:current_mask[0][1] + 1,
                                        current_mask[1][0]:current_mask[1][1] + 1,
                                        current_mask[2][0]:current_mask[2][1] + 1]
            else:
                masked[i, :, :, :] = feats[i, :, current_mask[0][0]:current_mask[0][1] + 1,
                                     current_mask[1][0]:current_mask[1][1] + 1]
        return masked

    def forward_predict(self, x):
        feats, feats_global = self.get_encoder_features(x)
        z_global = self.global_projector(feats_global)

        feats = self.decoder(feats)
        z = self.projector(feats)
        return z, z_global

    def forward_predict_features(self, x):
        feats, feats_global = self.get_encoder_features(x)
        feats = self.decoder(feats)
        return feats, feats_global

    def forward_predict_levels(self, x):
        feats, feats_global = self.get_encoder_features(x)
        z_global = self.global_projector(feats_global)

        feats = self.decoder(feats, True)
        z = self.projector(feats[0])

        if self.include_levels:
            z_levels = [level_projector[i](feats[i + 1])
                        for i, level_projector in enumerate([self.level_projector_2,
                                                             self.level_projector_4,
                                                             self.level_projector_8])]
            return z, z_global, z_levels
        else:
            return z, z_global, feats[0:]

    def get_encoder_features(self, x):
        outputs = {}
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act(x)
        if not self.encoder.no_max_pool:
            x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        outputs['res2'] = x
        x = self.encoder.layer2(x)
        outputs['res3'] = x
        x = self.encoder.layer3(x)
        outputs['res4'] = x
        x = self.encoder.layer4(x)
        outputs['res5'] = x

        x_pooled = self.encoder.avgpool(x)
        x_pooled = x_pooled.view(x_pooled.size(0), -1)

        return outputs, x_pooled

    def get_encoder_features_list(self, x):
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

        return outputs, x_pooled
