import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeSimilarityFPN(nn.Module):
    def __init__(self, feat_channels, out_channels: int,
                 predict_at_c3: bool = False,
                 use_context_head: bool = False,
                 use_dual_head: bool = False,
                 use_distance_head: bool = False,
                 num_classes: int = None,
                 distance_channels: int = None,
                 seg_head_hidden: int = None):
        super().__init__()
        assert len(feat_channels) == 4, (
            'feat_channels must be [c2, c3, c4, c5]'
        )

        c2_ch, c3_ch, c4_ch, c5_ch = feat_channels
        self.out_channels = out_channels
        self.predict_at_c3 = predict_at_c3
        self.use_context_head = use_context_head
        self.use_dual_head = use_dual_head
        self.use_distance_head = use_distance_head

        self.lat5 = nn.Conv3d(c5_ch, out_channels, 1)
        self.lat4 = nn.Conv3d(c4_ch, out_channels, 1)
        self.lat3 = nn.Conv3d(c3_ch, out_channels, 1)
        self.conv5 = nn.Conv3d(
            out_channels, out_channels, 3, padding=1, bias=True
        )
        self.conv4 = nn.Conv3d(
            out_channels, out_channels, 3, padding=1, bias=True
        )
        self.conv3 = nn.Conv3d(
            out_channels, out_channels, 3, padding=1, bias=True
        )

        self.lat2 = nn.Conv3d(c2_ch, out_channels, 1)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 3, padding=1, bias=True
        )

        if predict_at_c3:
            self.bottom_up_p2_to_p3 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )
            self.bottom_up_fuse = nn.Conv3d(
                2 * out_channels, out_channels, kernel_size=1
            )

        if use_context_head:
            self.context_head = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
            )

        if use_dual_head:
            if num_classes is None:
                raise ValueError(
                    'num_classes must be given when use_dual_head=True'
                )
            hidden = seg_head_hidden or out_channels
            self.seg_head = nn.Sequential(
                nn.Conv3d(
                    out_channels, hidden, 3, padding=1, bias=False
                ),
                nn.BatchNorm3d(hidden),
                nn.ReLU(inplace=False),
                nn.Conv3d(hidden, num_classes, 1),
            )

        if use_distance_head:
            if distance_channels is None:
                raise ValueError(
                    'distance_channels must be given when '
                    'use_distance_head=True'
                )
            self.distance_head = nn.Sequential(
                nn.Conv3d(
                    out_channels, out_channels, 3,
                    padding=1, bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv3d(
                    out_channels, out_channels, 3,
                    padding=1, bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv3d(
                    out_channels, distance_channels, 3, padding=1
                ),
            )

    def _upsample_add(self, x, y):
        return (
                F.interpolate(
                    x,
                    size=y.shape[-3:],
                    mode='trilinear',
                    align_corners=False,
                )
                + y
        )

    def _fpn_body(self, search_feats):
        s2, s3, s4, s5 = search_feats
        p5 = F.relu(self.conv5(self.lat5(s5)))
        p4 = F.relu(
            self.conv4(self._upsample_add(p5, self.lat4(s4)))
        )
        p3_pre = self.conv3(
            self._upsample_add(p4, self.lat3(s3))
        )

        if self.predict_at_c3:
            p3 = F.relu(p3_pre)
            p2_pre = self.conv2(
                self._upsample_add(p3, self.lat2(s2))
            )
            p2_down = self.bottom_up_p2_to_p3(F.relu(p2_pre))
            fused = torch.cat([p3_pre, p2_down], dim=1)
            return self.bottom_up_fuse(fused)

        p3 = F.relu(p3_pre)
        return self.conv2(
            self._upsample_add(p3, self.lat2(s2))
        )

    def forward(self, search_feats, output_size=None):
        raw_feats = self._fpn_body(search_feats)

        if self.use_context_head:
            raw_feats = self.context_head(raw_feats)

        if output_size is not None:
            raw_feats = F.interpolate(
                raw_feats,
                size=output_size,
                mode='trilinear',
                align_corners=False,
            )

        norm_feats = F.normalize(raw_feats, dim=1)
        seg_logits = (
            self.seg_head(raw_feats) if self.use_dual_head else None
        )
        distance_logits = (
            self.distance_head(raw_feats)
            if self.use_distance_head
            else None
        )

        # Preserve the original return values when the distance head is off.
        if self.use_dual_head and self.use_distance_head:
            return norm_feats, seg_logits, distance_logits
        if self.use_dual_head:
            return norm_feats, seg_logits
        if self.use_distance_head:
            return norm_feats, distance_logits
        return norm_feats
