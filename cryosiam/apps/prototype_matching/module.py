import os
import collections
import pickle

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data.utils import worker_init_fn
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    OneOf,
    Identityd,
    LoadImaged,
    RandRotated,
    EnsureTyped,
    SpatialPadd,
    RandFlipd,
    RandZoomd,
    CenterSpatialCropd,
    RandScaleIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
)
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from monai.losses import GeneralizedDiceLoss

from cryosiam.data import MrcReader, VoxelContrastiveDataset
from cryosiam.data.voxel_contrastive_dataset import episodic_collate
from cryosiam.networks.nets import DenseSimSiam, PrototypeSimilarityFPN
from cryosiam.transforms import (
    RandomLowPassBlurd,
    RandomGaussianNoised,
    RandomHighPassSharpend,
)
from cryosiam.utils import patch_train_val_split


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t()
    B = L.shape[1]
    K = L.shape[0]
    L = L / torch.sum(L)
    for _ in range(sinkhorn_iterations):
        L = L / torch.sum(L, dim=1, keepdim=True) / K
        L = L / torch.sum(L, dim=0, keepdim=True) / B
    L = L * B
    L = L.t()
    indexs = torch.argmax(L, dim=1)
    q = F.gumbel_softmax(L, tau=0.5, hard=True)
    return q, indexs


class MultiPrototypeBank(nn.Module):
    def __init__(self, n_classes, k_per_class, feat_dim, momentum=0.999):
        super().__init__()
        self.n_classes = n_classes
        self.k = k_per_class
        self.momentum = momentum
        protos = torch.empty(n_classes + 1, k_per_class, feat_dim)
        nn.init.trunc_normal_(protos, std=0.02)
        self.register_buffer('prototypes', protos)

    @torch.no_grad()
    def normalize_(self):
        self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))

    @torch.no_grad()
    def update(self, cls, k, feat):
        feat = F.normalize(feat.detach(), dim=0)
        new = self.momentum * self.prototypes[cls, k] + (1 - self.momentum) * feat
        self.prototypes[cls, k] = F.normalize(new, dim=0)


class PrototypeMatchingModule(pl.LightningModule):
    def __init__(self, config, dense_backbone_config):
        super().__init__()
        self.config = config
        self.dense_backbone_config = dense_backbone_config

        dense_net_cfg = dense_backbone_config['parameters']['network']
        decoder_cfg = config['parameters']['network']

        self.temperature = float(decoder_cfg.get('temperature', 0.1))
        self.k_per_class = int(config.get('k_per_class', 5))
        self.sinkhorn_iterations = int(decoder_cfg.get('sinkhorn_iterations', 3))
        self.sinkhorn_epsilon = float(decoder_cfg.get('sinkhorn_epsilon', 0.05))

        self._backbone = DenseSimSiam(block_type=dense_net_cfg['block_type'],
                                      n_input_channels=dense_net_cfg['in_channels'],
                                      spatial_dims=dense_net_cfg['spatial_dims'],
                                      num_layers=dense_net_cfg['num_layers'],
                                      num_filters=dense_net_cfg['num_filters'],
                                      no_max_pool=dense_net_cfg['no_max_pool'],
                                      fpn_channels=dense_net_cfg['fpn_channels'],
                                      dim=dense_net_cfg['dim'],
                                      pred_dim=dense_net_cfg['pred_dim'],
                                      dense_dim=dense_net_cfg['dense_dim'],
                                      dense_pred_dim=dense_net_cfg['dense_pred_dim'],
                                      decoder=False)

        embed_dim = decoder_cfg.get('embed_dim', decoder_cfg.get('out_channels', 128))
        self._decoder = PrototypeSimilarityFPN(feat_channels=dense_net_cfg['num_filters'],
                                               out_channels=embed_dim,
                                               use_context_head=decoder_cfg.get('use_context_head', False),
                                               predict_at_c3=decoder_cfg.get('predict_at_c3', False))
        self.embed_dim = embed_dim

        self.use_distance_loss = bool(decoder_cfg.get('use_distance_loss', False))
        self.distance_weight = float(decoder_cfg.get('distance_weight', 0.1))
        self.distance_clip = float(decoder_cfg.get('distance_clip', 8))

        if self.use_distance_loss:
            self.distance_head = nn.Sequential(nn.Conv3d(self.embed_dim + 1, self.embed_dim, 3, padding=1),
                                               nn.ReLU(inplace=False),
                                               nn.Conv3d(self.embed_dim, self.embed_dim, 3, padding=1),
                                               nn.ReLU(inplace=False),
                                               nn.Conv3d(self.embed_dim, 1, 1))

        self._load_pretrained(config)

        self.freeze_encoder = not decoder_cfg.get('unfreeze_encoder', False)
        if self.freeze_encoder:
            for p in self._backbone.parameters():
                p.requires_grad = False
            self._backbone.eval()
        print(f'Encoder: {"frozen" if self.freeze_encoder else "trainable"}')

        self.use_ppc_loss = bool(decoder_cfg.get('use_ppc_loss', True))
        self.ppc_weight = float(decoder_cfg.get('ppc_weight', 0.01))
        self.use_ppd_loss = bool(decoder_cfg.get('use_ppd_loss', True))
        self.ppd_weight = float(decoder_cfg.get('ppd_weight', 0.01))
        print(f'PPC loss: {"ENABLED" if self.use_ppc_loss else "disabled"} (weight={self.ppc_weight})')
        print(f'PPD loss: {"ENABLED" if self.use_ppd_loss else "disabled"} (weight={self.ppd_weight})')
        print(f'Distance loss: {"ENABLED" if self.use_distance_loss else "disabled"} '
              f'(weight={self.distance_weight}, clip={self.distance_clip})')

        self.seg_ce = nn.CrossEntropyLoss()

        self.seg_dice = GeneralizedDiceLoss(include_background=False,
                                            to_onehot_y=True,
                                            softmax=True,
                                            reduction='mean')

        self.proto_momentum = float(decoder_cfg.get('proto_momentum', 0.999))
        self.proto_bank = None  # built in setup(), once the dataset's class set is known
        self.label_lookup = None
        self.idx_to_label = {}

        self.batch_size = config['hyper_parameters']['batch_size']
        self.lr = config['hyper_parameters']['lr']
        self.optimizer_name = config['hyper_parameters']['optimizer']
        self.weight_decay = config['hyper_parameters']['weight_decay']
        self.max_epochs = config['hyper_parameters']['max_epochs']
        self.momentum = config['hyper_parameters'].get('momentum', None)
        self.sync_dist = int(config['parameters']['gpu_devices']) > 1

        self.save_hyperparameters()

    def _load_pretrained(self, config):
        def _load(model, prefix, state):
            weights = {k.replace(prefix, ''): v
                       for k, v in state.items() if k.startswith(prefix)}
            miss, unexp = model.load_state_dict(weights, strict=False)
            if miss:
                print(f'  [{prefix.strip("._")}] missing:    {miss[:3]}...')
            if unexp:
                print(f'  [{prefix.strip("._")}] unexpected: {unexp[:3]}...')

        if 'pretrained_model' in config:
            state = torch.load(config['pretrained_model'], weights_only=False)['state_dict']
            for enc_prefix in ('_encoder.', '_search_model.', '_backbone.'):
                if any(k.startswith(enc_prefix) for k in state):
                    _load(self._backbone, enc_prefix, state)
                    break
            _load(self._decoder, '_decoder.', state)
            print(f'Loaded backbone + FPN from {os.path.basename(config["pretrained_model"])}')
        elif 'pretrained_dense_simsiam_model' in config:
            ckpt = torch.load(config['pretrained_dense_simsiam_model'], weights_only=False)
            state = collections.OrderedDict(
                {k.replace('_model.', ''): v for k, v in ckpt['state_dict'].items()})
            self._backbone.load_state_dict(state, strict=False)
            print(f'Loaded backbone from {os.path.basename(config["pretrained_dense_simsiam_model"])} '
                  f'(FPN start fresh)')
        else:
            print('WARNING: no pretrained weights — training from scratch')

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self._backbone.eval()
        return self

    def setup_search_transforms(self):
        keys = ['image', 'mask']
        patch_size = self.config['parameters']['data']['patch_size']
        t = self.config['parameters']['transforms']
        return Compose([
            LoadImaged(keys=keys, reader=MrcReader(writable=False)),
            EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
            SpatialPadd(keys=keys, spatial_size=patch_size),
            ScaleIntensityRanged(keys=['image'],
                                 a_min=self.config['parameters']['data']['min'],
                                 a_max=self.config['parameters']['data']['max'],
                                 b_min=0, b_max=1, clip=True),
            OneOf(transforms=self.get_image_transforms()),
            RandZoomd(keys=keys, prob=0.8, min_zoom=t['zoom'][0], max_zoom=t['zoom'][1],
                      mode=['bilinear', 'nearest'], padding_mode='constant')
            if t.get('zoom') else Identityd(keys=['image']),
            RandRotated(keys=keys, prob=0.8, range_x=t['rotate'][0], range_y=t['rotate'][1],
                        range_z=t['rotate'][2], mode=['bilinear', 'nearest'], padding_mode='zeros')
            if t.get('rotate') else Identityd(keys=['image']),
            CenterSpatialCropd(keys=keys, roi_size=patch_size),
            RandFlipd(keys=keys, prob=0.8, spatial_axis=-1)
            if t.get('flip', True) else Identityd(keys=['image']),
            RandFlipd(keys=keys, prob=0.8, spatial_axis=-2)
            if t.get('flip', True) else Identityd(keys=['image']),
            RandFlipd(keys=keys, prob=0.8, spatial_axis=-3)
            if t.get('flip', True) else Identityd(keys=['image']),
            RandScaleIntensityd(['image'], prob=0.8, factors=t['scale_intensity_factors'])
            if t.get('scale_intensity_factors') else Identityd(keys=['image']),
            NormalizeIntensityd(keys=['image'],
                                subtrahend=self.config['parameters']['data']['mean'],
                                divisor=self.config['parameters']['data']['std']),
            EnsureTyped(keys=['image'], data_type='tensor', dtype=torch.float, track_meta=False),
            EnsureTyped(keys=['mask'], data_type='tensor', dtype=torch.long, track_meta=False)])

    def setup_search_val_transforms(self):
        keys = ['image', 'mask']
        patch_size = self.config['parameters']['data']['patch_size']
        return Compose([
            LoadImaged(keys=keys, reader=MrcReader(writable=False)),
            EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
            SpatialPadd(keys=keys, spatial_size=patch_size),
            ScaleIntensityRanged(keys=['image'],
                                 a_min=self.config['parameters']['data']['min'],
                                 a_max=self.config['parameters']['data']['max'],
                                 b_min=0, b_max=1, clip=True),
            CenterSpatialCropd(keys=keys, roi_size=patch_size),
            NormalizeIntensityd(keys=['image'],
                                subtrahend=self.config['parameters']['data']['mean'],
                                divisor=self.config['parameters']['data']['std']),
            EnsureTyped(keys=['image'], data_type='tensor', dtype=torch.float, track_meta=False),
            EnsureTyped(keys=['mask'], data_type='tensor', dtype=torch.long, track_meta=False)])

    def get_image_transforms(self):
        t = self.config['parameters']['transforms']
        transforms = [
            RandomLowPassBlurd(keys=['image'], prob=1.0, sigma=t['low_pass_sigma_range'])
            if t.get('low_pass_sigma_range') else None,
            RandomHighPassSharpend(keys=['image'], prob=1.0,
                                   sigma=t['high_pass_sigma_range'],
                                   sigma2=t['high_pass_sigma2_range'])
            if t.get('high_pass_sigma_range') else None,
            RandomGaussianNoised(keys=['image'], prob=1.0, sigma=t['noise_sigma_range'])
            if t.get('noise_sigma_range') else None,
            Compose([
                RandomLowPassBlurd(keys=['image'], prob=1.0, sigma=t['low_pass_sigma_range']),
                RandomHighPassSharpend(keys=['image'], prob=1.0,
                                       sigma=t['high_pass_sigma_range'],
                                       sigma2=t['high_pass_sigma2_range']),
                RandomGaussianNoised(keys=['image'], prob=1.0, sigma=t['noise_sigma_range'])])
            if t.get('combine_transforms') else None,
            Identityd(keys=['image'])]
        return [x for x in transforms if x is not None]

    def prepare_data(self):
        search_root = os.path.normpath(self.config['patches_folder'])
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        if not os.path.isfile(train_val_path):
            train_files, val_files = patch_train_val_split(
                os.path.join(search_root, 'images'),
                os.path.join(search_root, 'masks'),
                ratio=self.config['validation_ratio'],
                file_ext=self.config['file_extension'])
            with open(train_val_path, 'wb') as f:
                pickle.dump({'train_files': train_files, 'val_files': val_files}, f)

    def setup(self, stage=None):
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        with open(train_val_path, 'rb') as f:
            data = pickle.load(f)
        train_files, val_files = data['train_files'], data['val_files']
        set_determinism(seed=0)

        data_cfg = self.config['parameters']['data']
        shared = dict(min_foreground=data_cfg.get('min_foreground', 10),
                      exclude_labels=data_cfg.get('exclude_labels', None))

        if self.use_distance_loss:
            shared['use_distance_loss'] = True
            shared['distance_clip'] = self.distance_clip

        self.train_ds = VoxelContrastiveDataset(search_files=train_files,
                                                search_transform=self.setup_search_transforms(),
                                                is_val=False,
                                                samples_per_class=data_cfg.get('samples_per_class', 200),
                                                **shared)
        self.val_ds = VoxelContrastiveDataset(search_files=val_files,
                                              search_transform=self.setup_search_val_transforms(),
                                              is_val=True,
                                              samples_per_class=data_cfg.get('val_samples_per_class', 20),
                                              **shared)

        train_labels = set(self.train_ds.unique_labels)
        val_labels = set(self.val_ds.unique_labels)
        if train_labels != val_labels:
            print(f'  WARNING: train/val class sets differ.\n'
                  f'    train-only: {sorted(train_labels - val_labels)}\n'
                  f'    val-only:   {sorted(val_labels - train_labels)}')

        unique_labels = sorted(train_labels | val_labels)
        self.idx_to_label = {i + 1: lbl for i, lbl in enumerate(unique_labels)}
        max_label = max(unique_labels) if unique_labels else 0
        lookup = torch.zeros(max_label + 1, dtype=torch.long)
        for i, lbl in enumerate(unique_labels):
            lookup[lbl] = i + 1
        self.label_lookup = lookup.to(self.device)

        self.proto_bank = MultiPrototypeBank(
            n_classes=len(unique_labels), k_per_class=self.k_per_class,
            feat_dim=self.embed_dim, momentum=self.proto_momentum).to(self.device)

        print(f'Datasets prepared ({len(unique_labels)} classes total, '
              f'K={self.k_per_class} prototypes each)')

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          collate_fn=episodic_collate, shuffle=True,
                          num_workers=10, persistent_workers=True,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          collate_fn=episodic_collate, num_workers=4,
                          persistent_workers=True,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False, drop_last=True)

    def configure_optimizers(self):
        params = list(self._decoder.parameters())
        if self.use_distance_loss:
            params += list(self.distance_head.parameters())
        if not self.freeze_encoder:
            params += list(self._backbone.parameters())
            print('Optimizing encoder + FPN')
        else:
            print('Optimizing FPN only')

        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params, self.lr, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params, self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, self.lr, weight_decay=self.weight_decay)

        scheduler = OneCycleLR(optimizer, max_lr=self.lr,
                               total_steps=self.trainer.estimated_stepping_batches,
                               anneal_strategy='cos')
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def _forward(self, image):
        if self.freeze_encoder:
            with torch.no_grad():
                feats_list, _ = self._backbone.get_encoder_features_list(image)
        else:
            feats_list, _ = self._backbone.get_encoder_features_list(image)
        return self._decoder(feats_list, output_size=image.shape[-3:])

    def _run_episode(self, support_embed_b, support_mask_b, query_embed_b, query_mask_b,
                     metric_class, is_train, query_distance_b=None):

        C = support_embed_b.shape[0]

        self.proto_bank.normalize_()
        protos_now = self.proto_bank.prototypes.clone()

        # ---------------------------------------------------------
        # SUPPORT: update prototypes
        # ---------------------------------------------------------
        flat_supp_feat = support_embed_b.reshape(C, -1).T
        flat_supp_mask = support_mask_b.reshape(-1)
        sims_supp = torch.einsum('mc,nkc->mnk', flat_supp_feat, protos_now)

        # M x (N_classes + 1)
        out_seg_supp = sims_supp.max(dim=2).values
        with torch.no_grad():
            pred_supp = out_seg_supp.argmax(dim=1)
        correct_supp = (pred_supp == flat_supp_mask)
        n_proto_classes = self.proto_bank.prototypes.shape[0]

        for cls in range(n_proto_classes):
            cls_sel = (flat_supp_mask == cls)
            if cls_sel.sum() == 0:
                continue
            init_q = sims_supp[cls_sel, cls, :]
            q, indexs = distributed_sinkhorn(init_q.detach(),
                                             sinkhorn_iterations=self.sinkhorn_iterations,
                                             epsilon=self.sinkhorn_epsilon)
            correct_k = correct_supp[cls_sel].float()
            m_q = q * correct_k.unsqueeze(1)
            c_q = flat_supp_feat[cls_sel] * correct_k.unsqueeze(1)
            if is_train and self.trainer.is_global_zero:
                f = m_q.t() @ c_q
                n = m_q.sum(dim=0)
                if n.sum() > 0:
                    f_norm = F.normalize(f, dim=-1)
                    for kk in range(self.k_per_class):
                        if n[kk] > 0:
                            self.proto_bank.update(cls, kk, f_norm[kk])

        self.proto_bank.normalize_()
        protos_updated = self.proto_bank.prototypes.clone()
        D, H, W = query_mask_b.shape
        flat_query_feat = query_embed_b.reshape(C, -1).T
        flat_query_mask = query_mask_b.reshape(-1)

        sims_query = torch.einsum('mc,nkc->mnk', flat_query_feat, protos_updated)

        # nearest sub-prototype of each class
        out_seg_query = sims_query.max(dim=2).values
        # Fixed temperature, no calibrator
        class_logits = out_seg_query / self.temperature

        ce_loss = self.seg_ce(class_logits, flat_query_mask.long())
        logits_spatial = class_logits.reshape(D, H, W, n_proto_classes).permute(3, 0, 1, 2).unsqueeze(0)

        target_spatial = query_mask_b.unsqueeze(0).unsqueeze(0)
        dice_loss = self.seg_dice(logits_spatial, target_spatial)

        seg_loss_b = ce_loss + dice_loss
        proto_target_query = torch.full_like(flat_query_mask, -1)

        for cls in range(n_proto_classes):
            cls_sel = (flat_query_mask == cls)
            if cls_sel.sum() == 0:
                continue
            init_q = sims_query[cls_sel, cls, :]
            q, indexs = distributed_sinkhorn(init_q.detach(),
                                             sinkhorn_iterations=self.sinkhorn_iterations,
                                             epsilon=self.sinkhorn_epsilon)
            proto_target_query[cls_sel] = indexs + self.k_per_class * cls

        raw_proto_logits = sims_query.reshape(flat_query_feat.shape[0], -1)
        proto_logits_query = raw_proto_logits / self.temperature
        valid = (proto_target_query != -1)

        ppc_loss_b = None
        ppd_loss_b = None

        if self.use_ppc_loss and valid.any():
            ppc_loss_b = F.cross_entropy(proto_logits_query, proto_target_query.long(), ignore_index=-1)

        if self.use_ppd_loss and valid.any():
            gathered = raw_proto_logits[valid].gather(1, proto_target_query[valid].unsqueeze(1).long()).squeeze(1)
            ppd_loss_b = (1 - gathered).pow(2).mean()

        distance_loss_b = None

        if self.use_distance_loss:
            if query_distance_b is None:
                raise ValueError('query_distance is required when use_distance_loss=True')

            target_similarity = out_seg_query[:, metric_class].reshape(
                1, 1, D, H, W)
            distance_input = torch.cat([
                query_embed_b.unsqueeze(0),
                target_similarity,
            ], dim=1)
            predicted_distance = torch.tanh(
                self.distance_head(distance_input)
            ).squeeze(0).squeeze(0)

            inside = query_distance_b > 0
            near_outside = ((query_distance_b < 0) &
                            (query_distance_b > -1))
            distance_terms = []

            if inside.any():
                distance_terms.append(
                    F.smooth_l1_loss(predicted_distance[inside],
                                     query_distance_b[inside]))

            if near_outside.any():
                distance_terms.append(
                    F.smooth_l1_loss(predicted_distance[near_outside],
                                     query_distance_b[near_outside]))

            if distance_terms:
                distance_loss_b = torch.stack(distance_terms).mean()

        with torch.no_grad():
            pred = class_logits.argmax(dim=1)
            pred_bin = (pred == metric_class).float()
            target_bin = (flat_query_mask == metric_class).float()
            inter = (pred_bin * target_bin).sum().item()
            denom = pred_bin.sum().item() + target_bin.sum().item()

            dice = 2 * inter / denom if denom > 0 else 1.0

        return seg_loss_b, ppc_loss_b, ppd_loss_b, distance_loss_b, dice

    def _compute_loss(self, batch, is_train: bool):
        support_image = batch['support_image'].to(self.device)
        query_image = batch['query_image'].to(self.device)
        support_mask_raw = batch['support_mask'].to(self.device)[:, 0]
        query_mask_raw = batch['query_mask'].to(self.device)[:, 0]
        class_id_raw = batch['class_id'].to(self.device)
        query_distance = None
        if self.use_distance_loss:
            if 'query_distance' not in batch:
                raise KeyError('Batch has no query_distance. Update '
                               'VoxelContrastiveDataset and episodic_collate '
                               'for use_distance_loss=True.')
            query_distance = batch['query_distance'].to(self.device)

        B = support_image.shape[0]
        embed_all = self._forward(torch.cat([support_image, query_image], dim=0))
        support_embed, query_embed = embed_all[:B], embed_all[B:]

        seg_losses, ppc_losses, ppd_losses = [], [], []
        distance_losses, dices = [], []
        for b in range(B):
            cls_raw = int(class_id_raw[b].item())
            metric_class = int(self.label_lookup[cls_raw].item())

            support_mask_b = self.label_lookup[
                support_mask_raw[b].long()
            ]

            query_mask_b = self.label_lookup[
                query_mask_raw[b].long()
            ]

            query_distance_b = (query_distance[b]
                                if query_distance is not None else None)

            seg_loss_b, ppc_loss_b, ppd_loss_b, distance_loss_b, dice = self._run_episode(
                support_embed[b], support_mask_b, query_embed[b], query_mask_b,
                metric_class, is_train, query_distance_b)

            seg_losses.append(seg_loss_b)
            if ppc_loss_b is not None:
                ppc_losses.append(ppc_loss_b)
            if ppd_loss_b is not None:
                ppd_losses.append(ppd_loss_b)
            if distance_loss_b is not None:
                distance_losses.append(distance_loss_b)
            dices.append(dice)

        seg_loss = torch.stack(seg_losses).mean()
        ppc_loss = torch.stack(ppc_losses).mean() if ppc_losses else None
        ppd_loss = torch.stack(ppd_losses).mean() if ppd_losses else None
        distance_loss = (torch.stack(distance_losses).mean()
                         if distance_losses else None)

        total = seg_loss
        if ppc_loss is not None:
            total = total + self.ppc_weight * ppc_loss
        if ppd_loss is not None:
            total = total + self.ppd_weight * ppd_loss
        if distance_loss is not None:
            total = total + self.distance_weight * distance_loss

        return total, seg_loss, ppc_loss, ppd_loss, distance_loss, dices, class_id_raw

    def training_step(self, batch, batch_idx):
        total, seg_loss, ppc_loss, ppd_loss, distance_loss, *_ = self._compute_loss(
            batch, is_train=True)
        self.log('train_loss', total, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppc_loss is not None:
            self.log('train_ppc_loss', ppc_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppd_loss is not None:
            self.log('train_ppd_loss', ppd_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_loss is not None:
            self.log('train_distance_loss', distance_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        return total

    def on_train_epoch_start(self):
        if hasattr(self.train_ds, 'rebuild_index'):
            self.train_ds.rebuild_index()

    def on_validation_epoch_start(self):
        set_determinism(seed=42)
        self._val_dice_sum = collections.defaultdict(float)
        self._val_dice_count = collections.defaultdict(int)

    def validation_step(self, batch, batch_idx):
        total, seg_loss, ppc_loss, ppd_loss, distance_loss, dices, class_id_raw = self._compute_loss(
            batch, is_train=False)
        self.log('val_loss', total, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        self.log('val_seg_loss', seg_loss, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppc_loss is not None:
            self.log('val_ppc_loss', ppc_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppd_loss is not None:
            self.log('val_ppd_loss', ppd_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_loss is not None:
            self.log('val_distance_loss', distance_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)

        for cls_raw, dice in zip(class_id_raw.tolist(), dices):
            self._val_dice_sum[cls_raw] += dice
            self._val_dice_count[cls_raw] += 1

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        class_dice = {cls: self._val_dice_sum[cls] / self._val_dice_count[cls]
                      for cls in self._val_dice_sum if self._val_dice_count[cls] > 0}
        if not class_dice:
            print(f'Epoch {self.current_epoch}: no val dice accumulated')
            return

        dice_values = list(class_dice.values())
        nonzero = sum(1 for d in dice_values if d > 0.01)
        self.log('val_dice_mean', float(np.mean(dice_values)), sync_dist=False)
        self.log('val_dice_nonzero_classes', nonzero, sync_dist=False)

        sorted_dice = sorted(class_dice.items(), key=lambda x: x[1])
        print(f'\n--- Epoch {self.current_epoch} val ({len(class_dice)}/{len(self.idx_to_label)} classes) ---')
        print('  Dice — Bottom 5:')
        for label, dice in sorted_dice[:5]:
            print(f'    class {label:3d}: {dice:.4f}')
        print('  Dice — Top 5:')
        for label, dice in sorted_dice[-5:]:
            print(f'    class {label:3d}: {dice:.4f}')
        print(f'  Mean Dice: {np.mean(dice_values):.4f}  Non-zero: {nonzero}/{len(self.idx_to_label)}')
