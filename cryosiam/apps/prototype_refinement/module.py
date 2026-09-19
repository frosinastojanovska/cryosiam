import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F
import torch.distributed as dist
from monai.data import list_data_collate, Dataset, NumpyReader
from monai.data.utils import worker_init_fn
from monai.losses import GeneralizedDiceLoss
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
from monai.utils import set_determinism
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from cryosiam.data import MrcReader
from cryosiam.networks.nets import DenseSimSiam, PrototypeSimilarityFPN
from cryosiam.losses import BoundaryLoss
from cryosiam.transforms import (
    ClipIntensityd,
    RandomLowPassBlurd,
    RandomGaussianNoised,
    RandomHighPassSharpend,
)
from cryosiam.utils import patch_train_val_split


@torch.no_grad()
def sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
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
    def separate_(self, margin=0.98, strength=0.001):
        """Gently repel near-identical prototypes within each class."""
        protos = F.normalize(self.prototypes, dim=-1)
        pairwise = torch.einsum('nkc,njc->nkj', protos, protos)

        diagonal = torch.eye(self.k, dtype=torch.bool, device=protos.device).unsqueeze(0)

        excess = F.relu(pairwise - margin)
        excess = excess.masked_fill(diagonal, 0.0)

        # Direction from prototype j towards prototype i.
        differences = (protos.unsqueeze(2) - protos.unsqueeze(1))
        directions = F.normalize(differences, dim=-1)

        repulsion = (excess.unsqueeze(-1) * directions).sum(dim=2)
        self.prototypes.copy_(F.normalize(protos + strength * repulsion, dim=-1))

    @torch.no_grad()
    def update(self, cls, k, feat):
        feat = F.normalize(feat.detach(), dim=0)
        new = self.momentum * self.prototypes[cls, k] + (1 - self.momentum) * feat
        self.prototypes[cls, k] = F.normalize(new, dim=0)


class PrototypeRefinementModule(pl.LightningModule):

    def __init__(self, config, dense_backbone_config):
        super().__init__()
        self.config = config
        self.dense_backbone_config = dense_backbone_config

        dense_net_cfg = dense_backbone_config['parameters']['network']
        decoder_cfg = config['parameters']['network']

        self.class_names = list(config['class_names'])
        self.n_classes = len(self.class_names)
        self.k_per_class = int(config.get('k_per_class', 5))
        self.temperature = float(decoder_cfg.get('temperature', 0.1))
        self.sinkhorn_iterations = int(decoder_cfg.get('sinkhorn_iterations', 3))
        self.sinkhorn_epsilon = float(decoder_cfg.get('sinkhorn_epsilon', 0.05))
        self.use_dual_head = bool(decoder_cfg.get('use_dual_head', False))
        self.use_distance_head = bool(decoder_cfg.get('use_distance_head', False))
        self.use_boundary_loss = bool(config.get('use_boundary_loss', False))
        self.need_distance_maps = (
                self.use_distance_head or self.use_boundary_loss
        )
        self.distance_weight = float(config.get('distance_weight', 0.1))
        self.boundary_weight = float(config.get('boundary_weight', 0.01))

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

        embed_dim = self._read_embed_dim(config)
        self._decoder = PrototypeSimilarityFPN(feat_channels=dense_net_cfg['num_filters'],
                                               out_channels=embed_dim,
                                               use_context_head=decoder_cfg.get('use_context_head', False),
                                               predict_at_c3=decoder_cfg.get('predict_at_c3', False),
                                               use_dual_head=self.use_dual_head,
                                               use_distance_head=self.use_distance_head,
                                               num_classes=self.n_classes + 1 if self.use_dual_head else None,
                                               distance_channels=self.n_classes if self.use_distance_head else None,
                                               seg_head_hidden=decoder_cfg.get('seg_head_hidden'))
        self.embed_dim = embed_dim

        self._load_pretrained(config)

        self.freeze_encoder = not decoder_cfg.get(
            'unfreeze_backbone', decoder_cfg.get('unfreeze_encoder', False))
        self.freeze_fpn = not decoder_cfg.get('unfreeze_decoder', True)

        if self.freeze_encoder:
            for p in self._backbone.parameters():
                p.requires_grad = False
            self._backbone.eval()

        if self.freeze_fpn:
            for name, p in self._decoder.named_parameters():
                if self.use_dual_head and name.startswith('seg_head.'):
                    continue
                if self.use_distance_head and name.startswith('distance_head.'):
                    continue
                p.requires_grad = False
            self._decoder.eval()

        print(f'Encoder: {"frozen" if self.freeze_encoder else "trainable"}')
        print(f'FPN:     {"frozen" if self.freeze_fpn else "trainable"}')
        print(f'Classes: {self.class_names}  (+ background)')
        print(f'K prototypes per class: {self.k_per_class}')
        print(f'Boundary loss: {"ENABLED" if self.use_boundary_loss else "disabled"}'
              + (f' (weight={self.boundary_weight})' if self.use_boundary_loss else ''))
        print(f'Distance head: {"ENABLED" if self.use_distance_head else "disabled"}'
              + (f' (weight={self.distance_weight})' if self.use_distance_head else ''))

        self.semantic_loss = nn.CrossEntropyLoss()

        self.seg_loss = GeneralizedDiceLoss(include_background=False,
                                            to_onehot_y=True,
                                            sigmoid=False,
                                            softmax=True,
                                            reduction='mean')

        if self.use_boundary_loss:
            self.boundary_loss = BoundaryLoss()

        if self.use_distance_head:
            # The stored targets are clipped/scaled signed distances in [-1, 1].
            # Smooth L1 is less sensitive than MSE to imperfect object masks.
            self.distance_loss = nn.SmoothL1Loss()

        self.proto_momentum = float(config.get('proto_momentum', 0.999))
        self.proto_bank = MultiPrototypeBank(
            n_classes=self.n_classes, k_per_class=self.k_per_class,
            feat_dim=embed_dim, momentum=self.proto_momentum)

        self.use_ppc_loss = bool(config.get('use_ppc_loss', True))
        self.ppc_weight = float(config.get('ppc_weight', 0.01))
        self.use_ppd_loss = bool(config.get('use_ppd_loss', True))
        self.ppd_weight = float(config.get('ppd_weight', 0.001))
        self.use_proto_separation = bool(config.get('use_proto_separation', True))
        self.proto_div_margin = float(config.get('proto_div_margin', 0.95))
        self.proto_div_strength = float(config.get('proto_div_strength', 0.001))
        print(f'PPC loss: {"ENABLED" if self.use_ppc_loss else "disabled"} (weight={self.ppc_weight})')
        print(f'PPD loss: {"ENABLED" if self.use_ppd_loss else "disabled"} (weight={self.ppd_weight})')
        print(f'Prototype separation: '
              f'{"ENABLED" if self.use_proto_separation else "disabled"} '
              f'(margin={self.proto_div_margin}, strength={self.proto_div_strength})')

        self.aux_proto_seg_weight = float(config.get('aux_proto_seg_weight', 0.1))
        print(f'Dual-head seg: {"ENABLED" if self.use_dual_head else "disabled"}'
              + (f' (aux_proto_seg_weight={self.aux_proto_seg_weight})' if self.use_dual_head else ''))

        self.batch_size = config['hyper_parameters']['batch_size']
        self.lr = config['hyper_parameters']['lr']
        self.optimizer_name = config['hyper_parameters']['optimizer']
        self.weight_decay = config['hyper_parameters']['weight_decay']
        self.max_epochs = config['hyper_parameters']['max_epochs']
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

        if 'pretrained_model' not in config:
            print('WARNING: no pretrained weights — training from scratch')
            return

        print(f'Loading: {os.path.basename(config["pretrained_model"])}')
        state = torch.load(config['pretrained_model'], weights_only=False)['state_dict']
        for enc_prefix in ('_encoder.', '_search_model.', '_backbone.'):
            if any(k.startswith(enc_prefix) for k in state):
                _load(self._backbone, enc_prefix, state)
                break
        _load(self._decoder, '_decoder.', state)
        print('  Backbone + FPN loaded ✓')

    @staticmethod
    def _read_embed_dim(config):
        decoder_cfg = config['parameters']['network']

        if 'pretrained_model' in config:
            ckpt = torch.load(config['pretrained_model'], map_location='cpu', weights_only=False)
            state = ckpt['state_dict']
            if '_decoder.lat5.weight' in state:
                inferred = state['_decoder.lat5.weight'].shape[0]
                config_val = decoder_cfg.get('embed_dim', decoder_cfg.get('out_channels'))
                if config_val is not None and int(config_val) != inferred:
                    print(f'  WARNING: config embed_dim/out_channels={config_val} but the '
                          f'pretrained checkpoint\'s FPN weights imply {inferred} -- using '
                          f'the checkpoint-derived value, config value ignored.')
                return inferred
            print('  WARNING: pretrained_model given but no "_decoder.lat5.weight" found in '
                  'its state_dict -- falling back to config/default for embed_dim.')

        if 'embed_dim' in decoder_cfg or 'out_channels' in decoder_cfg:
            return decoder_cfg.get('embed_dim', decoder_cfg.get('out_channels', 128))
        return 128

    def _image_augments(self):
        t = self.config['parameters']['transforms']
        ts = [
            RandomLowPassBlurd(keys=['image'], prob=1.0, sigma=t['low_pass_sigma_range'])
            if t.get('low_pass_sigma_range') else None,
            RandomHighPassSharpend(keys=['image'], prob=1.0,
                                   sigma=t['high_pass_sigma_range'],
                                   sigma2=t['high_pass_sigma2_range'])
            if t.get('high_pass_sigma_range') else None,
            RandomGaussianNoised(keys=['image'], prob=1.0, sigma=t['noise_sigma_range'])
            if t.get('noise_sigma_range') else None,
            Identityd(keys=['image']),
        ]
        return [x for x in ts if x is not None]

    def _build_transforms(self, augment: bool):
        patch_size = self.config['parameters']['data']['patch_size']
        t = self.config['parameters']['transforms']
        spatial_keys = ['image', 'mask']
        interp_modes = ['trilinear', 'nearest']

        # distances participate in every SPATIAL transform (pad/crop/flip/
        # rotate/zoom) alongside image/mask, matching
        # SpecializedParticlePickingModule's own pattern exactly -- a
        # distance map is only meaningful if it stays spatially aligned
        # with the mask it was computed from.
        if self.need_distance_maps:
            spatial_keys.append('distances')
            interp_modes.append('trilinear')

        load_keys = ['image', 'mask']
        ops = [
            LoadImaged(keys=load_keys, reader=MrcReader(writable=False)),
            EnsureChannelFirstd(keys=load_keys, channel_dim='no_channel'),
        ]

        if self.need_distance_maps:
            ops += [
                LoadImaged(keys=['distances'],
                           reader=NumpyReader(npz_keys='data', channel_dim=0)),
                ClipIntensityd(keys=['distances'], a_min=-5, a_max=5),
                ScaleIntensityRanged(keys=['distances'], a_min=-5, a_max=5,
                                     b_min=-1, b_max=1, clip=True),
            ]

        ops += [
            SpatialPadd(keys=spatial_keys, spatial_size=patch_size),
            ScaleIntensityRanged(keys=['image'],
                                 a_min=self.config['parameters']['data']['min'],
                                 a_max=self.config['parameters']['data']['max'],
                                 b_min=0, b_max=1, clip=True),
        ]

        if augment:
            ops.append(OneOf(transforms=self._image_augments()))
            if t.get('zoom'):
                ops.append(RandZoomd(keys=spatial_keys, prob=0.8,
                                     min_zoom=t['zoom'][0], max_zoom=t['zoom'][1],
                                     mode=interp_modes, padding_mode='constant'))
            if t.get('rotate'):
                ops.append(RandRotated(keys=spatial_keys, prob=0.8,
                                       range_x=t['rotate'][0], range_y=t['rotate'][1],
                                       range_z=t['rotate'][2],
                                       mode=interp_modes, padding_mode='zeros'))
            if t.get('flip', True):
                for axis in (-1, -2, -3):
                    ops.append(RandFlipd(keys=spatial_keys, prob=0.8, spatial_axis=axis))
            if t.get('scale_intensity_factors'):
                ops.append(RandScaleIntensityd(keys=['image'], prob=0.8,
                                               factors=t['scale_intensity_factors']))

        ops += [
            CenterSpatialCropd(keys=spatial_keys, roi_size=patch_size),
            NormalizeIntensityd(keys=['image'],
                                subtrahend=self.config['parameters']['data']['mean'],
                                divisor=self.config['parameters']['data']['std']),
            EnsureTyped(keys=['image'], data_type='tensor', dtype=torch.float),
            EnsureTyped(keys=['mask'], data_type='tensor', dtype=torch.long),
        ]
        if self.need_distance_maps:
            ops.append(EnsureTyped(keys=['distances'], data_type='tensor', dtype=torch.float32))

        return Compose(ops)

    def prepare_data(self):
        root = os.path.normpath(self.config['patches_folder'])
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        if not os.path.isfile(train_val_path):
            train_files, val_files = patch_train_val_split(
                images_folder=os.path.join(root, 'images'),
                masks_folder=os.path.join(root, 'masks'),
                ratio=self.config['validation_ratio'],
                file_ext=self.config['file_extension'])

            with open(train_val_path, 'wb') as f:
                pickle.dump({'train_files': train_files, 'val_files': val_files}, f)

    def _add_distance_paths(self, files):
        if not self.need_distance_maps:
            return

        root = os.path.normpath(self.config['patches_folder'])
        dist_folder = os.path.join(root, 'distances')
        file_ext = self.config['file_extension']

        for sample in files:
            base = os.path.basename(sample['image']).replace(file_ext, '')
            distance_path = os.path.join(dist_folder, f'{base}.npz')
            if not os.path.isfile(distance_path):
                raise FileNotFoundError(
                    f'Missing distance map for {sample["image"]}: '
                    f'{distance_path}'
                )
            sample['distances'] = distance_path

    def setup(self, stage=None):
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        with open(train_val_path, 'rb') as f:
            data = pickle.load(f)
        train_files, val_files = data['train_files'], data['val_files']
        self._add_distance_paths(train_files)
        self._add_distance_paths(val_files)

        set_determinism(seed=0)
        self.train_ds = Dataset(data=train_files, transform=self._build_transforms(augment=True))
        self.val_ds = Dataset(data=val_files, transform=self._build_transforms(augment=False))

        print(f'Train: {len(self.train_ds)}  Val: {len(self.val_ds)} patches')

    def on_fit_start(self):
        self.proto_bank.normalize_()
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.proto_bank.prototypes, src=0)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          collate_fn=list_data_collate, shuffle=True,
                          num_workers=10, persistent_workers=True,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          collate_fn=list_data_collate, num_workers=4,
                          persistent_workers=True,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False, drop_last=False)

    def configure_optimizers(self):
        param_groups = []

        if self.use_dual_head:
            seg_head_params = [p for n, p in self._decoder.named_parameters()
                               if n.startswith('seg_head.')]
            param_groups.append({'params': seg_head_params, 'lr': self.lr})

        if self.use_distance_head:
            distance_head_params = [
                p for n, p in self._decoder.named_parameters()
                if n.startswith('distance_head.')
            ]
            param_groups.append({
                'params': distance_head_params,
                'lr': self.lr,
            })

        if not self.freeze_fpn:
            head_prefixes = []
            if self.use_dual_head:
                head_prefixes.append('seg_head.')
            if self.use_distance_head:
                head_prefixes.append('distance_head.')
            fpn_params = [p for n, p in self._decoder.named_parameters()
                          if not any(n.startswith(prefix)
                                     for prefix in head_prefixes)]
            param_groups.append({'params': fpn_params, 'lr': self.lr})

        if not self.freeze_encoder:
            param_groups.append({'params': self._backbone.parameters(), 'lr': self.lr})

        if not param_groups:
            raise RuntimeError('Both backbone and FPN are frozen; '
                               'there are no trainable parameters.')

        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=self.lr,
                                        momentum=self.config['hyper_parameters'].get('momentum', 0.9),
                                        weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = OneCycleLR(optimizer, max_lr=[g.get('lr', self.lr) for g in optimizer.param_groups],
                               total_steps=self.trainer.estimated_stepping_batches,
                               anneal_strategy='cos')
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self._backbone.eval()
        if self.freeze_fpn:
            self._decoder.eval()
            if self.use_dual_head:
                self._decoder.seg_head.train(mode)
            if self.use_distance_head:
                self._decoder.distance_head.train(mode)
        return self

    def _forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                feats, _ = self._backbone.get_encoder_features_list(x)
        else:
            feats, _ = self._backbone.get_encoder_features_list(x)

        decoder_out = self._decoder(feats, output_size=x.shape[-3:])

        if self.use_dual_head and self.use_distance_head:
            feat, dual_seg_logits, distance_logits = decoder_out
        elif self.use_dual_head:
            feat, dual_seg_logits = decoder_out
            distance_logits = None
        elif self.use_distance_head:
            feat, distance_logits = decoder_out
            dual_seg_logits = None
        else:
            feat = decoder_out
            dual_seg_logits = None
            distance_logits = None

        return feat, dual_seg_logits, distance_logits

    def _compute_loss(self, batch, is_train: bool):
        image = batch['image'].to(self.device)
        mask = batch['mask'].to(self.device)[:, 0]

        feat, dual_seg_logits, distance_logits = self._forward(image)
        B, C, D, H, W = feat.shape
        flat_feat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
        flat_mask = mask.reshape(-1)
        if flat_mask.min() < 0 or flat_mask.max() > self.n_classes:
            raise ValueError(f'Mask labels must be in [0, {self.n_classes}], '
                             f'got [{flat_mask.min().item()}, '
                             f'{flat_mask.max().item()}]')

        self.proto_bank.normalize_()
        all_protos = self.proto_bank.prototypes.clone()

        sims_full = torch.einsum('mc,nkc->mnk', flat_feat, all_protos)
        class_scores = sims_full.max(dim=2).values
        proto_logits_flat = class_scores / self.temperature
        proto_logits = proto_logits_flat.reshape(B, D, H, W, self.n_classes + 1).permute(0, 4, 1, 2, 3)

        aux_proto_loss = None
        if self.use_dual_head:
            seg_logits = dual_seg_logits
            aux_proto_loss = (self.semantic_loss(proto_logits, mask)
                              + self.seg_loss(proto_logits, mask.unsqueeze(1)))
        else:
            seg_logits = proto_logits

        semantic_loss = self.semantic_loss(seg_logits, mask)
        seg_loss = self.seg_loss(seg_logits, mask.unsqueeze(1))

        dist_gt = None
        if self.need_distance_maps:
            dist_gt = batch['distances'].to(self.device)
            if dist_gt.shape[1] != self.n_classes:
                raise ValueError(f'Expected {self.n_classes} foreground distance maps, '
                                 f'got {dist_gt.shape[1]}')

        boundary_loss = None
        if self.use_boundary_loss:
            # Distance maps contain foreground classes only
            foreground_probs = torch.softmax(seg_logits, dim=1)[:, 1:]
            boundary_loss = self.boundary_loss(foreground_probs, dist_gt)

        distance_loss = None
        if self.use_distance_head:
            if distance_logits.shape != dist_gt.shape:
                raise ValueError(
                    f'Distance-head output shape {tuple(distance_logits.shape)} '
                    f'does not match target shape {tuple(dist_gt.shape)}'
                )
            distance_prediction = torch.tanh(distance_logits)
            distance_loss = self.distance_loss(
                distance_prediction,
                dist_gt,
            )

        with torch.no_grad():
            # bank bookkeeping always keys off the prototype path's own
            # prediction -- unchanged whether use_dual_head is on or off.
            pred = proto_logits_flat.argmax(dim=1)

        correct = (pred == flat_mask)

        # ---------------------------------------------------------
        # PPC / PPD
        # ---------------------------------------------------------
        raw_proto_logits = sims_full.reshape(flat_feat.shape[0], -1)

        ppc_logits = raw_proto_logits

        ppc_losses = []
        ppd_losses = []

        proto_sums = torch.zeros_like(self.proto_bank.prototypes)
        proto_counts = torch.zeros(self.n_classes + 1, self.k_per_class, device=flat_feat.device, dtype=flat_feat.dtype)

        for cls in range(self.n_classes + 1):
            cls_sel = (flat_mask == cls)
            if cls_sel.sum() == 0:
                continue
            init_q = sims_full[cls_sel, cls, :]
            q, indexs = sinkhorn(init_q.detach(), sinkhorn_iterations=self.sinkhorn_iterations,
                                 epsilon=self.sinkhorn_epsilon)
            targets = indexs + self.k_per_class * cls

            if self.use_ppc_loss:
                ppc_losses.append(F.cross_entropy(ppc_logits[cls_sel], targets))

            if self.use_ppd_loss:
                gathered = raw_proto_logits[cls_sel].gather(1, targets.unsqueeze(1)).squeeze(1)
                ppd_losses.append((1 - gathered).pow(2).mean())

            correct_k = correct[cls_sel].float()
            m_q = q * correct_k.unsqueeze(1)
            c_q = flat_feat[cls_sel].detach() * correct_k.unsqueeze(1)

            if is_train:
                proto_sums[cls] = m_q.t() @ c_q
                proto_counts[cls] = m_q.sum(dim=0)

        if is_train:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(proto_sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(proto_counts, op=dist.ReduceOp.SUM)
            with torch.no_grad():
                valid_proto = proto_counts > 0
                proto_means = F.normalize(proto_sums / proto_counts.clamp_min(1e-12).unsqueeze(-1), dim=-1)
                updated = F.normalize(
                    self.proto_momentum * self.proto_bank.prototypes + (1 - self.proto_momentum) * proto_means, dim=-1)

                self.proto_bank.prototypes.copy_(
                    torch.where(valid_proto.unsqueeze(-1), updated, self.proto_bank.prototypes))

                if self.use_proto_separation:
                    self.proto_bank.separate_(
                        margin=self.proto_div_margin,
                        strength=self.proto_div_strength)

        ppc_loss = torch.stack(ppc_losses).mean() if ppc_losses else None
        ppd_loss = torch.stack(ppd_losses).mean() if ppd_losses else None
        total = semantic_loss + seg_loss

        if ppc_loss is not None:
            total = total + self.ppc_weight * ppc_loss
        if ppd_loss is not None:
            total = total + self.ppd_weight * ppd_loss
        if boundary_loss is not None:
            total = total + self.boundary_weight * boundary_loss
        if distance_loss is not None:
            total = total + self.distance_weight * distance_loss
        if aux_proto_loss is not None:
            total = total + self.aux_proto_seg_weight * aux_proto_loss

        return (total, semantic_loss, seg_loss, ppc_loss, ppd_loss,
                boundary_loss, distance_loss, seg_logits, mask,
                aux_proto_loss)

    def training_step(self, batch, batch_idx):
        total, semantic_loss, seg_loss, ppc_loss, ppd_loss, boundary_loss, distance_loss, _, _, aux_proto_loss = \
            self._compute_loss(batch, is_train=True)
        self.log('train_loss', total, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        if seg_loss is not None:
            self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if semantic_loss is not None:
            self.log('train_semantic_loss', semantic_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppc_loss is not None:
            self.log('train_ppc_loss', ppc_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppd_loss is not None:
            self.log('train_ppd_loss', ppd_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if boundary_loss is not None:
            self.log('train_boundary_loss', boundary_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_loss is not None:
            self.log('train_distance_loss', distance_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if aux_proto_loss is not None:
            self.log('train_aux_proto_loss', aux_proto_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        return total

    def on_validation_epoch_start(self):
        set_determinism(seed=42)
        self._val_intersection = {cls: 0.0 for cls in range(1, self.n_classes + 1)}
        self._val_pred_sum = {cls: 0.0 for cls in range(1, self.n_classes + 1)}
        self._val_target_sum = {cls: 0.0 for cls in range(1, self.n_classes + 1)}

    def validation_step(self, batch, batch_idx):
        total, semantic_loss, seg_loss, ppc_loss, ppd_loss, boundary_loss, distance_loss, seg_logits, mask, aux_proto_loss = \
            self._compute_loss(batch, is_train=False)
        self.log('val_loss', total, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        if seg_loss is not None:
            self.log('val_seg_loss', seg_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if semantic_loss is not None:
            self.log('val_semantic_loss', semantic_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppc_loss is not None:
            self.log('val_ppc_loss', ppc_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppd_loss is not None:
            self.log('val_ppd_loss', ppd_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if boundary_loss is not None:
            self.log('val_boundary_loss', boundary_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_loss is not None:
            self.log('val_distance_loss', distance_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if aux_proto_loss is not None:
            self.log('val_aux_proto_loss', aux_proto_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)

        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            for cls in range(1, self.n_classes + 1):
                target_bin = (mask == cls)
                pred_bin = (pred == cls)
                self._val_intersection[cls] += (pred_bin & target_bin).sum().item()
                self._val_pred_sum[cls] += pred_bin.sum().item()
                self._val_target_sum[cls] += target_bin.sum().item()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        stats = torch.tensor([[self._val_intersection[cls],
                               self._val_pred_sum[cls],
                               self._val_target_sum[cls]]
                              for cls in range(1, self.n_classes + 1)], device=self.device, dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        dice_vals = []

        for cls in range(1, self.n_classes + 1):
            intersection, pred_sum, target_sum = stats[cls - 1].tolist()
            denom = pred_sum + target_sum
            dice = 2.0 * intersection / denom if denom > 0 else float('nan')
            if not np.isnan(dice):
                dice_vals.append(dice)

            if self.trainer.is_global_zero:
                print(f'  {self.class_names[cls - 1]}: Dice={dice:.4f}')

        if dice_vals:
            mean_dice = float(np.mean(dice_vals))
            self.log('val_dice_mean', mean_dice, sync_dist=False)
