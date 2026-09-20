import os
import copy
import torch
import pickle
import yaml
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
from lightning.pytorch.callbacks import ModelCheckpoint

from cryosiam.data import MrcReader
from cryosiam.losses import BoundaryLoss
from cryosiam.networks.nets import DenseSimSiam, PrototypeSimilarityFPN
from cryosiam.transforms import (
    ClipIntensityd,
    RandomLowPassBlurd,
    RandomGaussianNoised,
    RandomHighPassSharpend,
)
from cryosiam.utils import patch_train_val_split, parser_helper


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
    q = F.one_hot(indexs, num_classes=K).to(dtype=L.dtype)
    return q, indexs


@torch.no_grad()
def spherical_kmeans(features, k, iterations=20, seed=0):
    """Small deterministic spherical k-means used to bootstrap new classes."""
    features = F.normalize(features.float(), dim=1)
    if features.shape[0] == 0:
        raise ValueError('Cannot initialize prototypes from zero feature vectors.')

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    if features.shape[0] >= k:
        indices = torch.randperm(features.shape[0], generator=generator)[:k]
    else:
        indices = torch.randint(0, features.shape[0], (k,), generator=generator)
    centers = features[indices.to(features.device)].clone()

    for _ in range(int(iterations)):
        assignment = (features @ centers.t()).argmax(dim=1)
        new_centers = []
        for cluster in range(k):
            selected = features[assignment == cluster]
            if selected.numel() == 0:
                new_centers.append(centers[cluster])
            else:
                new_centers.append(F.normalize(selected.mean(dim=0), dim=0))
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return F.normalize(centers, dim=1)


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
    def separate_(self, margin=0.98, strength=0.001, class_indices=None):
        """Gently repel near-identical sub-prototypes within selected classes."""
        protos = F.normalize(self.prototypes, dim=-1)
        if class_indices is None:
            class_indices = list(range(protos.shape[0]))
        if not class_indices:
            return

        selected = protos[class_indices]
        pairwise = torch.einsum('nkc,njc->nkj', selected, selected)
        diagonal = torch.eye(
            self.k, dtype=torch.bool, device=protos.device).unsqueeze(0)
        excess = F.relu(pairwise - margin).masked_fill(diagonal, 0.0)

        differences = selected.unsqueeze(2) - selected.unsqueeze(1)
        directions = F.normalize(differences, dim=-1)
        repulsion = (excess.unsqueeze(-1) * directions).sum(dim=2)
        updated = F.normalize(selected + strength * repulsion, dim=-1)
        self.prototypes[class_indices] = updated

    @torch.no_grad()
    def update(self, cls, k, feat):
        feat = F.normalize(feat.detach(), dim=0)
        new = self.momentum * self.prototypes[cls, k] + (1 - self.momentum) * feat
        self.prototypes[cls, k] = F.normalize(new, dim=0)


class PrototypeRefinementModule(pl.LightningModule):

    def __init__(self, config, dense_backbone_config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.dense_backbone_config = dense_backbone_config

        dense_net_cfg = dense_backbone_config['parameters']['network']
        decoder_cfg = self.config['parameters']['network']

        # ---------------------------------------------------------
        # Source checkpoint and active/model class sets
        # ---------------------------------------------------------
        if self.config.get('fine_tune_model'):
            self.initialization_mode = 'refinement'
            self.source_checkpoint = self.config['fine_tune_model']
        elif self.config.get('general_model') or self.config.get('pretrained_model'):
            self.initialization_mode = 'general'
            self.source_checkpoint = self.config.get(
                'general_model', self.config.get('pretrained_model'))
        elif self.config.get('pretrained_dense_simsiam_model'):
            self.initialization_mode = 'dense_backbone'
            self.source_checkpoint = self.config['pretrained_dense_simsiam_model']
        else:
            raise ValueError(
                'Provide fine_tune_model for an existing refinement checkpoint, '
                'general_model/pretrained_model for a generic checkpoint, or '
                'pretrained_dense_simsiam_model for backbone-only initialization.')

        if not os.path.isfile(self.source_checkpoint):
            raise FileNotFoundError(
                f'Initialization checkpoint not found: {self.source_checkpoint}')

        if 'class_names' not in self.config or not self.config['class_names']:
            raise ValueError(
                'class_names must list the classes that are actively fine-tuned.')

        self.active_class_names = list(self.config['class_names'])
        self.n_active_classes = len(self.active_class_names)

        source_checkpoint = torch.load(
            self.source_checkpoint, map_location='cpu', weights_only=False)
        source_state = source_checkpoint['state_dict']
        source_cfg = source_checkpoint.get('hyper_parameters', {}).get('config', {})

        if self.initialization_mode == 'refinement':
            source_class_names = (
                    source_cfg.get('all_class_names') or
                    source_cfg.get('class_names') or
                    self.config.get('source_class_names'))
            if not source_class_names:
                raise ValueError(
                    'Could not determine the class names in fine_tune_model. '
                    'Add source_class_names to the fine-tuning config.')
            self.model_class_names = list(source_class_names)
            missing = [name for name in self.active_class_names
                       if name not in self.model_class_names]
            if missing:
                raise ValueError(
                    f'Fine-tuning classes {missing} are not present in the '
                    f'refinement checkpoint classes {self.model_class_names}.')
        else:
            # Generic/backbone checkpoints do not define the task classes.
            self.model_class_names = list(self.active_class_names)

        self.class_names = self.model_class_names
        self.n_classes = len(self.model_class_names)
        self.active_model_indices = [
            self.model_class_names.index(name) + 1
            for name in self.active_class_names
        ]
        self.active_seg_channels = [0] + self.active_model_indices
        self.active_distance_channels = [i - 1 for i in self.active_model_indices]

        # Keep the saved checkpoint predictor-compatible.  For a refinement
        # source this preserves the original N-class channel layout even when
        # only M classes are actively fine-tuned.
        self.config['class_names'] = list(self.model_class_names)
        self.config['all_class_names'] = list(self.model_class_names)
        self.config['fine_tune_class_names'] = list(self.active_class_names)

        source_bank = source_state.get('proto_bank.prototypes')
        requested_k = self.config.get('k_per_class')
        if self.initialization_mode == 'refinement' and source_bank is not None:
            source_k = int(source_bank.shape[1])
            if requested_k is not None and int(requested_k) != source_k:
                print(f'  WARNING: k_per_class={requested_k} but the refinement '
                      f'checkpoint uses K={source_k}; using K={source_k}.')
            self.k_per_class = source_k
        else:
            self.k_per_class = int(requested_k if requested_k is not None else 5)
        self.config['k_per_class'] = self.k_per_class

        self.temperature = float(decoder_cfg.get('temperature', 0.1))
        self.sinkhorn_iterations = int(decoder_cfg.get('sinkhorn_iterations', 3))
        self.sinkhorn_epsilon = float(decoder_cfg.get('sinkhorn_epsilon', 0.05))
        self.partial_labels = bool(self.config.get('partial_labels', False))

        # One user-facing switch controls all distance functionality.
        self.use_distances = bool(self.config.get('use_distances', False))
        self.use_distance_head = self.use_distances
        self.use_boundary_loss = bool(self.config.get('use_boundary_loss', False))
        if self.partial_labels and self.use_boundary_loss:
            print('WARNING: use_boundary_loss=True is not safe with '
                  'partial_labels=True -- disabling boundary loss.')
            self.use_boundary_loss = False
        self.need_distance_maps = self.use_distances or self.use_boundary_loss
        self.distance_weight = float(self.config.get('distance_weight', 0.1))
        self.boundary_weight = float(self.config.get('boundary_weight', 0.01))
        source_decoder_cfg = (source_cfg.get('parameters', {})
                              .get('network', {}))
        default_distance_clip = 8.0 if self.initialization_mode == 'general' else 5.0
        self.distance_clip = float(decoder_cfg.get(
            'distance_clip',
            source_decoder_cfg.get('distance_clip', default_distance_clip)))

        # If omitted, preserve the refinement checkpoint's segmentation-head
        # choice.  Generic models do not have this refinement head by default.
        if 'use_dual_head' in decoder_cfg:
            self.use_dual_head = bool(decoder_cfg['use_dual_head'])
        else:
            source_has_seg_head = any(
                key.startswith('_decoder.seg_head.') for key in source_state)
            self.use_dual_head = (
                    self.initialization_mode == 'refinement' and source_has_seg_head)

        self._backbone = DenseSimSiam(
            block_type=dense_net_cfg['block_type'],
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

        embed_dim = self._read_embed_dim(self.config)
        self._decoder = PrototypeSimilarityFPN(
            feat_channels=dense_net_cfg['num_filters'],
            out_channels=embed_dim,
            use_context_head=decoder_cfg.get('use_context_head', False),
            predict_at_c3=decoder_cfg.get('predict_at_c3', False),
            use_dual_head=self.use_dual_head,
            use_distance_head=self.use_distance_head,
            num_classes=self.n_classes + 1 if self.use_dual_head else None,
            distance_channels=self.n_classes if self.use_distance_head else None,
            seg_head_hidden=decoder_cfg.get('seg_head_hidden'))
        self.embed_dim = embed_dim

        # A generic PrototypeMatching checkpoint may contain the older
        # conditional distance head (feature + class similarity -> one SDF).
        # It cannot be inserted directly into PrototypeSimilarityFPN without
        # changing the refinement predictor, so it is optionally retained as
        # a frozen teacher while the predictor-compatible distance head learns.
        self.general_distance_distill_weight = float(
            self.config.get('general_distance_distill_weight',
                            0.1 if (self.initialization_mode == 'general' and
                                    self.use_distances and
                                    any(k.startswith('distance_head.')
                                        for k in source_state)) else 0.0))
        self._general_distance_teacher = None

        self._load_pretrained(self.config)

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

        print(f'Initialization mode: {self.initialization_mode}')
        print(f'Model classes ({self.n_classes}): {self.model_class_names}')
        print(f'Fine-tuning classes ({self.n_active_classes}): '
              f'{self.active_class_names}')
        print(f'Encoder: {"frozen" if self.freeze_encoder else "trainable"}')
        print(f'FPN:     {"frozen" if self.freeze_fpn else "trainable"}')
        print(f'K prototypes per class: {self.k_per_class}')
        print(f'Annotation mode: '
              f'{"PU/PARTIAL (0 is unlabeled)" if self.partial_labels else "FULL (0 is background)"}')
        print(f'Distances: {"ENABLED" if self.use_distances else "disabled"}'
              + (f' (weight={self.distance_weight}, clip={self.distance_clip:g})'
                 if self.use_distances else ''))
        print(f'Boundary loss: {"ENABLED" if self.use_boundary_loss else "disabled"}'
              + (f' (weight={self.boundary_weight})' if self.use_boundary_loss else ''))

        self.semantic_loss = nn.CrossEntropyLoss()
        self.seg_loss = GeneralizedDiceLoss(
            include_background=False, to_onehot_y=True,
            sigmoid=False, softmax=True, reduction='mean')
        if self.use_boundary_loss:
            self.boundary_loss = BoundaryLoss()
        if self.use_distances:
            self.distance_loss = nn.SmoothL1Loss(reduction='none')

        self.proto_momentum = float(self.config.get('proto_momentum', 0.99))
        if not 0 <= self.proto_momentum <= 1:
            raise ValueError('proto_momentum must be in [0, 1].')
        self.proto_bank = MultiPrototypeBank(
            n_classes=self.n_classes,
            k_per_class=self.k_per_class,
            feat_dim=embed_dim,
            momentum=self.proto_momentum)
        self._load_prototype_bank(self.config)
        print(f'Prototype EMA momentum: {self.proto_momentum}')

        self.proto_init_max_voxels = int(
            self.config.get('proto_init_max_voxels', 20000))
        self.proto_init_voxels_per_batch = int(
            self.config.get('proto_init_voxels_per_batch', 2048))
        self.proto_init_kmeans_iters = int(
            self.config.get('proto_init_kmeans_iters', 20))
        self.proto_init_batch_size = int(
            self.config.get('proto_init_batch_size', 1))
        self.proto_init_seed = int(self.config.get('proto_init_seed', 0))
        self.bootstrap_background_from_unlabeled = bool(
            self.config.get('bootstrap_background_from_unlabeled', False))

        # ---------------------------------------------------------
        # PU loss: priors are specified only for the M active classes.
        # ---------------------------------------------------------
        self.use_pu_loss = bool(
            self.config.get('use_pu_loss', self.partial_labels))
        self.pu_weight = float(self.config.get('pu_weight', 0.1))
        self.register_buffer('_pu_class_priors', None)
        if self.partial_labels and self.use_pu_loss:
            priors_cfg = self.config.get('pu_class_priors')
            if priors_cfg is None:
                raise ValueError(
                    'partial_labels=True with use_pu_loss=True requires '
                    'pu_class_priors for the actively fine-tuned classes.')
            if isinstance(priors_cfg, dict):
                missing = [name for name in self.active_class_names
                           if name not in priors_cfg]
                if missing:
                    raise ValueError(
                        f'pu_class_priors is missing classes: {missing}')
                priors = [float(priors_cfg[name])
                          for name in self.active_class_names]
            else:
                priors = [float(x) for x in priors_cfg]
                if len(priors) != self.n_active_classes:
                    raise ValueError(
                        f'pu_class_priors must contain '
                        f'{self.n_active_classes} values, got {len(priors)}.')
            if any(p <= 0.0 or p >= 1.0 for p in priors):
                raise ValueError(
                    'Every pu_class_priors value must be strictly between 0 and 1.')
            self._pu_class_priors = torch.tensor(
                priors, dtype=torch.float32)
            print(f'nnPU loss: ENABLED (weight={self.pu_weight}, '
                  f'priors={priors})')
        else:
            print('nnPU loss: disabled')

        # ---------------------------------------------------------
        # EMA teacher for PU fine-tuning
        # ---------------------------------------------------------
        self.teacher_weight = float(self.config.get('teacher_weight', 0.05))
        self.teacher_temperature = float(
            self.config.get('teacher_temperature', 1.0))
        self.teacher_confidence_threshold = float(
            self.config.get('teacher_confidence_threshold', 0.90))
        self.teacher_ignore_radius = int(
            self.config.get('teacher_ignore_radius', 2))
        self.teacher_momentum = float(
            self.config.get('teacher_momentum', 0.99))
        if self.teacher_temperature <= 0:
            raise ValueError('teacher_temperature must be greater than 0.')
        if not 0 <= self.teacher_confidence_threshold <= 1:
            raise ValueError(
                'teacher_confidence_threshold must be in [0, 1].')
        if self.teacher_ignore_radius < 0:
            raise ValueError('teacher_ignore_radius must be non-negative.')
        if not 0 <= self.teacher_momentum <= 1:
            raise ValueError('teacher_momentum must be in [0, 1].')

        if self.partial_labels:
            self._teacher_backbone = copy.deepcopy(self._backbone)
            self._teacher_decoder = copy.deepcopy(self._decoder)
            self.register_buffer(
                '_teacher_prototypes',
                self.proto_bank.prototypes.detach().clone())
            for p in self._teacher_backbone.parameters():
                p.requires_grad = False
            for p in self._teacher_decoder.parameters():
                p.requires_grad = False
            self._teacher_backbone.eval()
            self._teacher_decoder.eval()
            print(f'Teacher consistency: ENABLED (weight={self.teacher_weight}, '
                  f'temperature={self.teacher_temperature}, '
                  f'confidence>={self.teacher_confidence_threshold}, '
                  f'ignore_radius={self.teacher_ignore_radius}, '
                  f'EMA momentum={self.teacher_momentum})')
        else:
            self._teacher_backbone = None
            self._teacher_decoder = None
            self.register_buffer('_teacher_prototypes', None)
            print('Teacher consistency: disabled (full-label mode)')

        self.use_ppc_loss = bool(self.config.get('use_ppc_loss', True))
        self.ppc_weight = float(self.config.get('ppc_weight', 0.01))
        self.use_ppd_loss = bool(self.config.get('use_ppd_loss', True))
        self.ppd_weight = float(self.config.get('ppd_weight', 0.001))
        self.use_proto_separation = bool(
            self.config.get('use_proto_separation', False))
        self.proto_div_margin = float(
            self.config.get('proto_div_margin', 0.95))
        self.proto_div_strength = float(
            self.config.get('proto_div_strength', 0.001))
        print(f'PPC loss: {"ENABLED" if self.use_ppc_loss else "disabled"} '
              f'(weight={self.ppc_weight})')
        print(f'PPD loss: {"ENABLED" if self.use_ppd_loss else "disabled"} '
              f'(weight={self.ppd_weight})')
        print(f'Prototype separation: '
              f'{"ENABLED" if self.use_proto_separation else "disabled"} '
              f'(margin={self.proto_div_margin}, '
              f'strength={self.proto_div_strength})')

        self.aux_proto_seg_weight = float(
            self.config.get('aux_proto_seg_weight', 0.1))
        print(f'Dual-head seg: '
              f'{"ENABLED" if self.use_dual_head else "disabled"}'
              + (f' (aux_proto_seg_weight={self.aux_proto_seg_weight})'
                 if self.use_dual_head else ''))

        self.batch_size = self.config['hyper_parameters']['batch_size']
        self.lr = float(self.config['hyper_parameters']['lr'])
        self.encoder_lr = float(
            self.config['hyper_parameters'].get('encoder_lr', self.lr * 0.1))
        self.optimizer_name = self.config['hyper_parameters']['optimizer']
        self.weight_decay = self.config['hyper_parameters']['weight_decay']
        self.max_epochs = self.config['hyper_parameters']['max_epochs']
        self.sync_dist = int(self.config['parameters']['gpu_devices']) > 1
        print(f'Learning rates: FPN/heads={self.lr:g}, '
              f'encoder={self.encoder_lr:g}')

        # Save the normalized config (full model class list + active subset),
        # which is what the refinement prediction utilities read later.
        self.save_hyperparameters({
            'config': self.config,
            'dense_backbone_config': self.dense_backbone_config,
        })

    @staticmethod
    def _load_compatible(model, prefix, state, skip_prefixes=()):
        target = model.state_dict()
        weights = {}
        skipped_shape = []
        for key, value in state.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if any(name.startswith(p) for p in skip_prefixes):
                continue
            if name not in target:
                continue
            if tuple(target[name].shape) != tuple(value.shape):
                skipped_shape.append((name, tuple(value.shape), tuple(target[name].shape)))
                continue
            weights[name] = value

        miss, unexp = model.load_state_dict(weights, strict=False)
        return len(weights), miss, unexp, skipped_shape

    def _load_pretrained(self, config):
        checkpoint_path = self.source_checkpoint
        print(f'Loading source checkpoint: {os.path.basename(checkpoint_path)}')
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint['state_dict']

        if self.initialization_mode == 'dense_backbone':
            dense_state = {
                k.replace('_model.', ''): v
                for k, v in state.items() if k.startswith('_model.')
            }
            if dense_state:
                self._backbone.load_state_dict(dense_state, strict=False)
            else:
                loaded = False
                for enc_prefix in ('_encoder.', '_search_model.', '_backbone.'):
                    n, _, _, _ = self._load_compatible(
                        self._backbone, enc_prefix, state)
                    if n:
                        loaded = True
                        break
                if not loaded:
                    raise RuntimeError(
                        'Could not find DenseSimSiam backbone weights in checkpoint.')
            print('  Backbone loaded; FPN/prototypes/task heads start fresh.')
            return

        loaded_backbone = False
        for enc_prefix in ('_encoder.', '_search_model.', '_backbone.'):
            n, _, _, _ = self._load_compatible(
                self._backbone, enc_prefix, state)
            if n:
                loaded_backbone = True
                break
        if not loaded_backbone:
            raise RuntimeError(
                'Could not find backbone weights in source checkpoint.')

        if self.initialization_mode == 'refinement':
            # model_class_names is the original N-class set, so the output
            # channel layout remains identical even if only M classes are
            # fine-tuned. Load every shape-compatible refinement tensor.
            n, _, _, skipped = self._load_compatible(
                self._decoder, '_decoder.', state)
            # Shape mismatches in task heads normally mean the checkpoint
            # metadata and class list disagree, which would make downstream
            # channel names unsafe.
            if skipped:
                raise ValueError(
                    'Refinement checkpoint has decoder tensors incompatible '
                    f'with its declared class layout: {skipped[:5]}')
            print(f'  Backbone + refinement decoder loaded '
                  f'({n} decoder tensors) ✓')
            if self.use_distances:
                has_source_distance = any(
                    key.startswith('_decoder.distance_head.') for key in state)
                if has_source_distance:
                    print('  Refinement distance head loaded ✓')
                else:
                    print('  Source refinement checkpoint has no distance head; '
                          'predictor-compatible distance head starts fresh.')
            return

        # Generic PrototypeMatching checkpoints provide the backbone and
        # shared FPN, but their task-specific heads use a different interface.
        n, _, _, skipped = self._load_compatible(
            self._decoder, '_decoder.', state,
            skip_prefixes=('seg_head.', 'distance_head.'))
        print(f'  Backbone + shared FPN loaded ({n} decoder tensors) ✓')
        if skipped:
            print(f'  Ignored {len(skipped)} incompatible generic decoder tensors.')
        if self.use_dual_head:
            print('  Predictor-compatible segmentation head starts fresh.')
        if self.use_distances:
            print('  Predictor-compatible multi-class distance head starts fresh.')

        if (self.use_distances and
                self.general_distance_distill_weight > 0 and
                any(key.startswith('distance_head.') for key in state)):
            self._load_general_distance_teacher(state)
            print('  Loaded generic conditional distance head as a frozen '
                  f'teacher (distill weight={self.general_distance_distill_weight}).')
        elif self.use_distances and self.general_distance_distill_weight > 0:
            print('  WARNING: general_distance_distill_weight > 0 but the '
                  'generic checkpoint has no distance_head.* weights; '
                  'distance distillation is disabled.')
            self.general_distance_distill_weight = 0.0

    def _load_general_distance_teacher(self, state):
        """Load the generic conditional distance head as a frozen teacher.

        The generic head consumes [embedding, class-similarity] and emits one
        signed-distance map.  The refinement predictor expects per-class
        channels from PrototypeSimilarityFPN, so the generic head is used only
        for distillation; the saved/output head remains predictor-compatible.
        """
        required = [
            'distance_head.0.weight', 'distance_head.0.bias',
            'distance_head.2.weight', 'distance_head.2.bias',
            'distance_head.4.weight', 'distance_head.4.bias',
        ]
        missing = [key for key in required if key not in state]
        if missing:
            raise RuntimeError(
                f'Generic distance head is incomplete; missing {missing}.')

        w0 = state['distance_head.0.weight']
        w2 = state['distance_head.2.weight']
        w4 = state['distance_head.4.weight']
        if (w0.ndim != 5 or w2.ndim != 5 or w4.ndim != 5 or
                w0.shape[1] != self.embed_dim + 1 or w4.shape[0] != 1):
            raise ValueError(
                'Generic distance head has an unexpected architecture: '
                f'{tuple(w0.shape)}, {tuple(w2.shape)}, {tuple(w4.shape)}')

        def conv_from_weight(weight, bias):
            kernel = tuple(int(x) for x in weight.shape[-3:])
            padding = tuple(k // 2 for k in kernel)
            layer = nn.Conv3d(
                int(weight.shape[1]), int(weight.shape[0]),
                kernel_size=kernel, padding=padding,
                bias=bias is not None)
            return layer

        teacher = nn.Sequential(
            conv_from_weight(w0, state.get('distance_head.0.bias')),
            nn.ReLU(inplace=False),
            conv_from_weight(w2, state.get('distance_head.2.bias')),
            nn.ReLU(inplace=False),
            conv_from_weight(w4, state.get('distance_head.4.bias')),
        )
        teacher_state = {
            key.replace('distance_head.', ''): value
            for key, value in state.items()
            if key.startswith('distance_head.')
        }
        teacher.load_state_dict(teacher_state, strict=True)
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()
        self._general_distance_teacher = teacher

    def _load_prototype_bank(self, config):
        checkpoint = torch.load(
            self.source_checkpoint, map_location='cpu', weights_only=False)
        state = checkpoint['state_dict']
        key = 'proto_bank.prototypes'

        self.bootstrap_target_prototypes = (
                self.initialization_mode != 'refinement')
        self.background_initialized = False
        self.proto_bank.normalize_()

        if self.initialization_mode == 'refinement':
            if key not in state:
                raise RuntimeError(
                    'The refinement checkpoint has no prototype bank.')
            prototypes = state[key].float()
            expected = self.proto_bank.prototypes.shape
            if prototypes.shape != expected:
                raise ValueError(
                    f'Refinement prototype shape {tuple(prototypes.shape)} '
                    f'does not match expected {tuple(expected)}.')
            with torch.no_grad():
                self.proto_bank.prototypes.copy_(
                    F.normalize(prototypes, dim=-1))
            self.background_initialized = True
            print(f'  Loaded full refinement prototype bank: '
                  f'{tuple(prototypes.shape)} ✓')
            if self.n_active_classes < self.n_classes:
                print('  Only these prototype classes will adapt: '
                      f'{self.active_class_names}')
            return

        # Generic checkpoints do not define the new target classes. Reuse only
        # their learned background prototypes; foreground prototypes are
        # initialized from the fine-tuning annotations at fit start.
        if key in state:
            source = state[key].float()
            if source.ndim == 3 and source.shape[-1] == self.embed_dim:
                source_bg = F.normalize(source[0], dim=-1)
                if source_bg.shape[0] >= self.k_per_class:
                    source_bg = source_bg[:self.k_per_class]
                else:
                    repeat = ((self.k_per_class + source_bg.shape[0] - 1) //
                              source_bg.shape[0])
                    source_bg = source_bg.repeat(repeat, 1)[:self.k_per_class]
                with torch.no_grad():
                    self.proto_bank.prototypes[0].copy_(source_bg)
                self.background_initialized = True
                print(f'  Reused generic background prototypes: '
                      f'{tuple(source_bg.shape)} ✓')

        if not self.background_initialized:
            if self.partial_labels and not bool(
                    config.get('bootstrap_background_from_unlabeled', False)):
                raise RuntimeError(
                    'Generic/backbone initialization has no reusable background '
                    'prototypes. In PU mode label 0 is unknown, so background '
                    'cannot be initialized safely. Use a general_model checkpoint '
                    'containing proto_bank.prototypes, or explicitly set '
                    'bootstrap_background_from_unlabeled: true.')
            print('  Background prototypes will be initialized from label-0 voxels.')

        print('  Target foreground prototypes will be initialized from annotations.')

    def _read_embed_dim(self, config):
        decoder_cfg = config['parameters']['network']
        checkpoint_path = self.source_checkpoint

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ckpt['state_dict']
        if '_decoder.lat5.weight' in state:
            inferred = state['_decoder.lat5.weight'].shape[0]
            config_val = decoder_cfg.get('embed_dim', decoder_cfg.get('out_channels'))
            if config_val is not None and int(config_val) != inferred:
                print(f'  WARNING: config embed_dim/out_channels={config_val} but the '
                      f'source checkpoint FPN weights imply {inferred} -- using '
                      f'the checkpoint-derived value, config value ignored.')
            return inferred

        if 'embed_dim' in decoder_cfg or 'out_channels' in decoder_cfg:
            return decoder_cfg.get('embed_dim', decoder_cfg.get('out_channels', 128))

        print('  WARNING: no "_decoder.lat5.weight" found in source checkpoint -- '
              'falling back to embed_dim=128.')
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
                ClipIntensityd(keys=['distances'], a_min=-self.distance_clip,
                               a_max=self.distance_clip),
                ScaleIntensityRanged(keys=['distances'],
                                     a_min=-self.distance_clip,
                                     a_max=self.distance_clip,
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
        train_val_path = os.path.join(self.config['log_dir'], 'fine_tune_train_val_split.pkl')
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
                    f'{distance_path}')
            sample['distances'] = distance_path

    def setup(self, stage=None):
        train_val_path = os.path.join(self.config['log_dir'], 'fine_tune_train_val_split.pkl')
        with open(train_val_path, 'rb') as f:
            data = pickle.load(f)
        train_files, val_files = data['train_files'], data['val_files']
        self._add_distance_paths(train_files)
        self._add_distance_paths(val_files)

        set_determinism(seed=0)
        self.train_ds = Dataset(data=train_files, transform=self._build_transforms(augment=True))
        self.val_ds = Dataset(data=val_files, transform=self._build_transforms(augment=False))
        self.proto_init_ds = None
        if self.bootstrap_target_prototypes:
            self.proto_init_ds = Dataset(
                data=train_files, transform=self._build_transforms(augment=False))

        print(f'Train: {len(self.train_ds)}  Val: {len(self.val_ds)} patches')

    @torch.no_grad()
    def _initialize_target_prototypes(self):
        if not self.bootstrap_target_prototypes:
            return
        if self.proto_init_ds is None:
            raise RuntimeError('Prototype initialization dataset is not available.')

        collect_classes = list(range(1, self.n_classes + 1))
        if not self.background_initialized:
            collect_classes = [0] + collect_classes

        collected = {cls: [] for cls in collect_classes}
        counts = {cls: 0 for cls in collect_classes}
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.proto_init_seed)

        loader = DataLoader(
            self.proto_init_ds, batch_size=self.proto_init_batch_size,
            collate_fn=list_data_collate, shuffle=False, num_workers=4,
            persistent_workers=False, worker_init_fn=worker_init_fn,
            pin_memory=False, drop_last=False)

        was_backbone_training = self._backbone.training
        was_decoder_training = self._decoder.training
        self._backbone.eval()
        self._decoder.eval()

        for batch in loader:
            if all(counts[c] >= self.proto_init_max_voxels for c in collect_classes):
                break

            image = batch['image'].to(self.device)
            mask = batch['mask'].to(self.device)[:, 0]
            feat, _, _ = self._forward(image)
            flat_feat = F.normalize(
                feat.permute(0, 2, 3, 4, 1).reshape(-1, feat.shape[1]), dim=1)
            flat_mask = mask.reshape(-1)

            for cls in collect_classes:
                remaining = self.proto_init_max_voxels - counts[cls]
                if remaining <= 0:
                    continue
                indices = torch.nonzero(flat_mask == cls, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                take = min(remaining, self.proto_init_voxels_per_batch, indices.numel())
                if indices.numel() > take:
                    order = torch.randperm(indices.numel(), generator=generator)[:take]
                    indices = indices[order.to(indices.device)]
                else:
                    indices = indices[:take]
                collected[cls].append(flat_feat[indices].detach().cpu())
                counts[cls] += int(indices.numel())

        self._backbone.train(was_backbone_training)
        self._decoder.train(was_decoder_training)

        for cls in collect_classes:
            if not collected[cls]:
                label = 'background/unlabeled' if cls == 0 else self.class_names[cls - 1]
                raise RuntimeError(
                    f'No voxels found to initialize prototypes for {label}.')
            features = torch.cat(collected[cls], dim=0).to(self.device)
            centers = spherical_kmeans(
                features, self.k_per_class,
                iterations=self.proto_init_kmeans_iters,
                seed=self.proto_init_seed + cls)
            self.proto_bank.prototypes[cls].copy_(centers)
            label = 'background' if cls == 0 else self.class_names[cls - 1]
            print(f'  Initialized {label}: {len(features)} voxels -> '
                  f'{self.k_per_class} prototypes')

        self.proto_bank.normalize_()
        if not self.background_initialized:
            self.background_initialized = True

    def on_fit_start(self):
        if self.bootstrap_target_prototypes and self.trainer.is_global_zero:
            print('Bootstrapping target prototype bank from fine-tuning annotations...')
            self._initialize_target_prototypes()

        self.proto_bank.normalize_()
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.proto_bank.prototypes, src=0)

        # The EMA teacher must start from the same freshly initialized target
        # prototype bank, rather than the random placeholders created in init.
        if self.partial_labels:
            self._teacher_prototypes.copy_(self.proto_bank.prototypes.detach())

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          collate_fn=list_data_collate, shuffle=True,
                          num_workers=10, persistent_workers=True,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False, drop_last=False)

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
            param_groups.append({'params': distance_head_params, 'lr': self.lr})

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
            param_groups.append({'params': self._backbone.parameters(), 'lr': self.encoder_lr})

        if not param_groups:
            raise RuntimeError('Backbone, FPN, and optional heads are all frozen; '
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
        if self.partial_labels:
            self._teacher_backbone.eval()
            self._teacher_decoder.eval()
        if self._general_distance_teacher is not None:
            self._general_distance_teacher.eval()
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

    @torch.no_grad()
    def _update_teacher_ema(self):
        if not self.partial_labels:
            return

        momentum = self.teacher_momentum
        one_minus_momentum = 1.0 - momentum

        for teacher_param, student_param in zip(
                self._teacher_backbone.parameters(), self._backbone.parameters()):
            teacher_param.mul_(momentum).add_(
                student_param.detach(), alpha=one_minus_momentum)
        for teacher_param, student_param in zip(
                self._teacher_decoder.parameters(), self._decoder.parameters()):
            teacher_param.mul_(momentum).add_(
                student_param.detach(), alpha=one_minus_momentum)

        # Keep non-parameter state (e.g. normalization running statistics) in
        # step with the student as well.
        for teacher_buffer, student_buffer in zip(
                self._teacher_backbone.buffers(), self._backbone.buffers()):
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(momentum).add_(
                    student_buffer.detach(), alpha=one_minus_momentum)
            else:
                teacher_buffer.copy_(student_buffer)
        for teacher_buffer, student_buffer in zip(
                self._teacher_decoder.buffers(), self._decoder.buffers()):
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(momentum).add_(
                    student_buffer.detach(), alpha=one_minus_momentum)
            else:
                teacher_buffer.copy_(student_buffer)

        updated_prototypes = (
                momentum * self._teacher_prototypes +
                one_minus_momentum * self.proto_bank.prototypes.detach())
        self._teacher_prototypes.copy_(
            F.normalize(updated_prototypes, dim=-1))

    def on_before_zero_grad(self, optimizer):
        # Lightning calls this after optimizer.step(), so the EMA teacher sees
        # the newly updated student rather than the pre-step parameters.
        if self.partial_labels:
            self._update_teacher_ema()

    @torch.no_grad()
    def _teacher_forward(self, x):
        feats, _ = self._teacher_backbone.get_encoder_features_list(x)
        decoder_out = self._teacher_decoder(feats, output_size=x.shape[-3:])

        if self.use_dual_head and self.use_distance_head:
            feat, dual_seg_logits, _ = decoder_out
        elif self.use_dual_head:
            feat, dual_seg_logits = decoder_out
        elif self.use_distance_head:
            feat, _ = decoder_out
            dual_seg_logits = None
        else:
            feat = decoder_out
            dual_seg_logits = None

        if self.use_dual_head:
            return dual_seg_logits

        B, C, D, H, W = feat.shape
        flat_feat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
        prototypes = F.normalize(self._teacher_prototypes, dim=-1)
        similarities = torch.einsum('mc,nkc->mnk', flat_feat, prototypes)
        class_scores = similarities.max(dim=2).values / self.temperature

        return class_scores.reshape(
            B, D, H, W, self.n_classes + 1).permute(0, 4, 1, 2, 3)

    def _compute_nnpu_loss(self, logits, mask):
        """One-vs-rest non-negative PU risk for the M active classes.

        ``logits`` contains [background/unknown + active classes] only, and
        ``mask`` uses local labels 0..M. Label 0 is the unlabeled mixture.
        """
        if not (self.partial_labels and self.use_pu_loss):
            return None

        unlabeled = (mask == 0)
        if not unlabeled.any():
            return None

        class_losses = []
        for cls in range(1, self.n_active_classes + 1):
            positive = (mask == cls)
            if not positive.any():
                continue

            cls_logit = logits[:, cls]
            other_logits = torch.cat(
                [logits[:, :cls], logits[:, cls + 1:]], dim=1)
            binary_logit = cls_logit - torch.logsumexp(
                other_logits, dim=1)

            positive_as_positive = F.softplus(
                -binary_logit[positive]).mean()
            positive_as_negative = F.softplus(
                binary_logit[positive]).mean()
            unlabeled_as_negative = F.softplus(
                binary_logit[unlabeled]).mean()

            prior = self._pu_class_priors[cls - 1].to(
                device=logits.device, dtype=logits.dtype)
            positive_risk = prior * positive_as_positive
            negative_risk = (
                    unlabeled_as_negative - prior * positive_as_negative)
            class_losses.append(
                positive_risk + torch.clamp(negative_risk, min=0.0))

        if not class_losses:
            return None
        return torch.stack(class_losses).mean()

    def _compute_loss(self, batch, is_train: bool):
        image = batch['image'].to(self.device)
        # Fine-tuning masks always use LOCAL labels:
        #   0 = unknown/background (depending on mode)
        #   1..M = class_names in the fine-tuning config
        mask = batch['mask'].to(self.device)[:, 0]
        flat_mask = mask.reshape(-1)
        if flat_mask.min() < 0 or flat_mask.max() > self.n_active_classes:
            raise ValueError(
                f'Fine-tuning mask labels must be in [0, '
                f'{self.n_active_classes}], got '
                f'[{flat_mask.min().item()}, {flat_mask.max().item()}].')

        feat, dual_seg_logits, distance_logits = self._forward(image)
        B, C, D, H, W = feat.shape
        flat_feat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)

        self.proto_bank.normalize_()
        all_protos = self.proto_bank.prototypes.clone()
        sims_full = torch.einsum('mc,nkc->mnk', flat_feat, all_protos)
        class_scores_full = sims_full.max(dim=2).values
        proto_logits_flat_full = class_scores_full / self.temperature
        proto_logits_full = proto_logits_flat_full.reshape(
            B, D, H, W, self.n_classes + 1).permute(0, 4, 1, 2, 3)

        if self.use_dual_head:
            seg_logits_full = dual_seg_logits
        else:
            seg_logits_full = proto_logits_full

        # Only background + the M actively fine-tuned classes participate in
        # the loss.  This leaves unselected refinement output channels out of
        # the direct segmentation/prototype objective while preserving their
        # checkpoint layout for the refinement predictor.
        seg_logits = seg_logits_full[:, self.active_seg_channels]
        proto_logits = proto_logits_full[:, self.active_seg_channels]

        semantic_loss = None
        seg_loss = None
        pu_loss = None
        teacher_loss = None
        aux_proto_loss = None

        if self.partial_labels:
            annotated = (mask > 0)
            unknown = (mask == 0)

            if annotated.any():
                semantic_map = F.cross_entropy(
                    seg_logits, mask, reduction='none')
                semantic_loss = semantic_map[annotated].mean()

                if self.use_dual_head:
                    aux_map = F.cross_entropy(
                        proto_logits, mask, reduction='none')
                    aux_proto_loss = aux_map[annotated].mean()

            pu_loss = self._compute_nnpu_loss(seg_logits, mask)

            if self.teacher_weight > 0 and unknown.any():
                if self.teacher_ignore_radius > 0:
                    radius = self.teacher_ignore_radius
                    annotated_halo = F.max_pool3d(
                        annotated.float().unsqueeze(1),
                        kernel_size=2 * radius + 1,
                        stride=1,
                        padding=radius)[:, 0].bool()
                else:
                    annotated_halo = annotated

                teacher_region = unknown & ~annotated_halo
                if teacher_region.any():
                    teacher_logits_full = self._teacher_forward(image)
                    teacher_logits = teacher_logits_full[:, self.active_seg_channels]
                    teacher_temperature = self.teacher_temperature
                    teacher_probs = torch.softmax(
                        teacher_logits / teacher_temperature, dim=1)
                    teacher_confidence, teacher_prediction = \
                        teacher_probs.max(dim=1)
                    teacher_valid = (
                            teacher_region &
                            (teacher_confidence >=
                             self.teacher_confidence_threshold))

                    if teacher_valid.any():
                        teacher_map = F.kl_div(
                            F.log_softmax(
                                seg_logits / teacher_temperature, dim=1),
                            teacher_probs,
                            reduction='none').sum(dim=1)
                        teacher_class_losses = []
                        for cls in range(self.n_active_classes + 1):
                            cls_sel = (
                                    teacher_valid &
                                    (teacher_prediction == cls))
                            if cls_sel.any():
                                teacher_class_losses.append(
                                    teacher_map[cls_sel].mean())
                        if teacher_class_losses:
                            teacher_loss = (
                                    torch.stack(teacher_class_losses).mean() *
                                    teacher_temperature ** 2)
        else:
            # Full-label mode, but still restricted to the M selected classes.
            semantic_loss = self.semantic_loss(seg_logits, mask)
            seg_loss = self.seg_loss(seg_logits, mask.unsqueeze(1))
            if self.use_dual_head:
                aux_proto_loss = (
                        self.semantic_loss(proto_logits, mask) +
                        self.seg_loss(proto_logits, mask.unsqueeze(1)))

        # ---------------------------------------------------------
        # Distance targets / boundary / predictor-compatible head
        # ---------------------------------------------------------
        dist_gt = None
        if self.need_distance_maps:
            dist_gt = batch['distances'].to(self.device)
            if dist_gt.shape[1] != self.n_active_classes:
                raise ValueError(
                    f'Expected {self.n_active_classes} distance target '
                    f'channels for fine-tuning classes '
                    f'{self.active_class_names}, got {dist_gt.shape[1]}.')

        boundary_loss = None
        if self.use_boundary_loss:
            foreground_probs = torch.softmax(seg_logits, dim=1)[:, 1:]
            boundary_loss = self.boundary_loss(
                foreground_probs, dist_gt)

        distance_loss = None
        distance_distill_loss = None
        if self.use_distances:
            if distance_logits is None:
                raise RuntimeError(
                    'use_distances=True but decoder returned no distance logits.')
            if distance_logits.shape[1] != self.n_classes:
                raise ValueError(
                    f'Distance head has {distance_logits.shape[1]} channels, '
                    f'expected {self.n_classes}.')

            active_distance_logits = distance_logits[
                                     :, self.active_distance_channels]
            if active_distance_logits.shape != dist_gt.shape:
                raise ValueError(
                    f'Active distance output shape '
                    f'{tuple(active_distance_logits.shape)} does not match '
                    f'target shape {tuple(dist_gt.shape)}.')

            distance_prediction = torch.tanh(active_distance_logits)
            distance_loss_map = self.distance_loss(
                distance_prediction, dist_gt)

            # Match the generic model's distance supervision exactly:
            # supervise the object interior plus a narrow exterior band, and
            # ignore the far exterior.  This is also safe for PU labels.
            class_distance_losses = []
            for local_cls in range(self.n_active_classes):
                gt = dist_gt[:, local_cls]
                pred_loss = distance_loss_map[:, local_cls]
                inside = gt > 0
                near_outside = (gt < 0) & (gt > -1)
                terms = []
                if inside.any():
                    terms.append(pred_loss[inside].mean())
                if near_outside.any():
                    terms.append(pred_loss[near_outside].mean())
                if terms:
                    class_distance_losses.append(
                        torch.stack(terms).mean())
            if class_distance_losses:
                distance_loss = torch.stack(
                    class_distance_losses).mean()

            # Generic-model only: transfer knowledge from the already-trained
            # conditional distance head into the refinement-compatible
            # multi-channel head.  This does not change downstream prediction.
            if (self._general_distance_teacher is not None and
                    self.general_distance_distill_weight > 0):
                teacher_distances = []
                with torch.no_grad():
                    score_volume = class_scores_full.reshape(
                        B, D, H, W, self.n_classes + 1).permute(0, 4, 1, 2, 3)
                    for model_cls in self.active_model_indices:
                        similarity = score_volume[:, model_cls:model_cls + 1]
                        teacher_input = torch.cat(
                            [feat.detach(), similarity.detach()], dim=1)
                        teacher_distances.append(torch.tanh(
                            self._general_distance_teacher(teacher_input))[:, 0])
                teacher_terms = [
                    F.smooth_l1_loss(
                        distance_prediction[:, local_cls], teacher_distance)
                    for local_cls, teacher_distance in enumerate(
                        teacher_distances)
                ]
                if teacher_terms:
                    distance_distill_loss = torch.stack(
                        teacher_terms).mean()

        # ---------------------------------------------------------
        # PPC / PPD + prototype EMA, restricted to active classes
        # ---------------------------------------------------------
        active_sims = sims_full[:, self.active_seg_channels, :]
        raw_proto_logits = active_sims.reshape(
            flat_feat.shape[0], -1)
        ppc_logits = raw_proto_logits
        ppc_losses = []
        ppd_losses = []

        proto_sums = torch.zeros_like(self.proto_bank.prototypes)
        proto_counts = torch.zeros(
            self.n_classes + 1, self.k_per_class,
            device=flat_feat.device, dtype=flat_feat.dtype)

        local_classes = list(range(1, self.n_active_classes + 1))
        if not self.partial_labels:
            local_classes = [0] + local_classes

        for local_cls in local_classes:
            model_cls = (0 if local_cls == 0 else
                         self.active_model_indices[local_cls - 1])
            cls_sel = (flat_mask == local_cls)
            if cls_sel.sum() == 0:
                continue

            init_q = sims_full[cls_sel, model_cls, :]
            q, indexs = sinkhorn(
                init_q.detach(),
                sinkhorn_iterations=self.sinkhorn_iterations,
                epsilon=self.sinkhorn_epsilon)
            targets = indexs + self.k_per_class * local_cls

            if self.use_ppc_loss:
                ppc_losses.append(F.cross_entropy(
                    ppc_logits[cls_sel], targets))

            if self.use_ppd_loss:
                gathered = raw_proto_logits[cls_sel].gather(
                    1, targets.unsqueeze(1)).squeeze(1)
                ppd_losses.append((1 - gathered).pow(2).mean())

            if is_train:
                proto_sums[model_cls] = q.t() @ \
                                        flat_feat[cls_sel].detach()
                proto_counts[model_cls] = q.sum(dim=0)

        if is_train:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(proto_sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(proto_counts, op=dist.ReduceOp.SUM)
            with torch.no_grad():
                valid_proto = proto_counts > 0
                proto_means = F.normalize(
                    proto_sums /
                    proto_counts.clamp_min(1e-12).unsqueeze(-1), dim=-1)
                updated = F.normalize(
                    self.proto_momentum * self.proto_bank.prototypes +
                    (1 - self.proto_momentum) * proto_means, dim=-1)
                self.proto_bank.prototypes.copy_(
                    torch.where(valid_proto.unsqueeze(-1), updated,
                                self.proto_bank.prototypes))

                if self.use_proto_separation:
                    separation_classes = list(self.active_model_indices)
                    if not self.partial_labels:
                        separation_classes = [0] + separation_classes
                    self.proto_bank.separate_(
                        margin=self.proto_div_margin,
                        strength=self.proto_div_strength,
                        class_indices=separation_classes)

        ppc_loss = (
            torch.stack(ppc_losses).mean() if ppc_losses else None)
        ppd_loss = (
            torch.stack(ppd_losses).mean() if ppd_losses else None)

        total = seg_logits.sum() * 0.0
        if semantic_loss is not None:
            total = total + semantic_loss
        if seg_loss is not None:
            total = total + seg_loss
        if pu_loss is not None:
            total = total + self.pu_weight * pu_loss
        if ppc_loss is not None:
            total = total + self.ppc_weight * ppc_loss
        if ppd_loss is not None:
            total = total + self.ppd_weight * ppd_loss
        if boundary_loss is not None:
            total = total + self.boundary_weight * boundary_loss
        if distance_loss is not None:
            total = total + self.distance_weight * distance_loss
        if distance_distill_loss is not None:
            total = total + (
                    self.general_distance_distill_weight * distance_distill_loss)
        if teacher_loss is not None:
            total = total + self.teacher_weight * teacher_loss
        if aux_proto_loss is not None:
            total = total + self.aux_proto_seg_weight * aux_proto_loss

        return (
            total, semantic_loss, seg_loss, pu_loss, ppc_loss, ppd_loss,
            boundary_loss, distance_loss, distance_distill_loss,
            teacher_loss, seg_logits, mask, aux_proto_loss)

    def training_step(self, batch, batch_idx):
        (total, semantic_loss, seg_loss, pu_loss, ppc_loss, ppd_loss,
         boundary_loss, distance_loss, distance_distill_loss,
         teacher_loss, _, _, aux_proto_loss) = \
            self._compute_loss(batch, is_train=True)

        self.log('train_loss', total, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        if seg_loss is not None:
            self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if semantic_loss is not None:
            self.log('train_semantic_loss', semantic_loss, on_step=False,
                     on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
        if pu_loss is not None:
            self.log('train_pu_loss', pu_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppc_loss is not None:
            self.log('train_ppc_loss', ppc_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppd_loss is not None:
            self.log('train_ppd_loss', ppd_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if boundary_loss is not None:
            self.log('train_boundary_loss', boundary_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_loss is not None:
            self.log('train_distance_loss', distance_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_distill_loss is not None:
            self.log('train_distance_distill_loss', distance_distill_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if teacher_loss is not None:
            self.log('train_teacher_loss', teacher_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if aux_proto_loss is not None:
            self.log('train_aux_proto_loss', aux_proto_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        return total

    def on_validation_epoch_start(self):
        set_determinism(seed=42)
        if self.partial_labels:
            self._val_correct = {
                cls: 0.0 for cls in range(1, self.n_active_classes + 1)}
            self._val_target_sum = {
                cls: 0.0 for cls in range(1, self.n_active_classes + 1)}
        else:
            self._val_intersection = {
                cls: 0.0 for cls in range(1, self.n_active_classes + 1)}
            self._val_pred_sum = {
                cls: 0.0 for cls in range(1, self.n_active_classes + 1)}
            self._val_target_sum = {
                cls: 0.0 for cls in range(1, self.n_active_classes + 1)}

    def validation_step(self, batch, batch_idx):
        (total, semantic_loss, seg_loss, pu_loss, ppc_loss, ppd_loss,
         boundary_loss, distance_loss, distance_distill_loss,
         teacher_loss, seg_logits, mask, aux_proto_loss) = \
            self._compute_loss(batch, is_train=False)

        self.log('val_loss', total, on_step=False, on_epoch=True,
                 batch_size=self.batch_size, sync_dist=self.sync_dist)
        if seg_loss is not None:
            self.log('val_seg_loss', seg_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if semantic_loss is not None:
            self.log('val_semantic_loss', semantic_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if pu_loss is not None:
            self.log('val_pu_loss', pu_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppc_loss is not None:
            self.log('val_ppc_loss', ppc_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if ppd_loss is not None:
            self.log('val_ppd_loss', ppd_loss, on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if boundary_loss is not None:
            self.log('val_boundary_loss', boundary_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_loss is not None:
            self.log('val_distance_loss', distance_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if distance_distill_loss is not None:
            self.log('val_distance_distill_loss', distance_distill_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if teacher_loss is not None:
            self.log('val_teacher_loss', teacher_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)
        if aux_proto_loss is not None:
            self.log('val_aux_proto_loss', aux_proto_loss,
                     on_step=False, on_epoch=True,
                     batch_size=self.batch_size, sync_dist=self.sync_dist)

        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            for cls in range(1, self.n_active_classes + 1):
                target_bin = (mask == cls)
                pred_bin = (pred == cls)
                if self.partial_labels:
                    self._val_correct[cls] += (
                            pred_bin & target_bin).sum().item()
                    self._val_target_sum[cls] += target_bin.sum().item()
                else:
                    self._val_intersection[cls] += (
                            pred_bin & target_bin).sum().item()
                    self._val_pred_sum[cls] += pred_bin.sum().item()
                    self._val_target_sum[cls] += target_bin.sum().item()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if self.partial_labels:
            stats = torch.tensor(
                [[self._val_correct[cls], self._val_target_sum[cls]]
                 for cls in range(1, self.n_active_classes + 1)],
                device=self.device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            recall_vals = []
            for cls in range(1, self.n_active_classes + 1):
                correct, target_sum = stats[cls - 1].tolist()
                recall = (correct / target_sum
                          if target_sum > 0 else float('nan'))
                if not np.isnan(recall):
                    recall_vals.append(recall)
                if self.trainer.is_global_zero:
                    print(f'  {self.active_class_names[cls - 1]}: '
                          f'annotated recall={recall:.4f}')
            if recall_vals:
                self.log('val_annotated_recall_mean',
                         float(np.mean(recall_vals)), sync_dist=False)
            return

        stats = torch.tensor(
            [[self._val_intersection[cls],
              self._val_pred_sum[cls],
              self._val_target_sum[cls]]
             for cls in range(1, self.n_active_classes + 1)],
            device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        dice_vals = []
        for cls in range(1, self.n_active_classes + 1):
            intersection, pred_sum, target_sum = stats[cls - 1].tolist()
            denominator = pred_sum + target_sum
            dice = (2.0 * intersection / denominator
                    if denominator > 0 else float('nan'))
            if not np.isnan(dice):
                dice_vals.append(dice)
            if self.trainer.is_global_zero:
                print(f'  {self.active_class_names[cls - 1]}: '
                      f'Dice={dice:.4f}')

        if dice_vals:
            self.log('val_dice_mean', float(np.mean(dice_vals)),
                     sync_dist=False)


def _read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _source_checkpoint_path(config):
    if config.get('fine_tune_model'):
        return config['fine_tune_model']
    if config.get('general_model'):
        return config['general_model']
    if config.get('pretrained_model'):
        return config['pretrained_model']
    if config.get('pretrained_dense_simsiam_model'):
        return config['pretrained_dense_simsiam_model']
    raise ValueError(
        'Provide fine_tune_model, general_model/pretrained_model, or '
        'pretrained_dense_simsiam_model.')


def _looks_like_dense_backbone_config(config):
    if not isinstance(config, dict):
        return False

    network = config.get('parameters', {}).get('network', {})
    required = {
        'block_type',
        'in_channels',
        'spatial_dims',
        'num_layers',
        'num_filters',
        'no_max_pool',
        'fpn_channels',
        'dim',
        'pred_dim',
        'dense_dim',
        'dense_pred_dim',
    }
    return required.issubset(network)


def _load_dense_backbone_config(config):
    """Recover the DenseSimSiam architecture config used by the source model."""
    configured = config.get('dense_backbone_config')

    if isinstance(configured, dict):
        return configured

    if configured:
        dense_config = _read_yaml(configured)
        if not _looks_like_dense_backbone_config(dense_config):
            raise ValueError(
                f'dense_backbone_config does not contain the expected '
                f'DenseSimSiam network settings: {configured}')
        return dense_config

    source_checkpoint = _source_checkpoint_path(config)
    checkpoint = torch.load(
        source_checkpoint, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})

    dense_config = hparams.get('dense_backbone_config')
    if _looks_like_dense_backbone_config(dense_config):
        return dense_config

    # Older DenseSimSiam checkpoints may store their own architecture directly
    # as hyper_parameters["config"].
    source_config = hparams.get('config')
    if _looks_like_dense_backbone_config(source_config):
        return source_config

    raise RuntimeError(
        'Could not recover dense_backbone_config from the source checkpoint. '
        'Set top-level "dense_backbone_config" in the fine-tuning YAML to the '
        'original DenseSimSiam YAML file.')


def _requested_devices(config):
    parameters = config.get('parameters', {})
    devices = int(parameters.get('gpu_devices', 1))
    nodes = int(parameters.get('nodes', 1))

    if devices < 1:
        raise ValueError('parameters.gpu_devices must be >= 1.')
    if nodes < 1:
        raise ValueError('parameters.nodes must be >= 1.')

    return devices, nodes


def _resume_checkpoint(config, checkpoint_dir):
    if not bool(config.get('continue_training', False)):
        return None

    explicit = config.get('resume_checkpoint')
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(
                f'resume_checkpoint does not exist: {explicit}')
        return explicit

    last_checkpoint = os.path.join(checkpoint_dir, 'last.ckpt')
    if not os.path.isfile(last_checkpoint):
        raise FileNotFoundError(
            'continue_training=True, but no last checkpoint was found at '
            f'{last_checkpoint}. Set resume_checkpoint explicitly or set '
            'continue_training: false.')

    return last_checkpoint


def main(config_file_path):
    """CLI entrypoint for prototype-refinement fine-tuning."""
    config = _read_yaml(config_file_path)

    if not isinstance(config, dict):
        raise ValueError(
            f'Fine-tuning config must be a YAML mapping: {config_file_path}')

    if not config.get('log_dir'):
        raise ValueError('Set log_dir in the fine-tuning config.')

    os.makedirs(config['log_dir'], exist_ok=True)
    checkpoint_dir = os.path.join(config['log_dir'], 'model')
    os.makedirs(checkpoint_dir, exist_ok=True)

    dense_backbone_config = _load_dense_backbone_config(config)
    net = PrototypeRefinementModule(config, dense_backbone_config)

    partial_labels = bool(config.get('partial_labels', False))
    default_monitor = (
        'val_annotated_recall_mean' if partial_labels
        else 'val_dice_mean'
    )
    monitor = str(config.get('checkpoint_monitor', default_monitor))
    monitor_mode = str(config.get('checkpoint_mode', 'max')).lower()
    if monitor_mode not in ('min', 'max'):
        raise ValueError(
            f'checkpoint_mode must be "min" or "max", got {monitor_mode!r}.')

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename='model_best',
                                          monitor=monitor,
                                          mode=monitor_mode,
                                          save_top_k=1,
                                          save_last=True,
                                          auto_insert_metric_name=False)

    devices, nodes = _requested_devices(config)
    world_size = devices * nodes

    if torch.cuda.is_available():
        accelerator = 'gpu'
        trainer_devices = devices
    else:
        if world_size > 1:
            raise RuntimeError(
                f'The config requests {nodes} node(s) x {devices} GPU(s), '
                'but CUDA is not available.')
        accelerator = 'cpu'
        trainer_devices = 1
        print('WARNING: CUDA is not available; fine-tuning will run on CPU.')

    hyper = config.get('hyper_parameters', {})
    max_epochs = int(hyper.get('max_epochs', 1))
    val_interval = int(hyper.get('val_interval', 1))
    if max_epochs < 1:
        raise ValueError('hyper_parameters.max_epochs must be >= 1.')
    if val_interval < 1:
        raise ValueError('hyper_parameters.val_interval must be >= 1.')

    strategy = 'ddp' if world_size > 1 else 'auto'

    resume_checkpoint = _resume_checkpoint(config, checkpoint_dir)

    print('\nPrototype-refinement fine-tuning')
    print(f'  config: {config_file_path}')
    print(f'  log dir: {config["log_dir"]}')
    print(f'  checkpoints: {checkpoint_dir}')
    print(f'  monitor: {monitor} ({monitor_mode})')
    print(f'  accelerator: {accelerator}')
    print(f'  devices per node: {trainer_devices}')
    print(f'  nodes: {nodes}')
    print(f'  strategy: {strategy}')
    print(f'  max epochs: {max_epochs}')
    if resume_checkpoint:
        print(f'  resuming training state from: {resume_checkpoint}')

    trainer = pl.Trainer(accelerator=accelerator,
                         devices=trainer_devices,
                         num_nodes=nodes,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=val_interval,
                         callbacks=[checkpoint_callback],
                         default_root_dir=config['log_dir'])

    trainer.fit(net, ckpt_path=resume_checkpoint)

    if trainer.is_global_zero:
        print('\nTraining finished.')
        print(f'  best checkpoint: {checkpoint_callback.best_model_path}')
        print(f'  last checkpoint: {checkpoint_callback.last_model_path}')


if __name__ == '__main__':
    parser = parser_helper('Prototype refinement fine-tuning')
    args = parser.parse_args()
    main(args.config_file)
