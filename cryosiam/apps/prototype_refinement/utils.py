import numpy as np
import collections

import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, SpatialPadd,
    ScaleIntensityRanged, CenterSpatialCropd, NormalizeIntensityd, EnsureTyped,
)

from cryosiam.data import MrcReader
from cryosiam.transforms import NumpyToTensord
from cryosiam.networks.nets import DenseSimSiam, PrototypeSimilarityFPN


def load_backbone_model(checkpoint_path, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['hyper_parameters']['dense_backbone_config']
    net_cfg = cfg['parameters']['network']

    model = DenseSimSiam(block_type=net_cfg['block_type'],
                         spatial_dims=net_cfg['spatial_dims'],
                         n_input_channels=net_cfg['in_channels'],
                         num_layers=net_cfg['num_layers'],
                         num_filters=net_cfg['num_filters'],
                         fpn_channels=net_cfg['fpn_channels'],
                         no_max_pool=net_cfg['no_max_pool'],
                         dim=net_cfg['dim'],
                         pred_dim=net_cfg['pred_dim'],
                         dense_dim=net_cfg['dense_dim'],
                         dense_pred_dim=net_cfg['dense_pred_dim'],
                         decoder=False)

    state_dict = collections.OrderedDict(
        (k.replace('_backbone.', ''), v)
        for k, v in checkpoint['state_dict'].items()
        if k.startswith('_backbone.'))

    miss, unexp = model.load_state_dict(state_dict, strict=False)
    if miss:
        print(f'  [backbone] missing:    {miss[:3]}...')
    if unexp:
        print(f'  [backbone] unexpected: {unexp[:3]}...')

    model.eval()
    model.to(torch.device(device))
    return model


def load_decoder_model(checkpoint_path, class_names=None, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    decoder_keys = {k.replace('_decoder.', ''): v
                    for k, v in state_dict.items() if k.startswith('_decoder.')}
    if not decoder_keys:
        raise RuntimeError(
            f'No "_decoder.*" keys found in checkpoint state_dict. '
            f'Available top-level prefixes: '
            f'{sorted({k.split(".")[0] for k in state_dict.keys()})}')

    required = ['lat5.weight', 'lat4.weight', 'lat3.weight', 'lat2.weight']
    missing = [k for k in required if k not in decoder_keys]
    if missing:
        raise RuntimeError(f'Cannot infer decoder architecture -- missing keys: {missing}')

    predict_at_c3 = any(k.startswith('bottom_up_p2_to_p3.') for k in decoder_keys)
    use_context_head = any(key.startswith('context_head.') for key in decoder_keys)
    use_dual_head = any(k.startswith('seg_head.') for k in decoder_keys)
    use_distance_head = any(k.startswith('distance_head.') for k in decoder_keys)
    out_channels = decoder_keys['lat5.weight'].shape[0]
    c5 = decoder_keys['lat5.weight'].shape[1]
    c4 = decoder_keys['lat4.weight'].shape[1]
    c3 = decoder_keys['lat3.weight'].shape[1]
    c2 = decoder_keys['lat2.weight'].shape[1]

    seg_head_hidden, num_classes = None, None
    if use_dual_head:
        seg_head_hidden = decoder_keys['seg_head.0.weight'].shape[0]
        num_classes = decoder_keys['seg_head.3.weight'].shape[0]

    distance_channels = None
    if use_distance_head:
        distance_channels = decoder_keys['distance_head.6.weight'].shape[0]

    print(f'  Inferred decoder arch from weights: out_channels={out_channels}  '
          f'feat_channels=({c2},{c3},{c4},{c5})  predict_at_c3={predict_at_c3}  '
          f'use_dual_head={use_dual_head}  '
          f'use_distance_head={use_distance_head}'
          + (f'  seg_head_hidden={seg_head_hidden}  num_classes={num_classes}'
             if use_dual_head else '')
          + (f'  distance_channels={distance_channels}'
             if use_distance_head else ''))

    model = PrototypeSimilarityFPN(feat_channels=(c2, c3, c4, c5),
                                   out_channels=out_channels,
                                   use_context_head=use_context_head,
                                   predict_at_c3=predict_at_c3,
                                   use_dual_head=use_dual_head,
                                   use_distance_head=use_distance_head,
                                   num_classes=num_classes,
                                   distance_channels=distance_channels,
                                   seg_head_hidden=seg_head_hidden)

    miss, unexp = model.load_state_dict(collections.OrderedDict(decoder_keys), strict=False)
    if miss:
        print(f'  [decoder] missing:    {miss[:3]}...')
    if unexp:
        print(f'  [decoder] unexpected: {unexp[:3]}...')

    model.eval()
    model.to(torch.device(device))
    return model


def load_class_names(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    if 'class_names' not in config:
        raise RuntimeError('Checkpoint config has no "class_names" -- was this '
                           'trained with PrototypeRefinementModule?')
    return list(config['class_names'])


def load_prototype_bank(checkpoint_path, device='cuda:0'):
    from cryosiam.apps.prototype_refinement.module import MultiPrototypeBank

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    class_names = list(config['class_names'])
    n_classes = len(class_names)

    state = checkpoint['state_dict']
    bank_state = {k.replace('proto_bank.', ''): v
                  for k, v in state.items() if k.startswith('proto_bank.')}
    if not bank_state:
        print('  No prototype bank in checkpoint -- returning None.')
        return None

    k_per_class = bank_state['prototypes'].shape[1]
    feat_dim = bank_state['prototypes'].shape[2]
    bank = MultiPrototypeBank(n_classes=n_classes, k_per_class=k_per_class, feat_dim=feat_dim)
    bank.load_state_dict(bank_state)
    bank.to(torch.device(device))
    print(f'  Prototype bank loaded: shape {tuple(bank.prototypes.shape)} (n_classes+1, K, C)')
    return bank


def load_temperature(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    temperature = float(config['parameters']['network'].get('temperature', 0.1))
    print(f'  Temperature loaded: {temperature}')
    return temperature


def prototypes_dict_from_bank(bank, class_names):
    out = {}
    for i, name in enumerate(class_names, start=1):
        out[name] = bank.prototypes[i].detach().clone()  # (K, C)
    return out


def load_prototypes_file(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'prototypes' not in saved:
        raise RuntimeError(f'"{path}" has no "prototypes" key -- was this saved by the '
                           f'prototype-derivation script?')
    prototypes = saved['prototypes']
    print(f'  Loaded prototypes from {path}: '
          f'{ {name: tuple(p.shape) for name, p in prototypes.items()} } '
          f'(mode={saved.get("mode", "?")})')
    return prototypes


def get_background_prototypes(bank):
    return bank.prototypes[0].detach().clone()


def _simple_kmeans(feats, k, n_iters=10, seed=0):
    g = torch.Generator(device=feats.device).manual_seed(seed)
    n = feats.shape[0]
    init_idx = torch.randperm(n, generator=g, device=feats.device)[:k]
    centers = feats[init_idx].clone()

    for _ in range(n_iters):
        sims = feats @ centers.T
        assign = sims.argmax(dim=1)
        new_centers = []
        for kk in range(k):
            sel = (assign == kk)
            if sel.any():
                new_centers.append(F.normalize(feats[sel].mean(0), dim=0))
            else:
                new_centers.append(centers[kk])
        centers = torch.stack(new_centers)
    return centers


def _build_support_transforms(patch_size, data_cfg):
    return Compose([
        LoadImaged(keys=['image', 'mask'], reader=MrcReader(writable=False)),
        EnsureChannelFirstd(keys=['image', 'mask'], channel_dim='no_channel'),
        SpatialPadd(keys=['image', 'mask'], spatial_size=patch_size),
        ScaleIntensityRanged(keys=['image'], a_min=data_cfg['min'], a_max=data_cfg['max'],
                             b_min=0, b_max=1, clip=True),
        CenterSpatialCropd(keys=['image', 'mask'], roi_size=patch_size),
        NormalizeIntensityd(keys=['image'], subtrahend=data_cfg['mean'], divisor=data_cfg['std']),
        EnsureTyped(keys=['image'], data_type='tensor', dtype=torch.float),
        EnsureTyped(keys=['mask'], data_type='tensor', dtype=torch.long),
    ])


@torch.no_grad()
def build_prototypes_from_support(backbone, decoder, support_files_by_class, patch_size,
                                  data_cfg, device='cuda:0', k_per_class=1, min_voxels_warn=20):
    transforms = _build_support_transforms(patch_size, data_cfg)
    was_training_backbone, was_training_decoder = backbone.training, decoder.training
    backbone.eval()
    decoder.eval()

    prototypes = {}
    for class_name, support_files in support_files_by_class.items():
        pooled_feats = []
        voxel_counts = []
        for i, sf in enumerate(support_files):
            class_id = sf.get('class_id', 1)
            sample = transforms(dict(sf))
            image = sample['image'].unsqueeze(0).to(device)
            mask = sample['mask'].to(device)

            feats_list, _ = backbone.get_encoder_features_list(image)
            decoder_out = decoder(feats_list, output_size=mask.shape[-3:])
            # Auxiliary heads are irrelevant for prototype derivation.
            # The normalized embedding is always the first tuple element.
            norm_feats = (
                decoder_out[0]
                if isinstance(decoder_out, tuple)
                else decoder_out
            )
            embed = norm_feats[0]
            flat_feat = embed.reshape(embed.shape[0], -1).T
            flat_lbl = mask.reshape(-1)
            cls_voxels = (flat_lbl == class_id)
            n_voxels = int(cls_voxels.sum().item())
            voxel_counts.append(n_voxels)

            if n_voxels == 0:
                print(f'    [{class_name} {i + 1}/{len(support_files)}] WARNING: 0 voxels '
                      f'of class_id={class_id} in {sf.get("mask", "?")} -- skipping')
                continue
            pooled_feats.append(flat_feat[cls_voxels])

        if not pooled_feats:
            print(f'  WARNING: no usable support patches for "{class_name}" -- skipped entirely')
            continue

        pooled = torch.cat(pooled_feats, dim=0)
        pooled = F.normalize(pooled, dim=1)

        if pooled.shape[0] >= k_per_class:
            protos = _simple_kmeans(pooled, k_per_class)
        else:
            print(f'    NOTE: only {pooled.shape[0]} voxels for "{class_name}", fewer than '
                  f'k_per_class={k_per_class} -- replicating the single mean prototype '
                  f'across all K slots instead of genuine sub-clusters')
            mean_proto = F.normalize(pooled.mean(0), dim=0)
            protos = mean_proto.unsqueeze(0).repeat(k_per_class, 1)

        prototypes[class_name] = protos
        print(f'  {class_name}: {k_per_class} prototype(s) from {len(pooled_feats)}/{len(support_files)} '
              f'patches, voxel counts min={min(voxel_counts)} '
              f'mean={sum(voxel_counts) / len(voxel_counts):.1f} max={max(voxel_counts)}')
        if min(voxel_counts) < min_voxels_warn:
            print(f'    NOTE: some support patches had very few voxels -- consider '
                  f'more/better support examples for a more stable prototype')

    if was_training_backbone:
        backbone.train()
    if was_training_decoder:
        decoder.train()

    return prototypes


def build_tomo_transforms(data_cfg):
    return Compose([
        LoadImaged(keys='image', reader=MrcReader(read_in_mem=True)),
        EnsureChannelFirstd(keys='image'),
        NumpyToTensord(keys='image'),
        ScaleIntensityRanged(keys='image',
                             a_min=data_cfg['min'], a_max=data_cfg['max'],
                             b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys='image',
                            subtrahend=data_cfg['mean'],
                            divisor=data_cfg['std']),
        EnsureTyped(keys='image', data_type='tensor')])


def padded_size_for_even_tiling(original_size, patch_size, overlap=0.5):
    if original_size <= patch_size:
        return patch_size
    stride = int(patch_size * (1 - overlap))
    n_extra_strides = int(np.ceil((original_size - patch_size) / stride))
    return patch_size + n_extra_strides * stride
