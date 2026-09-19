import os
import yaml
import torch
import mrcfile
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.data import GridPatchDataset
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, ScaleIntensityRanged,
    SpatialPad, EnsureChannelFirstd, EnsureTyped)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, PatchIter
from cryosiam.apps.prototype_matching import load_search_model, load_decoder


def build_tomo_transforms(data_cfg):
    return Compose([
        LoadImaged(keys='image', reader=MrcReader(read_in_mem=True)),
        EnsureChannelFirstd(keys='image'),
        NumpyToTensord(keys='image'),
        ScaleIntensityRanged(keys='image', a_min=data_cfg['min'], a_max=data_cfg['max'],
                             b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys='image', subtrahend=data_cfg['mean'], divisor=data_cfg['std']),
        EnsureTyped(keys='image', data_type='tensor')])


def _stitch_slices(c, vol_size, patch_size):
    slices = []
    for i in range(3):
        z0, z1, p, s = c[i][0], c[i][1], patch_size[i], vol_size[i]
        if z0 == 0 and z1 >= s:
            slices.append(slice(0, s))
        elif z0 == 0:
            slices.append(slice(z0, z1 - p // 4))
        elif z1 >= s:
            slices.append(slice(z0 + p // 4, z1))
        else:
            slices.append(slice(z0 + p // 4, z1 - p // 4))
    return slices


def _patch_slices(c, vol_size, patch_size):
    slices = []
    for i in range(3):
        z0, z1, p, s = c[i][0], c[i][1], patch_size[i], vol_size[i]
        if z0 == 0 and z1 >= s:
            slices.append(slice(0, s))
        elif z0 == 0:
            slices.append(slice(0, 3 * p // 4))
        elif z1 >= s:
            slices.append(slice(p // 4, p - (z1 - s)))
        else:
            slices.append(slice(p // 4, 3 * p // 4))
    return slices


def extract_fpn_features(img, encoder, decoder, patch_size, input_size,
                         batch_size, patch_iter, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *patch_size, device=device)
        s_dummy, _ = encoder.get_encoder_features_list(dummy)
        p_dummy = decoder._fpn_body(s_dummy)
        fpn_size = list(p_dummy.shape[-3:])

    out_ch = decoder.out_channels
    stride = [patch_size[i] // fpn_size[i] for i in range(3)]
    fpn_vol_size = [input_size[i] // stride[i] for i in range(3)]
    feat_vol = np.zeros((out_ch, *fpn_vol_size), dtype=np.float32)

    print(f'  FPN stride: {stride}  FPN volume: {fpn_vol_size}')

    loader = DataLoader(
        GridPatchDataset(data=[img], patch_iter=patch_iter),
        batch_size=batch_size, num_workers=2)

    with torch.no_grad():
        for item in loader:
            patches = item[0].to(device)
            coords = item[1].numpy().astype(int)

            s_feats, _ = encoder.get_encoder_features_list(patches)
            p = decoder._fpn_body(s_feats)
            p_np = p.cpu().numpy()

            for b in range(patches.shape[0]):
                c = coords[b][1:]
                fc = [[c[i][0] // stride[i], c[i][1] // stride[i]]
                      for i in range(3)]

                ps = list(p.shape[-3:])
                os_ = _stitch_slices(fc, fpn_vol_size, ps)
                sl_ = _patch_slices(fc, fpn_vol_size, ps)

                feat_vol[:, os_[0], os_[1], os_[2]] = \
                    p_np[b][:, sl_[0], sl_[1], sl_[2]]

    return feat_vol


def _resample_mask_to_fpn(mask, feat_vol_shape):
    C_f, D_f, H_f, W_f = feat_vol_shape
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_fpn = F.interpolate(mask_t, size=(D_f, H_f, W_f), mode='nearest')[0, 0].long().numpy()
    return mask_fpn


def _simple_kmeans(feats, k, n_iters=10, seed=0):
    feats_t = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = feats_t.shape[0]
    k = min(k, n)
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = F.normalize(feats_t[init_idx].clone(), dim=1)

    for _ in range(n_iters):
        sims = feats_t @ centers.T
        assign = sims.argmax(dim=1)
        new_centers = []
        for kk in range(k):
            sel = (assign == kk)
            if sel.any():
                new_centers.append(F.normalize(feats_t[sel].mean(0), dim=0))
            else:
                new_centers.append(centers[kk])
        centers = torch.stack(new_centers)
    return centers.numpy()


def _build_prototype(voxels, use_kmeans, k_per_class):
    if voxels.shape[0] == 0:
        return None
    if not use_kmeans:
        mean = voxels.mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-8)
        return mean[np.newaxis, :]
    return _simple_kmeans(voxels, k_per_class)


def derive_voxels_instance_mode(feat_vol, mask_fpn, class_instance_ids):
    flat_feat = feat_vol.reshape(feat_vol.shape[0], -1).T
    flat_mask = mask_fpn.reshape(-1)

    class_voxels = {}
    for class_name, ids in class_instance_ids.items():
        sel = np.isin(flat_mask, ids)
        voxels = flat_feat[sel]
        if voxels.shape[0] == 0:
            print(f'  WARNING: no voxels found for instance ids {ids} (class "{class_name}")')
            continue
        norms = np.linalg.norm(voxels, axis=1, keepdims=True)
        voxels = voxels / np.maximum(norms, 1e-8)
        class_voxels[class_name] = voxels
        print(f'  {class_name}: {voxels.shape[0]:,} voxels from instances {ids}')
    return class_voxels


def derive_voxels_scribble_mode(feat_vol, mask_fpn, class_label_ids):
    flat_feat = feat_vol.reshape(feat_vol.shape[0], -1).T
    flat_mask = mask_fpn.reshape(-1)

    class_voxels = {}
    for class_name, label_id in class_label_ids.items():
        sel = (flat_mask == label_id)
        voxels = flat_feat[sel]
        if voxels.shape[0] == 0:
            print(f'  WARNING: no voxels found for label {label_id} (class "{class_name}")')
            continue
        norms = np.linalg.norm(voxels, axis=1, keepdims=True)
        voxels = voxels / np.maximum(norms, 1e-8)
        class_voxels[class_name] = voxels
        print(f'  {class_name}: {voxels.shape[0]:,} voxels (label={label_id})')
    return class_voxels


def main(config_file_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg['trained_model']
    encoder = load_search_model(checkpoint_path, device)
    decoder = load_decoder(checkpoint_path, device)
    encoder.eval()
    decoder.eval()

    data_cfg = cfg['parameters']['data']
    patch_size = data_cfg['patch_size']
    batch_size = cfg['hyper_parameters']['batch_size']

    mode = cfg['mode']  # 'instance' | 'scribble'
    if mode not in ('instance', 'scribble'):
        raise ValueError(f'mode must be "instance" or "scribble", got "{mode}"')
    use_kmeans = bool(cfg.get('use_kmeans', False))
    k_per_class = int(cfg.get('k_per_class', 3))
    if k_per_class < 1:
        raise ValueError('k_per_class must be >= 1.')
    output_path = cfg['output_path']

    references = cfg['references']
    if not references:
        raise ValueError('No references specified.')

    print(f'Mode: {mode}')
    print(f'Prototypes per class: {"K-means, K=" + str(k_per_class) if use_kmeans else "single (mean)"}')
    print(f'{len(references)} reference file(s)')

    tomo_transforms = build_tomo_transforms(data_cfg)
    patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0),
                           overlap=(0, 0.5, 0.5, 0.5))
    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='edge')

    pooled_voxels = {}  # class_name -> list of (M, C) arrays, pooled across ALL references

    for ref in references:
        print(f'\nProcessing reference: {ref["image"]}')
        sample = tomo_transforms({'image': ref['image'], 'file_name': ref['image']})
        img = pad_transform(sample['image'])
        input_size = list(img[0].shape)

        feat_vol = extract_fpn_features(img, encoder, decoder, patch_size, input_size, batch_size, patch_iter, device)

        mask = mrcfile.open(ref['mask'], permissive=True).data.copy().astype(np.int32)
        mask_fpn = _resample_mask_to_fpn(mask, feat_vol.shape)

        if mode == 'instance':
            if 'class_instance_ids' not in ref:
                raise ValueError(f'Missing class_instance_ids for reference: {ref["image"]}')
            ref_voxels = derive_voxels_instance_mode(feat_vol, mask_fpn, ref['class_instance_ids'])
        else:
            if 'class_label_ids' not in ref:
                raise ValueError(f'Missing class_label_ids for reference: {ref["image"]}')
            ref_voxels = derive_voxels_scribble_mode(feat_vol, mask_fpn, ref['class_label_ids'])

        for class_name, voxels in ref_voxels.items():
            pooled_voxels.setdefault(class_name, []).append(voxels)

    print(f'\nBuilding prototypes...')
    prototypes = {}
    for class_name, voxel_list in pooled_voxels.items():
        all_voxels = np.concatenate(voxel_list, axis=0)
        proto = _build_prototype(all_voxels, use_kmeans, k_per_class)
        if proto is None:
            continue
        prototypes[class_name] = torch.from_numpy(proto).float()  # (k, C)
        print(f'  {class_name}: {proto.shape[0]} prototype(s) from {all_voxels.shape[0]:,} total pooled voxels')

    if not prototypes:
        raise RuntimeError('No prototypes could be built -- check references/mask ids.')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save({
        'prototypes': prototypes,  # {class_name: (k, C) tensor}
        'mode': mode,
        'use_kmeans': use_kmeans,
        'k_per_class': k_per_class,
    }, output_path)
    print(f'\nSaved {len(prototypes)} class prototype set(s) → {output_path}')


if __name__ == '__main__':
    parser = parser_helper('Derive class prototypes from instance-segmentation or scribble annotations')
    args = parser.parse_args()
    main(args.config_file)
