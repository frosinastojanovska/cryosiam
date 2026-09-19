import os
import h5py
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import SpatialPad

from cryosiam.utils import parser_helper
from cryosiam.data import PatchIter
from cryosiam.apps.prototype_refinement.utils import (
    load_backbone_model, load_decoder_model, load_class_names,
    load_prototype_bank, prototypes_dict_from_bank,
    build_tomo_transforms, padded_size_for_even_tiling)


def extract_plain_pca_visualization(img, backbone, decoder, patch_size,
                                    input_size, batch_size, patch_iter, device,
                                    n_components=9, n_fit_samples=100_000):
    stride = (1, 1, 1)
    volume_size = input_size
    print(f'  Feature stride: {stride}  feature volume: {volume_size}')

    all_feats, all_coords = [], []

    loader = DataLoader(GridPatchDataset(data=[img], patch_iter=patch_iter),
                        batch_size=batch_size, num_workers=2)

    with torch.no_grad():
        for item in loader:
            patches = item[0].to(device)
            coords = item[1].numpy().astype(int)

            s_feats, _ = backbone.get_encoder_features_list(patches)
            decoder_output = decoder(s_feats, output_size=patches.shape[-3:])
            feat = get_embedding_features(decoder_output)
            feat_np = feat.permute(0, 2, 3, 4, 1).cpu().numpy()

            for b in range(patches.shape[0]):
                c_batch = coords[b][1:]

                if (c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or
                        c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or
                        c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                    continue

                slices = tuple(slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, s) if c[1] >= s
                else slice(c[0] + p // 4, c[1] - p // 4) for c, s, p in zip(c_batch, input_size, patch_size))

                slices2 = tuple(slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s)) if c[1] >= s
                else slice(p // 4, 3 * p // 4) for c, s, p in zip(c_batch, input_size, patch_size))

                block = feat_np[b][slices2]
                all_feats.append(block.reshape(-1, block.shape[-1]))

                zz, yy, xx = np.meshgrid(np.arange(slices[0].start, slices[0].stop),
                                         np.arange(slices[1].start, slices[1].stop),
                                         np.arange(slices[2].start, slices[2].stop),
                                         indexing='ij')

                all_coords.append(np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1))

    all_feats = np.concatenate(all_feats, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)

    if all_feats.shape[0] < n_components:
        print(f'  WARNING: only {all_feats.shape[0]} voxels available -- cannot fit a '
              f'{n_components}-component PCA. Skipping.')
        return None, None, stride

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    centered_all = all_feats - feat_mean

    n_fit = min(n_fit_samples, centered_all.shape[0])
    if n_fit < centered_all.shape[0]:
        fit_idx = np.random.default_rng(42).choice(centered_all.shape[0], n_fit, replace=False)
        fit_feats = centered_all[fit_idx]
        print(f'  Fitting PCA on a random subsample: '
              f'{n_fit:,} of {centered_all.shape[0]:,} voxels')
    else:
        fit_feats = centered_all

    u, s, vt = np.linalg.svd(fit_feats, full_matrices=False)
    components = vt[:n_components]
    explained_variance_ratio = ((s[:n_components] ** 2) / (s ** 2).sum())
    projected = centered_all @ components.T
    axis_vols = np.zeros((n_components, *volume_size), dtype=np.float32)
    axis_vols[:, all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = projected.T

    print(f'  Explained variance ratio: {explained_variance_ratio}')
    return axis_vols, explained_variance_ratio, stride


def extract_class_anchored_pca_visualization(img, backbone, decoder, class_prototype, patch_size,
                                             input_size, batch_size, patch_iter, device,
                                             n_components=9, n_fit_samples=100_000):
    anchor = F.normalize(class_prototype.float().mean(dim=0), dim=0).to(device)

    stride = (1, 1, 1)
    volume_size = input_size
    print(f'  Feature stride: {stride}  feature volume: {volume_size}')

    all_feats, all_coords = [], []
    loader = DataLoader(GridPatchDataset(data=[img], patch_iter=patch_iter),
                        batch_size=batch_size, num_workers=2)

    with torch.no_grad():
        for item in loader:
            patches = item[0].to(device)
            coords = item[1].numpy().astype(int)

            s_feats, _ = backbone.get_encoder_features_list(patches)
            decoder_output = decoder(s_feats, output_size=patches.shape[-3:])
            feat = get_embedding_features(decoder_output)
            centered = (feat.permute(0, 2, 3, 4, 1) - anchor).cpu().numpy()

            for b in range(patches.shape[0]):
                c_batch = coords[b][1:]

                if (c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or
                        c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or
                        c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                    continue

                slices = tuple(
                    slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, s) if c[1] >= s
                    else slice(c[0] + p // 4, c[1] - p // 4) for c, s, p in zip(c_batch, input_size, patch_size))

                slices2 = tuple(slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s)) if c[1] >= s
                else slice(p // 4, 3 * p // 4) for c, s, p in zip(c_batch, input_size, patch_size))

                block = centered[b][slices2]
                all_feats.append(block.reshape(-1, block.shape[-1]))

                zz, yy, xx = np.meshgrid(np.arange(slices[0].start, slices[0].stop),
                                         np.arange(slices[1].start, slices[1].stop),
                                         np.arange(slices[2].start, slices[2].stop),
                                         indexing='ij')

                all_coords.append(np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1))

    all_feats = np.concatenate(all_feats, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)

    if all_feats.shape[0] < n_components:
        print(f'  WARNING: only {all_feats.shape[0]} voxels available -- cannot fit a '
              f'{n_components}-component PCA. Skipping.')
        return None, None, stride

    n_fit = min(n_fit_samples, all_feats.shape[0])
    if n_fit < all_feats.shape[0]:
        fit_idx = np.random.default_rng(42).choice(
            all_feats.shape[0], n_fit, replace=False
        )
        fit_feats = all_feats[fit_idx]
        print(f'  Fitting PCA on a random subsample: '
              f'{n_fit:,} of {all_feats.shape[0]:,} voxels')
    else:
        fit_feats = all_feats

    u, s, vt = np.linalg.svd(fit_feats, full_matrices=False)
    components = vt[:n_components]
    explained_variance_ratio = ((s[:n_components] ** 2) / (s ** 2).sum())
    projected = all_feats @ components.T
    axis_vols = np.zeros((n_components, *volume_size), dtype=np.float32)

    axis_vols[:, all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = projected.T

    print(f'  Explained variance ratio: {explained_variance_ratio}')
    return axis_vols, explained_variance_ratio, stride


def pca_axes_to_rgb(axis_vols, percentile_clip=(1, 99)):
    def _normalize_axis(axis, lo_pct, hi_pct):
        lo, hi = np.percentile(axis, [lo_pct, hi_pct])
        if hi <= lo:  # degenerate case: near-constant axis, avoid divide-by-zero
            return np.zeros_like(axis)
        return np.clip((axis - lo) / (hi - lo), 0, 1)

    channels = [_normalize_axis(axis_vols[i], *percentile_clip) for i in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)  # (D, H, W, 3)


def get_embedding_features(decoder_output):
    if isinstance(decoder_output, (tuple, list)):
        decoder_output = decoder_output[0]

    if not torch.is_tensor(decoder_output) or decoder_output.ndim != 5:
        raise ValueError(
            f'Expected feature tensor with shape [B, C, D, H, W], '
            f'got {type(decoder_output)}'
        )

    return decoder_output


def upsample_axes(axis_vols, stride, original_size):
    n = axis_vols.shape[0]
    D, H, W = axis_vols.shape[1:]

    real_fpn_size = [int(np.ceil(original_size[i] / stride[i])) for i in range(3)]
    real_fpn_size = [min(real_fpn_size[i], axis_vols.shape[1 + i]) for i in range(3)]
    cropped = axis_vols[:, :real_fpn_size[0], :real_fpn_size[1], :real_fpn_size[2]]

    if list(cropped.shape[1:]) == list(original_size):
        return cropped

    ups = []
    for i in range(n):
        up = F.interpolate(
            torch.from_numpy(cropped[i]).unsqueeze(0).unsqueeze(0),
            size=list(original_size), mode='trilinear', align_corners=False
        )[0, 0].numpy()
        ups.append(up)
    return np.stack(ups)


def _write_pca_to_h5(hf, prefix, axis_vols, var_ratio, pca_stride, original_size):
    axis_vols = upsample_axes(axis_vols, pca_stride, original_size)

    for i in range(axis_vols.shape[0]):
        hf.create_dataset(f'{prefix}pca_axis{i + 1}', data=axis_vols[i],
                          compression='gzip', compression_opts=4)

    for start in (0, 3, 6):
        if axis_vols.shape[0] < start + 3:
            continue
        rgb = pca_axes_to_rgb(axis_vols[start:start + 3])
        suffix = f'{start + 1}{start + 2}{start + 3}'
        rgb_ds = hf.create_dataset(f'{prefix}pca_rgb_{suffix}', data=rgb,
                                   compression='gzip', compression_opts=4)
        rgb_ds.attrs['explained_variance_ratio'] = var_ratio[start:start + 3]
        rgb_ds.attrs['original_size'] = original_size
        print(f'    {prefix}pca_rgb_{suffix}: shape {rgb.shape}, range [{rgb.min():.3f}, {rgb.max():.3f}]')


def main(config_file_path, filename=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg['trained_model']
    backbone = load_backbone_model(checkpoint_path, device)
    class_names = cfg.get('class_names') or load_class_names(checkpoint_path)
    decoder = load_decoder_model(checkpoint_path, class_names, device)

    compute_class_pca = bool(cfg.get('compute_class_pca', True))
    pca_anchor_class = cfg.get('pca_anchor_class')

    prototypes = None
    classes_to_process = []
    if compute_class_pca:
        bank = load_prototype_bank(checkpoint_path, device)
        if bank is None:
            print('  WARNING: compute_class_pca=True but no prototype bank in the checkpoint -- '
                  'class-anchored PCA will be skipped, only the plain result will be computed.')
            compute_class_pca = False
        else:
            prototypes = prototypes_dict_from_bank(bank, class_names)
            if pca_anchor_class is not None:
                if pca_anchor_class not in prototypes:
                    raise ValueError(f'pca_anchor_class={pca_anchor_class!r} not found -- available: '
                                     f'{list(prototypes.keys())}')
                classes_to_process = [pca_anchor_class]
            else:
                classes_to_process = list(prototypes.keys())
            print(f'Class-anchored PCA will also be computed for: {classes_to_process}')

    data_cfg = cfg['parameters']['data']
    patch_size = data_cfg['patch_size']
    file_ext = cfg['file_extension']
    batch_size = cfg['hyper_parameters']['batch_size']
    reuse = bool(cfg.get('reuse_predictions', True))
    pca_n_fit_samples = int(cfg.get('pca_n_fit_samples', 100_000))

    os.makedirs(cfg['prediction_folder'], exist_ok=True)

    tomo_transforms = build_tomo_transforms(data_cfg)
    patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0),
                           overlap=(0, 0.5, 0.5, 0.5))

    data_folder = cfg['data_folder']
    files = cfg.get('test_files') or [
        f for f in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, f)) and f.endswith(file_ext)]
    if filename:
        files = [filename]

    test_data = [{'image': os.path.join(data_folder, f), 'file_name': os.path.join(data_folder, f)}
                 for f in files]
    test_ds = Dataset(data=test_data, transform=tomo_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print(f'\n{len(files)} test tomogram(s)')

    with torch.no_grad():
        for test_sample in test_loader:
            current_file = test_sample['file_name'][0]
            tomo_name = os.path.basename(current_file).split(file_ext)[0]
            original_size = list(test_sample['image'][0][0].shape)

            out_path = os.path.join(cfg['prediction_folder'], f'{tomo_name}_pca.h5')

            print(f'\n{"=" * 55}')
            print(f'Tomogram: {tomo_name}  size: {original_size}')

            if reuse and os.path.exists(out_path):
                print('  Skipping — output already exists')
                continue

            img = test_sample['image'][0]
            padded_size = [padded_size_for_even_tiling(original_size[i], patch_size[i])
                           for i in range(3)]
            pad_transform = SpatialPad(spatial_size=padded_size, method='end', mode='edge')
            img = pad_transform(img)
            input_size = list(img[0].shape)

            with h5py.File(out_path, 'w') as hf:
                print('  Extracting plain PCA visualization...')
                axis_vols, var_ratio, pca_stride = extract_plain_pca_visualization(
                    img, backbone, decoder, patch_size, input_size, batch_size,
                    patch_iter, device, n_fit_samples=pca_n_fit_samples)

                if axis_vols is None:
                    print('  Skipped plain PCA -- not enough voxels to fit a PCA.')
                else:
                    _write_pca_to_h5(hf, '', axis_vols, var_ratio, pca_stride, original_size)

                if compute_class_pca:
                    for class_name in classes_to_process:
                        print(f'  Extracting class-anchored PCA visualization (anchor: {class_name})...')
                        axis_vols, var_ratio, pca_stride = extract_class_anchored_pca_visualization(
                            img, backbone, decoder, prototypes[class_name], patch_size, input_size,
                            batch_size, patch_iter, device, n_fit_samples=pca_n_fit_samples)

                        if axis_vols is None:
                            print(f'  Skipped {class_name} -- not enough voxels to fit a PCA.')
                            continue
                        _write_pca_to_h5(hf, f'{class_name}_', axis_vols, var_ratio, pca_stride, original_size)

            size_mb = os.path.getsize(out_path) / 1e6
            print(f'  Saved → {os.path.basename(out_path)}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    parser = parser_helper('Prototype refinement: PCA feature visualization')
    args = parser.parse_args()
    main(args.config_file, getattr(args, 'filename', None))
