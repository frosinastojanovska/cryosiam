import os
import h5py
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    SpatialPad,
    EnsureChannelFirstd,
    EnsureTyped,
)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, PatchIter
from cryosiam.apps.prototype_matching import load_encoder, load_decoder


def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    search_model = load_encoder(checkpoint_path, device)
    decoder = load_decoder(checkpoint_path, device)
    search_model.eval()
    decoder.eval()
    return search_model, decoder, config


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


def extract_fpn_features(img, search_model, decoder, patch_size, input_size, batch_size,
                         patch_iter, device, upsample: bool = True):
    out_ch = decoder.out_channels

    if upsample:
        feat_vol = np.zeros((out_ch, *input_size), dtype=np.float32)
        stride = None
    else:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *patch_size, device=device)
            s_dummy, _ = search_model.get_encoder_features_list(dummy)
            p_dummy = decoder._fpn_body(s_dummy)
            fpn_size = list(p_dummy.shape[-3:])
        stride = [patch_size[i] // fpn_size[i] for i in range(3)]
        fpn_vol_size = [input_size[i] // stride[i] for i in range(3)]
        feat_vol = np.zeros((out_ch, *fpn_vol_size), dtype=np.float32)
        print(f'  FPN stride: {stride}  FPN volume: {fpn_vol_size}')

    patch_dataset = GridPatchDataset(data=[img], patch_iter=patch_iter)
    loader = DataLoader(patch_dataset, batch_size=batch_size,
                        num_workers=2)

    with torch.no_grad():
        for item in loader:
            patches = item[0].to(device)
            coords = item[1].numpy().astype(int)
            B = patches.shape[0]

            s_feats, _ = search_model.get_encoder_features_list(patches)
            p = decoder._fpn_body(s_feats)
            p_up = (F.interpolate(p, size=tuple(patch_size), mode='trilinear', align_corners=False) if upsample else p)
            p_np = p_up.cpu().numpy()

            for b in range(B):
                c = coords[b][1:]

                if upsample:
                    os_ = _stitch_slices(c, input_size, patch_size)
                    ps_ = _patch_slices(c, input_size, patch_size)
                    feat_vol[:, os_[0], os_[1], os_[2]] = p_np[b][:, ps_[0], ps_[1], ps_[2]]

                else:
                    fc = [[c[i][0] // stride[i], c[i][1] // stride[i]] for i in range(3)]
                    fp_size = list(p.shape[-3:])
                    os_ = _stitch_slices(fc, fpn_vol_size, fp_size)
                    ps_ = _patch_slices(fc, fpn_vol_size, fp_size)

                    feat_vol[:, os_[0], os_[1], os_[2]] = p_np[b][:, ps_[0], ps_[1], ps_[2]]

    return feat_vol, stride


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


def _central_z_indices(D, n_slices):
    n_slices = min(n_slices, D)
    center = D // 2
    half = n_slices // 2
    start = max(0, center - half)
    end = min(D, start + n_slices)
    start = max(0, end - n_slices)  # re-clamp if we hit the upper edge
    return np.arange(start, end)


def detect_fg(feat_vol_fpn, z_slice=None, fg_threshold=None,
              fg_sign=None, fg_percentile=85, n_center_slices=5):
    C, D, H, W = feat_vol_fpn.shape

    if z_slice is not None:
        z_indices = np.array([int(np.clip(z_slice, 0, D - 1))])
    else:
        z_indices = _central_z_indices(D, n_center_slices)

    raw_z = feat_vol_fpn[:, z_indices, :, :].reshape(C, -1).T
    norms = np.linalg.norm(raw_z, axis=1)
    vox_z = raw_z / np.maximum(norms[:, np.newaxis], 1e-8)

    pca1 = PCA(n_components=1, random_state=42)
    pc1 = pca1.fit_transform(vox_z)[:, 0]

    # unsigned threshold, used ONLY to determine sign below
    thr_for_sign = fg_threshold if fg_threshold is not None \
        else np.percentile(pc1, fg_percentile)

    if fg_sign is None:
        pos = pc1 > thr_for_sign
        pos_norm = norms[pos].mean() if pos.any() else 0.0
        neg_norm = norms[~pos].mean() if (~pos).any() else 0.0
        fg_sign = 1 if pos_norm >= neg_norm else -1
        print(f'  L2 norm: pos={pos_norm:.3f}  neg={neg_norm:.3f}'
              f'  -> fg_sign={fg_sign:+d}')

    if fg_threshold is not None:
        thr = float(fg_threshold)
    else:
        thr_signed = float(np.percentile(pc1 * fg_sign, fg_percentile))
        thr = thr_signed * fg_sign  # back to the original, unsigned pc1 scale

    fg_mask_z = (pc1 * fg_sign) > (thr * fg_sign)
    n_fg = fg_mask_z.sum()
    print(f'  Z={list(z_indices)}: thr={thr:.3f}  '
          f'fg={n_fg:,}/{len(pc1):,} ({100 * n_fg / len(pc1):.1f}%)')
    if n_fg < 100:
        print('  WARNING: very few fg voxels — '
              'lower pca_fg_percentile or pca_fg_threshold')

    vox_all = feat_vol_fpn.reshape(C, -1).T
    n_all = np.linalg.norm(vox_all, axis=1, keepdims=True)
    vox_all = vox_all / np.maximum(n_all, 1e-8)
    pc1_vol = (pca1.transform(vox_all)[:, 0] * fg_sign).reshape(
        D, H, W).astype(np.float32)

    pc1_flat = pc1_vol.flatten()
    print(f'\n  PC1 percentile table:')
    print(f'  {"Percentile":>11}  {"Threshold":>10}  {"fg%":>7}')
    print(f'  {"-" * 32}')
    for p in [70, 75, 80, 85, 90, 95]:
        v = float(np.percentile(pc1_flat, p))
        pct = 100 * (pc1_flat > v).mean()
        marker = ' ← current' if fg_threshold is None \
                                 and p == fg_percentile else ''
        print(f'  {p:>10}%  {v:>10.3f}  {pct:>6.1f}%{marker}')

    return pca1, fg_sign, float(thr), pc1_vol


def _get_fg_voxels(feat_vol_fpn, pca1_fg, fg_sign, fg_thr, n_slices):
    C, D, H, W = feat_vol_fpn.shape
    z_indices = _central_z_indices(D, n_slices)
    fg_samples = []

    for z in z_indices:
        raw = feat_vol_fpn[:, z, :, :].reshape(C, -1).T
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        v_z = raw / np.maximum(norms, 1e-8)
        pc1_z = pca1_fg.transform(v_z)[:, 0] * fg_sign
        fg_z = pc1_z > (fg_thr * fg_sign)
        if fg_z.any():
            fg_samples.append(v_z[fg_z])
        else:
            print(f'  Z={z}: no fg voxels, skipping')

    if not fg_samples:
        print('  WARNING: no fg voxels found in central slices — '
              'falling back to full volume.')
        voxels = feat_vol_fpn.reshape(C, -1).T
        norms = np.linalg.norm(voxels, axis=1, keepdims=True)
        fg_samples.append(voxels / np.maximum(norms, 1e-8))

    return np.concatenate(fg_samples, axis=0)


def _crop_to_real_region(channels_dhw, stride, original_size):
    if stride is None:
        return channels_dhw  # upsample=True path never hits this -- no padding-crop needed there
    D, H, W = channels_dhw[0].shape
    real_size = [int(np.ceil(original_size[i] / stride[i])) for i in range(3)]
    real_size = [min(real_size[0], D), min(real_size[1], H), min(real_size[2], W)]
    return [c[:real_size[0], :real_size[1], :real_size[2]] for c in channels_dhw]


def _apply_reducer(feat_vol_fpn, pca1_fg, reducer, fg_sign, fg_thr,
                   clip_pct=1, original_size=None, stride=None):
    C, D, H, W = feat_vol_fpn.shape
    voxels = feat_vol_fpn.reshape(C, -1).T
    norms = np.linalg.norm(voxels, axis=1, keepdims=True)
    voxels = voxels / np.maximum(norms, 1e-8)

    pc1 = pca1_fg.transform(voxels)[:, 0] * fg_sign
    fg_mask = pc1 > (fg_thr * fg_sign)  # FIX: see _get_fg_voxels for the full explanation

    print(f'  Transforming {fg_mask.sum():,} fg voxels...')
    fg_proj = reducer.transform(voxels[fg_mask])

    return _normalise_and_reshape(fg_proj, fg_mask, D, H, W,
                                  clip_pct, original_size, stride)


def _normalise_and_reshape(fg_proj, fg_mask, D, H, W,
                           clip_pct, original_size=None, stride=None):
    n_comp = fg_proj.shape[1]
    N = D * H * W
    components = np.zeros((n_comp, N), dtype=np.float32)

    for i in range(n_comp):
        comp = fg_proj[:, i]
        lo = np.percentile(comp, clip_pct)
        hi = np.percentile(comp, 100 - clip_pct)
        components[i, fg_mask] = np.clip(
            (comp - lo) / (hi - lo + 1e-8), 0, 1)

    channels = [components[i].reshape(D, H, W) for i in range(n_comp)]

    if original_size is not None and list(original_size) != [D, H, W]:
        channels = _crop_to_real_region(channels, stride, original_size)
        channels = [
            F.interpolate(
                torch.from_numpy(c).unsqueeze(0).unsqueeze(0),
                size=list(original_size),
                mode='trilinear', align_corners=False
            )[0, 0].numpy()
            for c in channels]

    return channels


def fit_pca(feat_vol_fpn, pca1_fg, fg_sign, fg_thr, n_slices=5):
    fg_voxels = _get_fg_voxels(feat_vol_fpn, pca1_fg, fg_sign, fg_thr, n_slices)

    pca2 = PCA(n_components=9, random_state=42)
    pca2.fit(fg_voxels)

    var = pca2.explained_variance_ratio_
    print(f'  PCA variance (fit on {fg_voxels.shape[0]:,} fg voxels '
          f'from {n_slices} central slices):')
    print(f'    PC1-3: {var[0]:.1%}  {var[1]:.1%}  {var[2]:.1%}'
          f'  (total {sum(var[:3]):.1%})')
    print(f'    PC4-6: {var[3]:.1%}  {var[4]:.1%}  {var[5]:.1%}'
          f'  (total {sum(var[3:6]):.1%})')
    print(f'    PC7-9: {var[6]:.1%}  {var[7]:.1%}  {var[8]:.1%}'
          f'  (total {sum(var[6:9]):.1%})')
    return pca2


def apply_pca(feat_vol, pca1_fg, pca2, fg_sign, fg_thr, clip_pct=1,
              original_size=None, stride=None):
    """Applied to the FULL volume regardless of the central-slice fit sample."""
    C, D, H, W = feat_vol.shape
    voxels = feat_vol.reshape(C, -1).T
    norms = np.linalg.norm(voxels, axis=1, keepdims=True)
    voxels = voxels / np.maximum(norms, 1e-8)

    pc1 = pca1_fg.transform(voxels)[:, 0] * fg_sign
    fg_mask = pc1 > (fg_thr * fg_sign)  # FIX: see _get_fg_voxels for the full explanation
    fg_proj = pca2.transform(voxels[fg_mask])

    all_ch = _normalise_and_reshape(fg_proj, fg_mask, D, H, W, clip_pct,
                                    original_size, stride)
    return all_ch[:3], all_ch[3:6], all_ch[6:9]


def fit_umap(feat_vol_fpn, pca1_fg, fg_sign, fg_thr,
             n_neighbors=15, n_fit_samples=50_000, n_slices=5):
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError('pip install umap-learn')

    all_fg = _get_fg_voxels(feat_vol_fpn, pca1_fg, fg_sign, fg_thr, n_slices)
    n_fit = min(n_fit_samples, all_fg.shape[0])
    idx = np.random.default_rng(42).choice(
        all_fg.shape[0], size=n_fit, replace=False)

    print(f'  Fitting UMAP on {n_fit:,} fg voxels from {n_slices} central '
          f'slices (n_neighbors={n_neighbors})...')
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors,
                   min_dist=0.1, metric='cosine',
                   random_state=42, low_memory=True, n_jobs=-1)
    reducer.fit(all_fg[idx])
    print('  UMAP fitted.')
    return reducer


def apply_umap(feat_vol_fpn, pca1_fg, reducer, fg_sign, fg_thr,
               clip_pct=1, original_size=None, stride=None):
    return _apply_reducer(feat_vol_fpn, pca1_fg, reducer, fg_sign,
                          fg_thr, clip_pct, original_size, stride)


def fit_cluster(feat_vol_fpn, pca1_fg, fg_sign, fg_thr,
                n_clusters=20, n_fit_samples=50_000, n_slices=5):
    from sklearn.cluster import MiniBatchKMeans

    all_fg = _get_fg_voxels(feat_vol_fpn, pca1_fg, fg_sign, fg_thr, n_slices)
    n_fit = min(n_fit_samples, all_fg.shape[0])
    idx = np.random.default_rng(42).choice(
        all_fg.shape[0], size=n_fit, replace=False)

    print(f'  Fitting K-means ({n_clusters} clusters) on '
          f'{n_fit:,} fg voxels from {n_slices} central slices...')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             n_init=3, batch_size=10_000)
    kmeans.fit(all_fg[idx])
    print('  K-means fitted.')
    return kmeans


def apply_cluster(feat_vol_fpn, pca1_fg, kmeans, fg_sign, fg_thr,
                  original_size=None, stride=None):
    import matplotlib.cm as cm
    from scipy.ndimage import zoom

    C, D, H, W = feat_vol_fpn.shape
    voxels = feat_vol_fpn.reshape(C, -1).T
    norms = np.linalg.norm(voxels, axis=1, keepdims=True)
    voxels = voxels / np.maximum(norms, 1e-8)

    pc1 = pca1_fg.transform(voxels)[:, 0] * fg_sign
    fg_mask = pc1 > (fg_thr * fg_sign)  # FIX: see _get_fg_voxels for the full explanation

    print(f'  Assigning cluster colors to {fg_mask.sum():,} fg voxels...')
    labels = kmeans.predict(voxels[fg_mask])  # (N_fg,)

    cmap = cm.get_cmap('tab20', kmeans.n_clusters)

    fg_colors = np.array([cmap(int(k))[:3] for k in labels],
                         dtype=np.float32)  # (N_fg, 3)

    color_vol = np.zeros((D * H * W, 3), dtype=np.float32)
    color_vol[fg_mask] = fg_colors

    channels = [color_vol[:, i].reshape(D, H, W) for i in range(3)]

    if original_size is not None and list(original_size) != [D, H, W]:
        channels = _crop_to_real_region(channels, stride, original_size)
        factors = [original_size[i] / channels[0].shape[i] for i in range(3)]
        channels = [zoom(c, factors, order=0) for c in channels]

    return channels


def _save_nonlinear(out_path, rgb, pc1_vol, original_size,
                    fg_sign, fg_thr, fg_percentile, method):
    """Used for umap and cluster outputs."""
    key = f'{method}_rgb'  # 'umap_rgb' or 'cluster_rgb'
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset(key, data=rgb,
                          compression='lzf', compression_opts=4)
        hf.create_dataset('fg_pc1', data=pc1_vol,
                          compression='lzf', compression_opts=4)
        hf.attrs['original_size'] = original_size
        hf.attrs['fg_sign'] = fg_sign
        hf.attrs['fg_thr'] = fg_thr
        hf.attrs['fg_percentile'] = fg_percentile
        hf.attrs['method'] = method
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved → {os.path.basename(out_path)}  ({size_mb:.1f} MB)')
    print(f'    {key}: 3 channels')
    print(f'    fg_pc1: fg/bg map for interactive thresholding')


def _save_pca(out_path, pca_rgb_123, pca_rgb_456, pca_rgb_789, pc1_vol, original_size,
              fg_sign, fg_thr, fg_percentile):
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('pca_rgb_123', data=pca_rgb_123,
                          compression='lzf', compression_opts=4)
        hf.create_dataset('pca_rgb_456', data=pca_rgb_456,
                          compression='lzf', compression_opts=4)
        hf.create_dataset('pca_rgb_789', data=pca_rgb_789,
                          compression='lzf', compression_opts=4)
        hf.create_dataset('fg_pc1', data=pc1_vol,
                          compression='lzf', compression_opts=4)
        hf.attrs['original_size'] = original_size
        hf.attrs['fg_sign'] = fg_sign
        hf.attrs['fg_thr'] = fg_thr
        hf.attrs['fg_percentile'] = fg_percentile
        hf.attrs['method'] = 'pca'
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved → {os.path.basename(out_path)}  ({size_mb:.1f} MB)')
    print(f'    pca_rgb_123: PC1-3')
    print(f'    pca_rgb_456: PC4-6')
    print(f'    pca_rgb_789: PC7-9')
    print(f'    fg_pc1:      fg/bg map for interactive thresholding')


def main(config_file_path, filename=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg['trained_model']
    search_model, decoder, config = load_models(checkpoint_path, device)

    data_cfg = cfg['parameters']['data']
    patch_size = data_cfg['patch_size']
    file_ext = cfg['file_extension']
    batch_size = cfg['hyper_parameters']['batch_size']

    os.makedirs(cfg['prediction_folder'], exist_ok=True)

    tomo_transforms = build_tomo_transforms(data_cfg)
    patch_iter = PatchIter(patch_size=tuple(patch_size),
                           start_pos=(0, 0, 0),
                           overlap=(0, 0.5, 0.5, 0.5))

    files = cfg.get('test_files') or [
        f for f in os.listdir(cfg['data_folder'])
        if os.path.isfile(os.path.join(cfg['data_folder'], f))
           and f.endswith(file_ext)]
    if filename:
        files = [filename]

    method = cfg.get('pca_method', 'umap')  # pca | umap | cluster
    z_slice = cfg.get('pca_z_slice', None)
    fg_threshold = cfg.get('pca_fg_threshold', None)
    fg_sign = cfg.get('pca_fg_sign', None)
    fg_percentile = cfg.get('pca_fg_percentile', 85)
    filter_foreground = bool(cfg.get('filter_foreground', False))
    clip_pct = cfg.get('pca_clip_percentile', 1)
    n_neighbors = cfg.get('umap_n_neighbors', 15)
    n_fit = cfg.get('umap_n_fit_samples', 50_000)
    n_clusters = cfg.get('n_clusters', 20)

    vis_downsample_factor = int(cfg.get('vis_downsample_factor', 1))

    n_fit_slices = cfg.get('n_fit_slices', 5)

    print(f'Mode: {method.upper()}  (fg_pc1 always saved alongside the colorization output)')
    print(f'Fitting on {n_fit_slices} central Z slices (n_fit_slices in config)')
    if vis_downsample_factor > 1:
        print(f'Display resolution downsample factor: {vis_downsample_factor}x')

    test_data = [{'image': os.path.join(cfg['data_folder'], f),
                  'file_name': os.path.join(cfg['data_folder'], f)}
                 for f in files]
    test_dataset = Dataset(data=test_data, transform=tomo_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1,
                             collate_fn=list_data_collate)

    for test_sample in test_loader:
        current_file = test_sample['file_name'][0]
        tomo_name = os.path.basename(current_file).split(file_ext)[0]
        original_size = list(test_sample['image'][0][0].shape)

        display_size = [max(1, s // vis_downsample_factor) for s in original_size]

        out_path = os.path.join(cfg['prediction_folder'], f'{tomo_name}_{method}.h5')

        if cfg.get('reuse_predictions', True) and os.path.exists(out_path):
            print(f'Skipping {tomo_name} — output already exists')
            continue

        print(f'\nProcessing: {tomo_name}  size: {original_size}'
              + (f'  -> display size: {display_size}' if vis_downsample_factor > 1 else ''))

        padded_size = [padded_size_for_even_tiling(original_size[i], patch_size[i])
                       for i in range(3)]
        pad_transform = SpatialPad(spatial_size=padded_size, method='end', mode='edge')
        img = pad_transform(test_sample['image'][0])
        input_size = list(img[0].shape)

        # ----------------------------------------------------------
        # Step 1 (always): native FPN features -- ALWAYS memory-safe,
        # unaffected by vis_downsample_factor
        # ----------------------------------------------------------
        print('  Extracting FPN features (native resolution)...')
        feat_vol_fpn, stride = extract_fpn_features(
            img=img, search_model=search_model, decoder=decoder,
            patch_size=patch_size, input_size=input_size,
            batch_size=batch_size, patch_iter=patch_iter,
            device=device, upsample=False)

        # ----------------------------------------------------------
        # Step 2 (always): fg detection + PC1 map
        # fits on n_fit_slices central slices unless pca_z_slice is
        # explicitly set in config (single-slice override)
        # ----------------------------------------------------------
        print('  Detecting foreground...')
        pca1_fg, detected_sign, detected_thr, pc1_vol = detect_fg(
            feat_vol_fpn, z_slice, fg_threshold, fg_sign, fg_percentile,
            n_center_slices=n_fit_slices)

        if not filter_foreground:
            detected_thr = -np.inf * detected_sign
            print('  filter_foreground=False (default) -- including every voxel '
                  '(fg_pc1 still saved for reference)')

        pc1_channels = _crop_to_real_region([pc1_vol], stride, display_size)
        pc1_vol_display = pc1_channels[0]
        if list(pc1_vol_display.shape) != list(display_size):
            pc1_vol_display = F.interpolate(
                torch.from_numpy(pc1_vol_display).unsqueeze(0).unsqueeze(0),
                size=list(display_size), mode='trilinear', align_corners=False
            )[0, 0].numpy()

        # ----------------------------------------------------------
        # Step 3: colorization -- all three methods pass display_size
        # (not the raw original_size) as their upsample target, and
        # stride so the padding-crop fix knows where the real region
        # ends at FPN resolution. fg_pc1 (pc1_vol) is always saved
        # alongside the colorization output via _save_nonlinear/
        # _save_pca below -- no separate fg_only mode needed.
        # ----------------------------------------------------------
        if method == 'umap':
            print('  Fitting UMAP...')
            reducer = fit_umap(
                feat_vol_fpn, pca1_fg, detected_sign, detected_thr,
                n_neighbors=n_neighbors, n_fit_samples=n_fit,
                n_slices=n_fit_slices)
            print('  Projecting + upsampling...')
            ch = apply_umap(feat_vol_fpn, pca1_fg, reducer,
                            detected_sign, detected_thr,
                            clip_pct=clip_pct, original_size=display_size,
                            stride=stride)
            _save_nonlinear(out_path, np.stack(ch, axis=-1), pc1_vol_display,
                            display_size, detected_sign, detected_thr,
                            fg_percentile, 'umap')

        elif method == 'cluster':
            print('  Fitting K-means...')
            kmeans = fit_cluster(
                feat_vol_fpn, pca1_fg, detected_sign, detected_thr,
                n_clusters=n_clusters, n_fit_samples=n_fit,
                n_slices=n_fit_slices)
            print('  Assigning cluster colors + upsampling...')
            ch = apply_cluster(feat_vol_fpn, pca1_fg, kmeans,
                               detected_sign, detected_thr,
                               original_size=display_size, stride=stride)
            _save_nonlinear(out_path, np.stack(ch, axis=-1), pc1_vol_display,
                            display_size, detected_sign, detected_thr,
                            fg_percentile, 'cluster')

        else:  # pca
            print('  Fitting PCA(9)...')
            pca2 = fit_pca(feat_vol_fpn, pca1_fg, detected_sign, detected_thr,
                           n_slices=n_fit_slices)
            print('  Projecting...')
            ch_123, ch_456, ch_789 = apply_pca(feat_vol_fpn, pca1_fg, pca2,
                                               detected_sign, detected_thr,
                                               clip_pct=clip_pct, original_size=display_size,
                                               stride=stride)
            _save_pca(out_path, np.stack(ch_123, axis=-1),
                      np.stack(ch_456, axis=-1), np.stack(ch_789, axis=-1),
                      pc1_vol_display, display_size,
                      detected_sign, detected_thr, fg_percentile)


if __name__ == '__main__':
    parser = parser_helper('FPN embedding visualization (PCA / UMAP / Cluster)')
    args = parser.parse_args()
    main(args.config_file, getattr(args, 'filename', None))
