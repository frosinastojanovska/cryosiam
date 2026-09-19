import os
import h5py
import yaml
import torch
import mrcfile
import starfile
import numpy as np
import pandas as pd
import torch.nn.functional as F
from skimage.transform import resize
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt, label as nd_label, maximum_position
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd,
    ScaleIntensityRanged, SpatialPad, EnsureChannelFirstd, EnsureTyped)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, MrcWriter, PatchIter
from cryosiam.apps.prototype_matching import load_search_model, load_decoder
from cryosiam.apps.dense_simsiam_instance import find_markers


def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    search_model = load_search_model(checkpoint_path, device)
    decoder = load_decoder(checkpoint_path, device)
    search_model.eval()
    decoder.eval()
    return search_model, decoder, config


def load_background_prototypes(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']

    proto_key = None
    for key in state:
        if key.endswith('proto_bank.prototypes'):
            proto_key = key
            break

    if proto_key is None:
        raise RuntimeError('No trained prototype bank found in checkpoint.')

    prototypes = state[proto_key]
    bg_prototypes = F.normalize(prototypes[0].float(), dim=-1)

    print(f'Loaded trained background prototypes: K={bg_prototypes.shape[0]}')

    return bg_prototypes


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


def extract_fpn_features(img, search_model, decoder, patch_size,
                         input_size, batch_size, patch_iter, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *patch_size, device=device)
        s_dummy, _ = search_model.get_encoder_features_list(dummy)
        p_dummy = decoder._fpn_body(s_dummy)
        fpn_size = list(p_dummy.shape[-3:])

    out_ch = decoder.out_channels
    stride = [patch_size[i] // fpn_size[i] for i in range(3)]
    fpn_vol_size = [input_size[i] // stride[i] for i in range(3)]
    feat_vol = np.zeros((out_ch, *fpn_vol_size), dtype=np.float32)

    print(f'  FPN stride: {stride}  FPN volume: {fpn_vol_size}')

    loader = DataLoader(GridPatchDataset(data=[img], patch_iter=patch_iter),
                        batch_size=batch_size, num_workers=2)

    with torch.no_grad():
        for item in loader:
            patches = item[0].to(device)
            coords = item[1].numpy().astype(int)
            s_feats, _ = search_model.get_encoder_features_list(patches)
            p = decoder._fpn_body(s_feats)
            p_np = p.cpu().numpy()
            for b in range(patches.shape[0]):
                c = coords[b][1:]
                fc = [[c[i][0] // stride[i], c[i][1] // stride[i]] for i in range(3)]
                ps = list(p.shape[-3:])
                os_ = _stitch_slices(fc, fpn_vol_size, ps)
                sl_ = _patch_slices(fc, fpn_vol_size, ps)
                feat_vol[:, os_[0], os_[1], os_[2]] = p_np[b][:, sl_[0], sl_[1], sl_[2]]

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


def _load_or_extract_feats(tomo_path, feat_path, search_model, decoder,
                           patch_size, batch_size, patch_iter,
                           pad_transform, tomo_transforms, reuse, device):
    if reuse and os.path.exists(feat_path):
        print(f'  Loading cached FPN features')
        with h5py.File(feat_path, 'r') as hf:
            return hf['feat_vol'][()]

    sample = tomo_transforms({'image': tomo_path, 'file_name': tomo_path})
    img = pad_transform(sample['image'])
    input_size = list(img[0].shape)

    print('  Extracting FPN features...')
    with torch.no_grad():
        feat_vol, stride = extract_fpn_features(
            img=img, search_model=search_model, decoder=decoder,
            patch_size=patch_size, input_size=input_size,
            batch_size=batch_size, patch_iter=patch_iter, device=device)

    with h5py.File(feat_path, 'w') as hf:
        hf.create_dataset('feat_vol', data=feat_vol,
                          compression='gzip', compression_opts=4)
        hf.attrs['stride'] = stride
    return feat_vol


def _prepare_voxels_and_mask(feat_vol, mask):
    C_f, D_f, H_f, W_f = feat_vol.shape
    mask_t = torch.from_numpy(
        mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_fpn = F.interpolate(mask_t, size=(D_f, H_f, W_f),
                             mode='nearest')[0, 0].long().numpy()

    voxels = feat_vol.reshape(C_f, -1).T.astype(np.float32)
    norms = np.linalg.norm(voxels, axis=1, keepdims=True)
    voxels = voxels / np.maximum(norms, 1e-8)
    return voxels, mask_fpn


def _extract_instance_prototypes(feat_vol, mask, n_classes, class_names,
                                 min_inst_voxels=5):
    voxels, mask_fpn = _prepare_voxels_and_mask(feat_vol, mask)
    mask_vol = mask_fpn.reshape(feat_vol.shape[1:])

    class_protos = {cls: [] for cls in range(1, n_classes + 1)}

    for cls in range(1, n_classes + 1):
        cls_mask = (mask_vol == cls)
        if not cls_mask.any():
            continue

        labeled, n_inst = nd_label(cls_mask)
        for i in range(1, n_inst + 1):
            inst_flat = labeled.flatten() == i
            inst_vox = voxels[inst_flat]
            if len(inst_vox) < min_inst_voxels:
                continue
            proto = F.normalize(
                torch.from_numpy(
                    inst_vox.mean(axis=0).astype(np.float32)
                ).unsqueeze(0), dim=-1)
            class_protos[cls].append(proto)

    return class_protos


def compute_prototypes(reference_file, reference_mask, file_ext,
                       search_model, decoder, patch_size, batch_size,
                       patch_iter, pad_transform, tomo_transforms,
                       n_classes, class_names, prediction_folder,
                       reuse, device, min_inst_voxels=5):
    is_folder = os.path.isdir(reference_file)
    ref_pairs = []  # list of (feat_vol, mask)

    if is_folder:
        ref_files = sorted(f for f in os.listdir(reference_file)
                           if f.endswith(file_ext))
        print(f'\n{len(ref_files)} reference tomogram(s) (folder mode)')

        for ref_fname in ref_files:
            ref_name = ref_fname.split(file_ext)[0]
            tomo_path = os.path.join(reference_file, ref_fname)
            feat_path = os.path.join(prediction_folder,
                                     f'{ref_name}_ref_feats.h5')
            mask_path = os.path.join(reference_mask, ref_fname) \
                if os.path.isdir(reference_mask) else reference_mask

            if not os.path.exists(mask_path):
                print(f'  WARNING: no mask for {ref_name} — skipping')
                continue

            print(f'\n  Reference: {ref_name}')
            mask = mrcfile.open(mask_path, permissive=True).data.copy()
            mask = mask.astype(np.int32)
            present = [int(c) for c in np.unique(mask) if c > 0]
            print(f'  Mask classes: {present}')

            feat_vol = _load_or_extract_feats(
                tomo_path, feat_path, search_model, decoder,
                patch_size, batch_size, patch_iter,
                pad_transform, tomo_transforms, reuse, device)

            ref_pairs.append((feat_vol, mask))
    else:
        ref_name = os.path.basename(reference_file).split(file_ext)[0]
        feat_path = os.path.join(prediction_folder,
                                 f'{ref_name}_ref_feats.h5')

        print(f'\nReference: {ref_name}  (single file mode)')
        mask = mrcfile.open(reference_mask, permissive=True).data.copy()
        mask = mask.astype(np.int32)
        present = [int(c) for c in np.unique(mask) if c > 0]
        print(f'Mask classes: {present}')

        feat_vol = _load_or_extract_feats(
            reference_file, feat_path, search_model, decoder,
            patch_size, batch_size, patch_iter,
            pad_transform, tomo_transforms, reuse, device)

        ref_pairs.append((feat_vol, mask))

    # accumulate instance prototypes across all reference pairs
    print('\nExtracting instance prototypes...')
    class_protos = {cls: [] for cls in range(1, n_classes + 1)}

    for feat_vol, mask in ref_pairs:
        per_ref = _extract_instance_prototypes(
            feat_vol, mask, n_classes, class_names, min_inst_voxels)
        for cls in range(1, n_classes + 1):
            class_protos[cls].extend(per_ref[cls])

    # finalize — stack per class
    all_prototypes = []
    valid_classes = []

    for cls in range(1, n_classes + 1):
        name = class_names[cls - 1] if cls <= len(class_names) \
            else f'class_{cls}'
        if not class_protos[cls]:
            print(f'  Class {cls} ({name}): no instances found — skipping')
            continue

        protos = torch.cat(class_protos[cls], dim=0)  # (n_inst, C)
        all_prototypes.append(protos)
        valid_classes.append(cls)
        print(f'  Class {cls} ({name}): {len(class_protos[cls])} instances '
              f'→ {protos.shape[0]} prototypes')

    if not all_prototypes:
        raise ValueError('No valid prototypes — check mask labels')

    # log inter-class similarity (closest instance pair)
    _log_interclass_similarity(all_prototypes, valid_classes, class_names)
    return all_prototypes, valid_classes


def _log_interclass_similarity(all_prototypes, valid_classes, class_names):
    if len(valid_classes) < 2:
        return
    print('\n  Inter-class similarity (max over instance pairs):')
    for i, ci in enumerate(valid_classes):
        ni = class_names[ci - 1] if ci <= len(class_names) else f'cls{ci}'
        for j, cj in enumerate(valid_classes):
            if j <= i:
                continue
            nj = class_names[cj - 1] if cj <= len(class_names) else f'cls{cj}'
            sim = (all_prototypes[i] @ all_prototypes[j].T).max().item()
            print(f'    {ni} ↔ {nj}: {sim:.3f}'
                  f'{"  ← may confuse" if sim > 0.9 else ""}')


def compute_similarity_maps(feat_vol_fpn, bg_prototypes, all_prototypes,
                            temperature, original_size=None):
    """
    For each class: similarity = max raw cosine over all instance prototypes.

    Probabilities are softmax(similarity / temperature) over the trained
    background channel plus the reference foreground classes.

    Returns foreground similarity maps and foreground probability maps.
    """
    C, D, H, W = feat_vol_fpn.shape
    voxels = feat_vol_fpn.reshape(C, -1).T.astype(np.float32)
    norms = np.linalg.norm(voxels, axis=1, keepdims=True)
    voxels = voxels / np.maximum(norms, 1e-8)
    voxels_t = torch.from_numpy(voxels)  # (N, C)

    all_sims = []

    bg_protos = F.normalize(bg_prototypes.cpu().float(), dim=1)
    bg_sims = voxels_t @ bg_protos.T
    all_sims.append(bg_sims.max(dim=1).values)

    for protos in all_prototypes:
        Q = F.normalize(protos.cpu().float(), dim=1)
        sims = voxels_t @ Q.T
        sim = sims.max(dim=1).values
        all_sims.append(sim)

    all_sim_maps = torch.stack(all_sims, dim=0).numpy().reshape(
        len(all_prototypes) + 1, D, H, W)

    if original_size is not None and list(original_size) != [D, H, W]:
        ups = []
        for k in range(all_sim_maps.shape[0]):
            up = F.interpolate(
                torch.from_numpy(all_sim_maps[k]).unsqueeze(0).unsqueeze(0),
                size=list(original_size),
                mode='trilinear', align_corners=False)[0, 0].numpy()
            ups.append(up)
        all_sim_maps = np.stack(ups)

    logits = torch.from_numpy(all_sim_maps) / temperature
    all_prob_maps = torch.softmax(logits, dim=0).numpy()

    sim_maps = all_sim_maps[1:]
    prob_maps = all_prob_maps[1:]

    return sim_maps, prob_maps


def load_distances(distances_folder, tomo_name, original_size):
    """Try to load precomputed distances from distances_folder.
    Returns None if distances_folder is not set or file is not found."""
    if not distances_folder:
        return None
    dist_path = os.path.join(distances_folder, f'{tomo_name}_distances.h5')
    if not os.path.isfile(dist_path):
        print(f'  No precomputed distances at {dist_path}, falling back to EDT.')
        return None
    print(f'  Loading precomputed distances from {dist_path}')
    with h5py.File(dist_path, 'r') as f:
        distances = f['distances'][()].astype(np.float32)
    if distances.shape != tuple(original_size):
        distances = resize(distances, original_size, mode='constant',
                           preserve_range=True).astype(np.float32)
    return distances


def extract_centers_per_class(sim_maps, prob_maps, valid_classes, class_names,
                              sim_threshold, dist_threshold, min_voxels, distances=None):
    """
    dist_threshold: float or dict {class_name: float}
    distances: (D, H, W) array or None — if None, falls back to EDT per class
    """
    all_peaks = []
    columns = ['z', 'y', 'x', 'n_voxels',
               'mean_sim', 'sum_sim',
               'mean_prob', 'sum_prob', 'class']

    for k, cls in enumerate(valid_classes):
        name = class_names[cls - 1] if cls <= len(class_names) \
            else f'class_{cls}'
        thr = dist_threshold[name] if isinstance(dist_threshold, dict) \
            else dist_threshold
        sim = sim_maps[k]
        prob = prob_maps[k]
        binary = (sim > sim_threshold)

        if not binary.any():
            print(f'  {name}: no voxels above threshold')
            all_peaks.append(pd.DataFrame(columns=columns))
            continue

        if thr > 0:
            if distances is not None:
                dist = distances * binary.astype(np.float32)
            else:
                print(f'  {name}: computing EDT...')
                dist = distance_transform_edt(binary).astype(np.float32)

            markers, n_instances = nd_label(dist >= thr)

            if n_instances > 0:
                instances = watershed(-dist, markers, mask=binary)
            else:
                instances = markers
        else:
            markers, n_instances = nd_label(binary)
            instances = markers

            if distances is not None:
                dist = distances * binary.astype(np.float32)
            else:
                dist = distance_transform_edt(binary).astype(np.float32)

        if n_instances == 0:
            print(f'  {name}: no markers found')
            all_peaks.append(pd.DataFrame(columns=columns))
            continue

        instance_ids = list(range(1, n_instances + 1))
        peaks = maximum_position(dist, labels=markers, index=instance_ids)
        if isinstance(peaks, tuple):
            peaks = [peaks]

        rows = []
        for instance_id, peak_idx in zip(instance_ids, peaks):
            instance_mask = (instances == instance_id)
            n_voxels = int(instance_mask.sum())

            if n_voxels <= min_voxels:
                continue

            rows.append({
                'z': float(peak_idx[0]),
                'y': float(peak_idx[1]),
                'x': float(peak_idx[2]),
                'n_voxels': n_voxels,
                'mean_sim': float(sim[instance_mask].mean()),
                'sum_sim': float(sim[instance_mask].sum()),
                'mean_prob': float(prob[instance_mask].mean()),
                'sum_prob': float(prob[instance_mask].sum()),
                'class': name})

        df = pd.DataFrame(rows, columns=columns)
        all_peaks.append(df)
        print(f'  {name}: {len(df)} particles  '
              f'(instances={n_instances}  dist_threshold={thr})')

    return all_peaks


def main(config_file_path, filename=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = cfg['trained_model']
    search_model, decoder, config = load_models(checkpoint_path, device)
    bg_prototypes = load_background_prototypes(checkpoint_path)

    data_cfg = config['parameters']['data']
    patch_size = data_cfg['patch_size']
    file_ext = cfg['file_extension']
    batch_size = cfg['hyper_parameters']['batch_size']

    n_classes = int(cfg['n_classes'])
    class_names = cfg.get('class_names',
                          [f'class_{i}' for i in range(1, n_classes + 1)])
    temperature = float(config['parameters']['network'].get('temperature', 0.1))

    sim_threshold = float(cfg.get('sim_threshold', 0.5))
    dist_threshold = cfg.get('dist_threshold', 3.0)
    if not isinstance(dist_threshold, dict):
        dist_threshold = float(dist_threshold)
    min_inst_voxels = int(cfg.get('min_inst_voxels', 5))
    min_voxels = int(cfg.get('min_voxels', 50))
    reuse = bool(cfg.get('reuse_predictions', True))
    distances_folder = cfg.get('distances_folder') or None
    save_individual_centers = bool(cfg.get('save_individual_centers', False))

    os.makedirs(cfg['prediction_folder'], exist_ok=True)
    output_folder = cfg.get('output_folder', cfg['prediction_folder'])
    os.makedirs(output_folder, exist_ok=True)

    tomo_transforms = build_tomo_transforms(data_cfg)
    patch_iter = PatchIter(patch_size=tuple(patch_size),
                           start_pos=(0, 0, 0),
                           overlap=(0, 0.5, 0.5, 0.5))
    pad_transform = SpatialPad(spatial_size=patch_size,
                               method='end', mode='edge')
    mask_writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    mask_writer.set_metadata({'voxel_size': 1})

    rename_map = {
        'z': 'rlnCoordinateZ',
        'y': 'rlnCoordinateY',
        'x': 'rlnCoordinateX',
        'n_voxels': 'rlnNVoxels',
        'mean_sim': 'rlnMeanSim',
        'sum_sim': 'rlnSumSim',
        'mean_prob': 'rlnMeanProb',
        'sum_prob': 'rlnSumProb',
        'class': 'rlnClassLabel',
        'tomo': 'rlnMicrographName'}

    all_prototypes, valid_classes = compute_prototypes(
        reference_file=cfg['reference_file'],
        reference_mask=cfg['reference_mask'],
        file_ext=file_ext,
        search_model=search_model,
        decoder=decoder,
        patch_size=patch_size,
        batch_size=batch_size,
        patch_iter=patch_iter,
        pad_transform=pad_transform,
        tomo_transforms=tomo_transforms,
        n_classes=n_classes,
        class_names=class_names,
        prediction_folder=cfg['prediction_folder'],
        reuse=reuse,
        device=device,
        min_inst_voxels=min_inst_voxels)

    valid_names = [class_names[c - 1] for c in valid_classes]
    n_inst_per_class = [all_prototypes[k].shape[0]
                        for k in range(len(valid_classes))]
    print(f'\nPrototypes ready: '
          f'{dict(zip(valid_names, n_inst_per_class))} instances')
    print(f'Applying to tomograms in {cfg["data_folder"]}')

    data_folder = cfg['data_folder']
    files = cfg.get('test_files') or [
        f for f in os.listdir(data_folder)
        if os.path.isfile(os.path.join(data_folder, f))
           and f.endswith(file_ext)]
    if filename:
        files = [filename]

    test_data = [{'image': os.path.join(data_folder, f),
                  'file_name': os.path.join(data_folder, f)}
                 for f in files]
    test_ds = Dataset(data=test_data, transform=tomo_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1,
                             collate_fn=list_data_collate)

    all_combined = []
    print(f'\n{len(files)} test tomogram(s)')

    with torch.no_grad():
        for test_sample in test_loader:
            current_file = test_sample['file_name'][0]
            tomo_name = os.path.basename(current_file).split(file_ext)[0]
            original_size = list(test_sample['image'][0][0].shape)
            feat_path = os.path.join(cfg['prediction_folder'],
                                     f'{tomo_name}_feats.h5')

            print(f'\n{"=" * 55}')
            print(f'Tomogram: {tomo_name}  size: {original_size}')

            feat_vol_fpn = _load_or_extract_feats(
                current_file, feat_path, search_model, decoder,
                patch_size, batch_size, patch_iter,
                pad_transform, tomo_transforms, reuse, device)

            print('  Computing similarity maps and probabilities...')
            sim_maps, prob_maps = compute_similarity_maps(
                feat_vol_fpn, bg_prototypes, all_prototypes,
                temperature, original_size=original_size)

            # save similarity and probability maps
            h5_out = os.path.join(cfg['prediction_folder'],
                                  f'{tomo_name}_simmaps.h5')
            with h5py.File(h5_out, 'w') as hf:
                for k, cls in enumerate(valid_classes):
                    name = class_names[cls - 1] if cls <= len(class_names) \
                        else f'class_{cls}'
                    hf.create_dataset(f'sim_{name}', data=sim_maps[k],
                                      compression='gzip', compression_opts=4)
                    hf.create_dataset(f'prob_{name}', data=prob_maps[k],
                                      compression='gzip', compression_opts=4)
                hf.attrs['class_names'] = valid_names
                hf.attrs['valid_classes'] = valid_classes
                hf.attrs['sim_threshold'] = sim_threshold
                hf.attrs['n_instances'] = n_inst_per_class
                hf.attrs['temperature'] = temperature
                hf.attrs['similarity_type'] = 'raw_cosine'
            print(f'  Saved: {os.path.basename(h5_out)}')

            if cfg.get('save_sim_mrc', True):
                for k, cls in enumerate(valid_classes):
                    name = class_names[cls - 1] if cls <= len(class_names) \
                        else f'class_{cls}'
                    mrc_out = os.path.join(output_folder,
                                           f'{tomo_name}_{name}_sim.mrc')
                    mask_writer.set_data_array(sim_maps[k], channel_dim=None)
                    mask_writer.write(mrc_out)

            if cfg.get('save_prob_mrc', False):
                for k, cls in enumerate(valid_classes):
                    name = class_names[cls - 1] if cls <= len(class_names) \
                        else f'class_{cls}'
                    mrc_out = os.path.join(output_folder,
                                           f'{tomo_name}_{name}_prob.mrc')
                    mask_writer.set_data_array(prob_maps[k], channel_dim=None)
                    mask_writer.write(mrc_out)

            # load precomputed distances or fall back to EDT inside extract_centers
            distances = load_distances(distances_folder, tomo_name, original_size)

            print('  Extracting centers...')
            peaks_per_class = extract_centers_per_class(
                sim_maps, prob_maps, valid_classes, class_names,
                sim_threshold, dist_threshold, min_voxels, distances=distances)

            tomo_combined = []
            for k, (cls, peaks) in enumerate(
                    zip(valid_classes, peaks_per_class)):
                if len(peaks) == 0:
                    continue
                name = class_names[cls - 1] if cls <= len(class_names) \
                    else f'class_{cls}'
                peaks['tomo'] = tomo_name

                if save_individual_centers:
                    starfile.write(
                        peaks.rename(columns=rename_map, errors='ignore'),
                        os.path.join(output_folder,
                                     f'{tomo_name}_{name}_centers.star'),
                        overwrite=True)

                tomo_combined.append(peaks)

            if tomo_combined:
                combined = pd.concat(tomo_combined, ignore_index=True)
                combined['tomo'] = tomo_name
                starfile.write(
                    combined.rename(columns=rename_map, errors='ignore'),
                    os.path.join(output_folder,
                                 f'{tomo_name}_all_centers.star'),
                    overwrite=True)
                print(f'  {len(combined)} total particles')
                all_combined.append(combined)

    if all_combined:
        final = pd.concat(all_combined, ignore_index=True)
        out_all = os.path.join(output_folder, 'all_tomograms_centers.star')
        starfile.write(
            final.rename(columns=rename_map, errors='ignore'),
            out_all, overwrite=True)
        print(f'\nDone. {len(final)} particles across '
              f'{len(all_combined)} tomogram(s) → {out_all}')
    else:
        print('\nNo particles found.')


if __name__ == '__main__':
    parser = parser_helper(
        'Instance prototype-based multi-class particle picking')
    args = parser.parse_args()
    main(args.config_file, getattr(args, 'filename', None))
