import os
import yaml
import edt
import numpy as np
from datetime import datetime
from scipy.ndimage import label as nd_label
from skimage.segmentation import expand_labels
from monai.data.utils import iter_patch_slices

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter


def save_metadata(log_dir, class_names, n_classes, use_distances, cfg):
    os.makedirs(log_dir, exist_ok=True)
    metadata = {
        'n_classes': n_classes,
        'class_names': class_names,
        'use_distances': use_distances,
        'patch_size': cfg['parameters']['data']['patch_size'],
        'created': datetime.now().isoformat(),
    }
    path = os.path.join(log_dir, 'metadata.yaml')
    with open(path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f'Saved metadata → {path}')


def build_class_mapping(class_names, original_class_indices=None):
    n = len(class_names)
    if original_class_indices is None:
        return {i: i for i in range(1, n + 1)}
    if len(original_class_indices) != n:
        raise ValueError(
            f'original_class_indices length ({len(original_class_indices)}) '
            f'must match class_names ({n})')
    return {orig: new for new, orig in enumerate(original_class_indices, start=1)}


def remap_mask(mask, orig_to_new):
    remapped = np.zeros_like(mask)
    for orig, new in orig_to_new.items():
        remapped[mask == orig] = new
    return remapped


def generate_distance_map(semantic_segmentation, num_classes=1):
    """Per-class signed distance field, matching the specialized module's
    create_patches script exactly -- same edt.sdf call, same convention."""
    dist = np.zeros((num_classes,) + semantic_segmentation.shape)
    for i in range(num_classes):
        dist[i] = edt.sdf(semantic_segmentation == i + 1, black_border=True, parallel=1)
    return dist.astype(np.float32)


def remove_small_semantic_components(mask, min_voxels):
    if min_voxels <= 0:
        return mask
    cleaned = mask.copy()
    for cls in np.unique(mask[mask > 0]):
        labeled, _ = nd_label(mask == cls)
        sizes = np.bincount(labeled.ravel())
        valid = np.where(sizes[1:] >= min_voxels)[0] + 1
        cleaned[np.isin(labeled, valid, invert=True) & (mask == cls)] = 0
    removed = int((mask > 0).sum() - (cleaned > 0).sum())
    if removed:
        print(f'  Removed {removed:,} voxels in small components')
    return cleaned


def expand_semantic_mask(mask, expand_voxels):
    if expand_voxels <= 0:
        return mask
    expanded = expand_labels(mask, distance=expand_voxels).astype(mask.dtype)
    print(f'  Expanded labels by {expand_voxels} voxels')
    return expanded


def build_semantic_training_mask(mask, min_instance_voxels, expand_voxels):
    mask = remove_small_semantic_components(mask, min_instance_voxels)
    mask = expand_semantic_mask(mask, expand_voxels)
    return mask


def estimate_instance_sizes(mask, class_names):
    sizes = {}
    D, H, W = mask.shape
    for cls_idx, name in enumerate(class_names):
        cls = cls_idx + 1
        binary = (mask == cls)
        if not binary.any():
            sizes[name] = 0.0
            print(f'  {name}: median instance size = 0 voxels (0 interior instances)')
            continue
        labeled, n = nd_label(binary)
        voxel_counts = []
        for i in range(1, n + 1):
            inst = (labeled == i)
            zz, yy, xx = np.where(inst)
            if (zz.min() == 0 or zz.max() == D - 1 or
                    yy.min() == 0 or yy.max() == H - 1 or
                    xx.min() == 0 or xx.max() == W - 1):
                continue
            voxel_counts.append(inst.sum())
        sizes[name] = float(np.median(voxel_counts)) if voxel_counts else 0.0
        print(f'  {name}: median instance size = {sizes[name]:.0f} voxels '
              f'({len(voxel_counts)} interior instances)')
    return sizes


def remove_small_boundary_instances(mask_patch, class_names, instance_sizes,
                                    min_fraction=0.2):
    cleaned = mask_patch.copy()
    for cls_idx, name in enumerate(class_names):
        cls = cls_idx + 1
        binary = (mask_patch == cls)
        if not binary.any():
            continue
        expected = instance_sizes.get(name, 0.0)
        labeled, n = nd_label(binary)
        for i in range(1, n + 1):
            inst = (labeled == i)
            touches = (inst[0].any() or inst[-1].any() or
                       inst[:, 0].any() or inst[:, -1].any() or
                       inst[:, :, 0].any() or inst[:, :, -1].any())
            if touches and inst.sum() < min_fraction * expected:
                cleaned[inst] = 0
    return cleaned


def _save_patch(slices, image, mask, dist_vol, root, file_ext, patches_folder,
                img_writer, mask_writer, class_names, instance_sizes, min_fraction,
                use_distances):
    z0, y0, x0 = slices[0].start, slices[1].start, slices[2].start
    patch_name = f'{root}_z{z0}_y{y0}_x{x0}'

    mask_patch = remove_small_boundary_instances(mask[slices].copy(), class_names,
                                                 instance_sizes, min_fraction)

    img_writer.set_data_array(image[slices], channel_dim=None)
    img_writer.write(os.path.join(patches_folder, 'images', f'{patch_name}{file_ext}'))

    mask_writer.set_data_array(mask_patch, channel_dim=None)
    mask_writer.write(os.path.join(patches_folder, 'masks', f'{patch_name}{file_ext}'))

    if use_distances and dist_vol is not None:
        dist_patch = dist_vol[(slice(None),) + slices].copy()
        np.savez_compressed(
            os.path.join(patches_folder, 'distances', f'{patch_name}.npz'),
            data=dist_patch.astype(np.float32))


def main(config_file_path):
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    data_folder = cfg['data_folder']
    labels_folder = cfg['labels_folder']
    patches_folder = cfg['patches_folder']
    log_dir = cfg['log_dir']
    file_ext = cfg['file_extension']

    patch_size = cfg['parameters']['data']['patch_size']
    overlap = cfg['parameters']['data'].get('patch_overlap', 0)
    bg_keep_fraction = float(cfg['parameters']['data'].get('background_fraction', 0.5))
    expand_voxels = int(cfg.get('expand_labels', 1))
    min_instance_voxels = int(cfg.get('min_instance_voxels', 0))
    min_fraction = float(cfg.get('min_boundary_fraction', 0.2))
    use_distances = bool(cfg.get('use_distances', False))

    class_names = cfg['class_names']
    original_class_indices = cfg.get('original_class_indices', None)
    n_classes = len(class_names)
    orig_to_new = build_class_mapping(class_names, original_class_indices)

    rng = np.random.default_rng(int(cfg.get('random_seed', 42)))

    save_metadata(log_dir, class_names, n_classes, use_distances, cfg)

    os.makedirs(os.path.join(patches_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'masks'), exist_ok=True)
    if use_distances:
        os.makedirs(os.path.join(patches_folder, 'distances'), exist_ok=True)

    reader = MrcReader(read_in_mem=True)
    img_writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    img_writer.set_metadata({'voxel_size': 1})
    mask_writer = MrcWriter(output_dtype=np.int16, overwrite=True)
    mask_writer.set_metadata({'voxel_size': 1})

    files = [f for f in os.listdir(data_folder) if f.endswith(file_ext)]
    allowed = set(cfg.get('train_files') or []) | set(cfg.get('val_files') or [])
    if allowed:
        files = [f for f in files if f in allowed]

    total_patches = 0
    total_fg = 0
    total_bg = 0

    for file in sorted(files):
        root = file.split(file_ext)[0]
        print(f'\nProcessing: {file}')

        image = reader.read(os.path.join(data_folder, file)).data.astype(np.float32)
        image.setflags(write=True)

        mask_path = os.path.join(labels_folder, file)
        if not os.path.exists(mask_path):
            print(f'  WARNING: semantic mask not found — skipping')
            continue
        mask = reader.read(mask_path).data.astype(np.int16)
        mask.setflags(write=True)
        mask = remap_mask(mask, orig_to_new)
        print(f'  Classes: {np.unique(mask[mask > 0]).tolist()}')

        training_mask = build_semantic_training_mask(mask, min_instance_voxels, expand_voxels)

        dist_vol = None
        if use_distances:
            dist_vol = generate_distance_map(training_mask, num_classes=n_classes)

        instance_sizes = estimate_instance_sizes(training_mask, class_names)

        fg_slices, bg_slices = [], []
        for slices in iter_patch_slices(image.shape, patch_size, (0,) * len(patch_size),
                                        overlap, padded=False):
            (fg_slices if training_mask[slices].max() > 0 else bg_slices).append(slices)

        n_bg_target = int(round(bg_keep_fraction * len(fg_slices)))
        bg_selected = []
        if n_bg_target > 0 and bg_slices:
            bg_idx = rng.choice(len(bg_slices), min(n_bg_target, len(bg_slices)),
                                replace=False)
            bg_selected = [bg_slices[i] for i in bg_idx]

        print(f'  Patches: {len(fg_slices)} fg, {len(bg_selected)} bg '
              f'(of {len(bg_slices)} available)')

        for slices in fg_slices + bg_selected:
            _save_patch(slices, image, training_mask, dist_vol, root, file_ext, patches_folder,
                        img_writer, mask_writer, class_names, instance_sizes, min_fraction,
                        use_distances)
            total_patches += 1

        total_fg += len(fg_slices)
        total_bg += len(bg_selected)

    print(f'\nDone. {total_patches} patches ({total_fg} fg + {total_bg} bg)')


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
