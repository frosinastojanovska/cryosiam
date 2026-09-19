import os
import json
import yaml
import numpy as np
from skimage.segmentation import expand_labels
from monai.data.utils import iter_patch_slices

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter


def main(config_file_path: str):
    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    reader = MrcReader(read_in_mem=True)
    image_writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    image_writer.set_metadata({'voxel_size': 1})
    mask_writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    mask_writer.set_metadata({'voxel_size': 1})

    data_folder = cfg['data_folder']
    masks_folder = cfg['masks_folder']
    patches_folder = cfg['patches_folder']
    patch_size = cfg['parameters']['data']['patch_size']
    overlap = cfg['parameters']['data']['patch_overlap']
    expand_vox = cfg['parameters']['data'].get('expand_vox', 0)
    min_foreground = cfg.get('min_foreground_voxels', 0)
    file_ext = cfg['file_extension']

    target_class = cfg['parameters']['data'].get('target_class', None)
    if target_class is not None:
        target_classes = ([target_class] if isinstance(target_class, int)
                          else list(target_class))
        print(f'Target class(es): {target_classes} → binarized to 1')

    os.makedirs(os.path.join(patches_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'masks'), exist_ok=True)

    files = [x for x in os.listdir(data_folder) if x.endswith(file_ext)]
    if cfg.get('train_files') is not None:
        val_files = cfg.get('val_files') or []
        files = [x for x in files
                 if x in cfg['train_files'] or x in val_files]

    total_saved = 0
    total_skipped = 0
    manifest = {}

    for file in files:
        root = file.split(file_ext)[0]
        print(f'\nProcessing {file}')

        image = reader.read(os.path.join(data_folder, file)).data
        image.setflags(write=True)

        mask_file = os.path.join(masks_folder, file)
        if not os.path.exists(mask_file):
            print(f'  WARNING: no semantic mask found, skipping')
            continue

        mask = reader.read(mask_file).data.astype(np.int32)
        mask.setflags(write=True)

        if target_class is not None:
            mask = np.isin(mask, target_classes).astype(np.int32)

        if expand_vox > 0:
            mask = expand_labels(mask, expand_vox)

        if image.shape != mask.shape:
            print(f'  WARNING: shape mismatch image={image.shape} '
                  f'mask={mask.shape}, skipping')
            continue

        for slices in iter_patch_slices(
                image.shape, patch_size, (0, 0, 0), overlap, padded=False):

            mask_patch = mask[slices]

            if int(np.sum(mask_patch > 0)) < min_foreground:
                total_skipped += 1
                continue

            coords = np.array([(s.start, s.stop) for s in slices])
            z, y, x = coords[0, 0], coords[1, 0], coords[2, 0]
            base_name = f'{root}_z{z}_y{y}_x{x}'

            manifest[base_name] = [
                int(c) for c in np.unique(mask_patch) if c != 0]

            image_writer.set_data_array(
                image[slices].astype(np.float32), channel_dim=None)
            image_writer.write(
                os.path.join(patches_folder, 'images',
                             f'{base_name}{file_ext}'))

            mask_writer.set_data_array(
                mask_patch.astype(np.float32), channel_dim=None)
            mask_writer.write(
                os.path.join(patches_folder, 'masks',
                             f'{base_name}{file_ext}'))

            total_saved += 1

    print(f'\nDone: {total_saved} patches saved, '
          f'{total_skipped} skipped (< {min_foreground} foreground voxels)')

    manifest_path = os.path.join(patches_folder, 'class_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    print(f'Class manifest → {manifest_path}  ({len(manifest)} entries)')


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
