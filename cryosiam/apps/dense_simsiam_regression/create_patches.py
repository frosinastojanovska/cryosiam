import os
import yaml
import numpy as np
from monai.data.utils import iter_patch_slices

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer.set_metadata({'voxel_size': 1})
    writer_gt = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer_gt.set_metadata({'voxel_size': 1})
    reader = MrcReader(read_in_mem=True)
    gt_reader = MrcReader(read_in_mem=True)

    data_folder = cfg['data_folder']
    gt_folder = cfg['gt_folder']
    patches_folder = cfg['patches_folder']
    patch_size = cfg['parameters']['data']['patch_size']
    overlap = cfg['parameters']['data']['patch_overlap']

    os.makedirs(os.path.join(patches_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'gt'), exist_ok=True)
    files = [x for x in os.listdir(gt_folder) if x.endswith(cfg['file_extension'])]
    if cfg['train_files'] is not None:
        if cfg['val_files'] is None:
            cfg['val_files'] = []
        files = [x for x in files if x in cfg['train_files'] or x in cfg['val_files']]
    for file in files:
        print(f'Processing tomo {file}')
        file_path = os.path.join(data_folder, file)

        image = reader.read(file_path)
        image = image.data
        image.setflags(write=True)

        root_file_name = file.split(cfg['file_extension'])[0]
        gt = gt_reader.read(os.path.join(gt_folder, f'{root_file_name}{cfg["file_extension"]}'))
        gt = gt.data
        gt.setflags(write=True)

        if len(patch_size) == 2:
            start_pos = (0, 0)
        else:
            start_pos = (0, 0, 0)
        iter_size = image.shape
        for slices in iter_patch_slices(iter_size, patch_size, start_pos, overlap, padded=False):
            coords = tuple((coord.start, coord.stop) for coord in slices)
            coords_array = np.asarray(coords)
            patch = image[slices].astype(np.float32)
            gt_patch = gt[slices].astype(np.float32)

            if 'remove_empty_patches' in cfg['parameters']['data'] and cfg['parameters']['data'][
                'remove_empty_patches']:
                if np.sum(gt_patch) == 0:
                    continue
            if len(patch_size) == 2:
                y, x = coords_array[0][0], coords_array[1][0]
                patch_path = f'{file.split(cfg["file_extension"])[0]}_y{y}_x{x}{cfg["file_extension"]}'
            else:
                z, y, x = coords_array[0][0], coords_array[1][0], coords_array[2][0]
                patch_path = f'{file.split(cfg["file_extension"])[0]}_z{z}_y{y}_x{x}{cfg["file_extension"]}'

            subtomo_path = os.path.join(patches_folder, 'image', patch_path)
            # save the subtomograms
            writer.set_data_array(patch, channel_dim=None)
            writer.write(subtomo_path)

            subtomo_gt_path = os.path.join(patches_folder, 'gt', patch_path)
            # save the gt
            writer_gt.set_data_array(gt_patch, channel_dim=None)
            writer_gt.write(subtomo_gt_path)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
