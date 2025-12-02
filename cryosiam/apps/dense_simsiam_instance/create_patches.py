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
    int_writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    int_writer.set_metadata({'voxel_size': 1})
    reader = MrcReader(read_in_mem=True)

    data_folder = cfg['data_folder']
    noisy_data_folder = cfg['noisy_data_folder'] if 'noisy_data_folder' in cfg and cfg['noisy_data_folder'] else None
    temp_dir = cfg['temp_dir']
    patches_folder = cfg['patches_folder']
    patch_size = cfg['parameters']['data']['patch_size']
    overlap = cfg['parameters']['data']['patch_overlap']

    os.makedirs(os.path.join(patches_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'foreground'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'distances'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'boundaries'), exist_ok=True)
    if noisy_data_folder:
        os.makedirs(os.path.join(patches_folder, 'noisy_image'), exist_ok=True)
    files = [x for x in os.listdir(data_folder) if x.endswith(cfg['file_extension'])]
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

        if noisy_data_folder:
            noisy_image = reader.read(os.path.join(noisy_data_folder, file))
            noisy_image = noisy_image.data
            noisy_image.setflags(write=True)

        root_file_name = file.split(cfg['file_extension'])[0]
        foreground = reader.read(os.path.join(temp_dir, 'foreground', f'{root_file_name}{cfg["file_extension"]}'))
        distances = reader.read(os.path.join(temp_dir, 'distances', f'{root_file_name}{cfg["file_extension"]}'))
        boundaries = reader.read(os.path.join(temp_dir, 'boundaries', f'{root_file_name}{cfg["file_extension"]}'))
        foreground = foreground.data
        foreground.setflags(write=True)
        distances = distances.data
        distances.setflags(write=True)
        boundaries = boundaries.data
        boundaries.setflags(write=True)

        if len(patch_size) == 2:
            start_pos = (0, 0)
        else:
            start_pos = (0, 0, 0)
        iter_size = image.shape
        for slices in iter_patch_slices(iter_size, patch_size, start_pos, overlap, padded=False):
            coords = tuple((coord.start, coord.stop) for coord in slices)
            coords_array = np.asarray(coords)
            patch = image[slices].astype(np.float32)
            foreground_patch = foreground[slices].astype(np.int8)
            distances_patch = distances[slices].astype(np.float32)
            boundaries_patch = boundaries[slices].astype(np.int8)
            if np.sum(foreground_patch) == 0:
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

            if noisy_data_folder:
                noisy_patch = noisy_image[slices].astype(np.float32)
                subtomo_path = os.path.join(patches_folder, 'noisy_image', patch_path)
                # save the subtomograms
                writer.set_data_array(noisy_patch, channel_dim=None)
                writer.write(subtomo_path)

            subtomo_out_path = os.path.join(patches_folder, 'foreground', patch_path)
            int_writer.set_data_array(foreground_patch, channel_dim=None)
            int_writer.write(subtomo_out_path)

            subtomo_out_path = os.path.join(patches_folder, 'distances', patch_path)
            # save the labels
            writer.set_data_array(distances_patch, channel_dim=None)
            writer.write(subtomo_out_path)

            subtomo_out_path = os.path.join(patches_folder, 'boundaries', patch_path)
            # save the labels
            int_writer.set_data_array(boundaries_patch, channel_dim=None)
            int_writer.write(subtomo_out_path)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
