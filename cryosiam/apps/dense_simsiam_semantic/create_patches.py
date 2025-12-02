import os
import yaml
import numpy as np
from monai.data import NumpyReader
from monai.data.utils import iter_patch_slices

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    writer = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer.set_metadata({'voxel_size': 1})
    writer_labels = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer_labels.set_metadata({'voxel_size': 1})
    reader = MrcReader(read_in_mem=True)
    labels_reader = MrcReader(read_in_mem=True)

    distance_reader = NumpyReader(npz_keys='data', channel_dim=0)

    data_folder = cfg['data_folder']
    noisy_data_folder = cfg['noisy_data_folder'] if 'noisy_data_folder' in cfg and cfg['noisy_data_folder'] else None
    labels_folder = cfg['labels_folder']
    patches_folder = cfg['patches_folder']
    temp_dir = cfg['temp_dir']
    patch_size = cfg['parameters']['data']['patch_size']
    overlap = cfg['parameters']['data']['patch_overlap']

    os.makedirs(os.path.join(patches_folder, 'image'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(patches_folder, 'distances'), exist_ok=True)
    if noisy_data_folder:
        os.makedirs(os.path.join(patches_folder, 'noisy_image'), exist_ok=True)
    files = [x for x in os.listdir(labels_folder) if x.endswith(cfg['file_extension'])]
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
        labels = labels_reader.read(os.path.join(labels_folder, f'{root_file_name}{cfg["file_extension"]}'))
        distances = distance_reader.read(os.path.join(temp_dir, 'distances', f'{root_file_name}.npz'))
        labels = labels.data
        labels.setflags(write=True)

        if len(patch_size) == 2:
            start_pos = (0, 0)
        else:
            start_pos = (0, 0, 0)
        iter_size = image.shape
        dist_shape = distances.shape[0]
        for slices in iter_patch_slices(iter_size, patch_size, start_pos, overlap, padded=False):
            coords = tuple((coord.start, coord.stop) for coord in slices)
            coords_array = np.asarray(coords)
            patch = image[slices].astype(np.float32)
            label_patch = labels[slices].astype(np.float32)
            distance_patch = distances[(slice(dist_shape),) + slices].astype(np.float32)
            # if np.sum(label_patch) == 0:
            #     continue
            if len(patch_size) == 2:
                y, x = coords_array[0][0], coords_array[1][0]
                patch_path = f'{file.split(cfg["file_extension"])[0]}_y{y}_x{x}{cfg["file_extension"]}'
                distance_patch_path = f'{file.split(cfg["file_extension"])[0]}_y{y}_x{x}.npz'
            else:
                z, y, x = coords_array[0][0], coords_array[1][0], coords_array[2][0]
                patch_path = f'{file.split(cfg["file_extension"])[0]}_z{z}_y{y}_x{x}{cfg["file_extension"]}'
                distance_patch_path = f'{file.split(cfg["file_extension"])[0]}_z{z}_y{y}_x{x}.npz'

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

            subtomo_labels_path = os.path.join(patches_folder, 'labels', patch_path)
            # save the labels
            writer_labels.set_data_array(label_patch, channel_dim=None)
            writer_labels.write(subtomo_labels_path)

            # save the distances
            np.savez_compressed(os.path.join(patches_folder, 'distances', distance_patch_path), data=distance_patch)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
