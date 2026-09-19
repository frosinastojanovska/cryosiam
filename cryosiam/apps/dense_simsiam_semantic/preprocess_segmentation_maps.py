import os
import edt
import yaml
import numpy as np
import scipy.ndimage as ndi

from cryosiam.data import MrcReader
from cryosiam.utils import parser_helper


def generate_distance_map(semantic_segmentation, num_classes=1):
    if num_classes == 1:
        return edt.sdf(semantic_segmentation > 0, black_border=True, parallel=1)[None]
    dist = np.zeros((num_classes,) + semantic_segmentation.shape)
    for i in range(num_classes):
        dist[i] = edt.sdf(semantic_segmentation == i, black_border=True, parallel=1)
    return dist


def generate_tubed_skeleton(labels, distance_map, num_classes, tube_fraction=0.5, window=11):
    labels = labels.astype(np.int32)
    tube = np.zeros((num_classes,) + labels.shape, dtype=np.uint8)

    for c in range(1, num_classes) if num_classes > 1 else [1]:
        mask = labels == c if num_classes > 1 else labels > 0
        if not mask.any():
            continue
        d = distance_map[0] if num_classes == 1 else distance_map[c]
        local_max = ndi.maximum_filter(d, size=window)
        tube[0 if num_classes == 1 else c] = mask & (d >= tube_fraction * local_max)

    return tube


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    reader = MrcReader(read_in_mem=True)

    data_cfg = cfg['parameters']['data']
    use_distances = data_cfg.get('use_distances', True)
    use_skeletons = data_cfg.get('use_skeletons', False)
    tube_fraction = data_cfg.get('skeleton_tube_fraction', 0.5)

    labels_folder = cfg['labels_folder']
    temp_dir = cfg['temp_dir']
    out_channels = cfg['parameters']['network']['out_channels']

    if use_distances:
        os.makedirs(os.path.join(temp_dir, 'distances'), exist_ok=True)
    if use_skeletons:
        os.makedirs(os.path.join(temp_dir, 'skeletons'), exist_ok=True)

    files = [x for x in os.listdir(labels_folder) if x.endswith(cfg['file_extension'])]
    if cfg['train_files'] is not None:
        if cfg['val_files'] is None:
            cfg['val_files'] = []
        files = [x for x in files if x in cfg['train_files'] or x in cfg['val_files']]

    for file in files:
        print(f'Processing tomo {file}')
        root_file_name = file.split(cfg['file_extension'])[0]
        labels = reader.read(os.path.join(labels_folder, f'{root_file_name}{cfg["file_extension"]}'))
        labels = labels.data
        labels.setflags(write=True)

        distance_map = generate_distance_map(labels, num_classes=out_channels).astype(np.float32)

        if use_distances:
            np.savez_compressed(os.path.join(temp_dir, 'distances', f'{root_file_name}.npz'), data=distance_map)

        if use_skeletons:
            skeleton = generate_tubed_skeleton(labels, distance_map, num_classes=out_channels,
                                               tube_fraction=tube_fraction)
            np.savez_compressed(os.path.join(temp_dir, 'skeletons', f'{root_file_name}.npz'), data=skeleton)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
