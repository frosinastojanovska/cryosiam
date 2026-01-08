import os
import h5py
import yaml
import mrcfile
import argparse
import starfile
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table, label
from skimage.morphology import binary_erosion


def parser_helper(description=None):
    description = "Plot instance segmentation with napari" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True,
                        help='path to the config file used for running CryoSiam')
    parser.add_argument('--filename', type=str, required=True,
                        help='Tomogram filename (including the file extension')
    return parser


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    files = [file for file in os.listdir(cfg['prediction_folder']) if file.endswith('.h5')]

    classes_range = cfg['parameters']['network']['out_channels'] if cfg['parameters']['network'][
                                                                        'out_channels'] > 1 else 2
    labels = {i: [] for i in range(1, classes_range)}
    if 'min_voxels' in cfg['parameters']['network']:
        min_voxels = cfg['parameters']['network']['min_voxels']
        if not isinstance(min_voxels, list):
            min_voxels = [min_voxels] * classes_range
    else:
        min_voxels = [100] * classes_range
    for label_id in range(1, classes_range):
        for file in files:
            basename = os.path.basename(file).split('_preds.h5')[0]
            print('Processing {}'.format(basename))
            with h5py.File(os.path.join(cfg['prediction_folder'], file), 'r') as f:
                data = f['labels'][()]
            instances = label(binary_erosion(data == label_id))
            regions = regionprops_table(instances, properties=['label', 'area', 'centroid'])
            regions = pd.DataFrame(regions)
            regions = regions[regions['area'] > min_voxels[label_id - 1]]
            regions['tomo'] = basename
            labels[label_id].append(regions)

        labels_merged = pd.concat(labels[label_id], ignore_index=True, sort=False)

        labels_merged.to_csv(os.path.join(cfg['prediction_folder'], f'labels_{label_id}.csv'), index=False)

        labels_merged.rename(columns={'centroid-0': 'rlnCoordinateZ', 'centroid-1': 'rlnCoordinateY',
                                      'centroid-2': 'rlnCoordinateX', 'tomo': 'rlnMicrographName',
                                      'label': 'rlnLabel', 'area': 'rlnArea'}, inplace=True)
        starfile.write(labels_merged,
                       os.path.join(cfg['prediction_folder'], f'labels_{label_id}_particles.star'),
                       overwrite=True)


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
