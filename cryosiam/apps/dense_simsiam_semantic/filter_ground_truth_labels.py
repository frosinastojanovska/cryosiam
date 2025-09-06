import os
import yaml
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter

def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    reader = MrcReader(read_in_mem=True)
    writer_labels = MrcWriter(output_dtype=np.float32, overwrite=True)
    writer_labels.set_metadata({'voxel_size': 1})

    labels_folder = cfg['labels_folder']
    selected_labels = cfg['selected_labels']
    filtered_labels_folder = cfg['filtered_labels_folder']

    os.makedirs(filtered_labels_folder, exist_ok=True)
    files = [x for x in os.listdir(labels_folder) if x.endswith(cfg['file_extension'])]

    if cfg['train_files'] is not None:
        if cfg['test_files'] is None:
            cfg['test_files'] = []
        files = [x for x in files if x in cfg['train_files'] or x in cfg['test_files']]

    for file in files:
        print(f'Processing tomo {file}')
        root_file_name = file.split(cfg['file_extension'])[0]
        labels = reader.read(os.path.join(labels_folder, f'{root_file_name}{cfg["file_extension"]}'))
        labels = labels.data
        labels.setflags(write=True)

        labels[~np.isin(labels, selected_labels)] = 0
        labels[labels > 0] = 1

        labels = expand_labels(labels, 1)
        labels_path = os.path.join(filtered_labels_folder, f'{root_file_name}{cfg["file_extension"]}')
        # save the labels
        writer_labels.set_data_array(labels, channel_dim=None)
        writer_labels.write(labels_path)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
