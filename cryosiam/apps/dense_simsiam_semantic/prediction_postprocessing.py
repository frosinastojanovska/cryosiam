import os
import yaml
import h5py
import numpy as np
import pandas as pd
from monai.data import ITKReader
from skimage.measure import label, regionprops_table

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, MrcWriter


def postprocessing_labels(labels, min_sizes):
    for i, min_size in enumerate(min_sizes):
        if min_size == -1:
            continue
        print(f'Processing labels {i}, with min size {min_size}')
        segments = label(labels == i)
        regions = pd.DataFrame(regionprops_table(segments, properties=['label', 'area']))
        print(regions.shape)
        regions = regions[regions.area > min_size]
        print(regions.shape)
        segments[~np.isin(segments, regions.label.tolist())] = 0
        labels[labels == i] = ((segments > 0) * i)[labels == i].astype(np.uint8)

    return labels


def main(config_file_path, filename=None):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    data_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    files = cfg['test_files']

    if filename:
        files = [filename]

    reader = MrcReader(read_in_mem=True)
    writer = MrcWriter()

    if files is None:
        files = [x for x in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, x))]

    print('Post-processing')
    for file in files:
        filename = os.path.join(prediction_folder, f"{file.split(cfg['file_extension'])[0]}_preds.h5")
        if os.path.isfile(filename):
            with h5py.File(filename) as f:
                labels_out = f['labels'][()]
            labels_out = postprocessing_labels(labels_out, cfg['parameters']['network']['postprocessing_sizes'])
            with h5py.File(filename, 'w') as f:
                f.create_dataset('labels', data=labels_out)
        else:
            filename = os.path.join(prediction_folder, f"{file.split(cfg['file_extension'])[0]}_preds.mrc")
            labels_out = reader.read(filename)
            if cfg['file_extension'] in ['.mrc', '.rec']:
                labels_out = labels_out.data
                labels_out.setflags(write=True)
            else:
                labels_out = labels_out[0]
            labels_out = postprocessing_labels(labels_out,
                                               cfg['parameters']['network']['postprocessing_sizes']).astype(np.uint8)
            writer.set_data_array(labels_out, channel_dim=None)
            writer.write(filename)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename)
