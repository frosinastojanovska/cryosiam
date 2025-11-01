import os
import h5py
import yaml
import numpy as np
from torch.utils.data import DataLoader
from skimage.segmentation import expand_labels
from monai.data import Dataset, list_data_collate

from cryosiam.utils import parser_helper


def main(config_file_path, filename=None):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    semantic_mask = cfg['filtering_mask_folder']
    os.makedirs(prediction_folder + '_filtered', exist_ok=True)
    files = cfg['test_files']
    if files is None:
        files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x)) and
                 x.endswith(cfg['file_extension'])]

    if filename:
        files = [filename]

    test_data = []
    for idx, file in enumerate(files):
        test_data.append({'image': os.path.join(test_folder, file),
                          'file_name': os.path.join(test_folder, file)})

    test_dataset = Dataset(data=test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print('Prediction')
    for i, test_sample in enumerate(test_loader):
        out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
        with h5py.File(out_file.split(cfg['file_extension'])[0] + '_instance_preds.h5', 'r') as f:
            instances = f['instances'][()]

        file_path = os.path.join(semantic_mask, os.path.basename(test_sample['file_name'][0]))
        with h5py.File(file_path.split(cfg['file_extension'])[0] + '_preds.h5', 'r') as f:
            semantic = f['labels'][()]

        semantic = expand_labels((semantic == 1).astype(int), 2)
        instances[~np.isin(instances, np.unique(instances[semantic == 1]))] = 0

        out_file = os.path.join(prediction_folder + '_filtered', os.path.basename(test_sample['file_name'][0]))
        suffix = f'_instance_preds.h5'
        with h5py.File(out_file.split(cfg['file_extension'])[0] + suffix, 'w') as f:
            f.create_dataset('instances', data=instances)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename)
