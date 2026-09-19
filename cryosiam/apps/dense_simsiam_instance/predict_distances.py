import os
import h5py
import yaml
import torch
import numpy as np
from skimage.transform import resize
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import (
    Compose,
    EnsureType,
    SpatialPad,
    LoadImaged,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, PatchIter
from cryosiam.apps.dense_simsiam_instance import load_backbone_model, load_prediction_model


def main(config_file_path, filename=None):
    if not torch.cuda.is_available():
        print('CUDA not available, using CPU instead.')
        device = 'cpu'
    else:
        device = 'cuda:0'

    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if 'trained_model' in cfg and cfg['trained_model'] is not None:
        checkpoint_path = cfg['trained_model']
    else:
        checkpoint_path = os.path.join(cfg['log_dir'], 'model', 'model_best.ckpt')
    backbone = load_backbone_model(checkpoint_path, device)
    prediction_model = load_prediction_model(checkpoint_path, device)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    mask_folder = cfg['mask_folder'] if 'mask_folder' in cfg and cfg['mask_folder'] else None
    patch_size = net_config['parameters']['data']['patch_size']
    spatial_dims = net_config['parameters']['network']['spatial_dims']
    os.makedirs(prediction_folder, exist_ok=True)
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
    reader = MrcReader(read_in_mem=True)

    transforms = Compose(
        [
            LoadImaged(keys='image', reader=reader),
            EnsureChannelFirstd(keys='image'),
            NumpyToTensord(keys='image'),
            ScaleIntensityRanged(keys='image', a_min=cfg['parameters']['data']['min'],
                                 a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
            NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                                divisor=cfg['parameters']['data']['std']),
            EnsureTyped(keys='image', data_type='tensor')
        ]
    )
    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0), overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0), overlap=(0, 0.5, 0.5, 0.5))
    post_pred = Compose([EnsureType('numpy', dtype=np.float32, device=torch.device('cpu'))])
    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')

    test_dataset = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print('Prediction')
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            current_file = test_sample['file_name'][0]
            print(f'Running prediction for file {current_file}')
            out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
            original_size = test_sample['image'][0][0].shape
            img = pad_transform(test_sample['image'][0])
            patch_dataset = GridPatchDataset(data=[img], patch_iter=patch_iter)
            input_size = list(img[0].shape)
            if not os.path.isfile(out_file.split(cfg['file_extension'])[0] + '_distances.h5'):
                distances_out = np.zeros(input_size, dtype=np.float32)
                loader = DataLoader(patch_dataset, batch_size=cfg['hyper_parameters']['batch_size'], num_workers=2)
                for item in loader:
                    img, coord = item[0], item[1].numpy().astype(int)
                    if device == 'cuda:0':
                        z, _ = backbone.forward_predict(img.cuda())
                    else:
                        z, _ = backbone.forward_predict(img)
                    _, distance_pred, _ = prediction_model(z)
                    distance_pred = post_pred(distance_pred)

                    for batch_i in range(img.shape[0]):
                        c_batch = coord[batch_i][1:]
                        d_batch = distance_pred[batch_i][0]
                        if c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or \
                                c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or \
                                (spatial_dims == 3 and c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                            continue
                        slices = tuple(
                            slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, c[1])
                            if c[1] >= s else slice(c[0] + p // 4, c[1] - p // 4)
                            for c, s, p in zip(c_batch, input_size, patch_size))
                        slices2 = tuple(
                            slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s))
                            if c[1] >= s else slice(p // 4, 3 * p // 4)
                            for c, s, p in zip(c_batch, input_size, patch_size))
                        distances_out[slices] = d_batch[slices2]

                distances_out = distances_out[tuple([slice(0, n) for n in original_size])]

                if mask_folder:
                    print(f'Masking out of file {current_file}')
                    mask_filename = os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[
                                        0] + '_preds.h5'
                    with h5py.File(os.path.join(mask_folder, mask_filename), 'r') as f:
                        mask = f['labels'][()].astype(np.int8)
                    if mask.shape != original_size:
                        mask = resize(mask, original_size, mode='constant', preserve_range=True).astype(np.int8)
                    distances_out = distances_out * mask

                print(f'Saving predictions for file {current_file}')
                with h5py.File(out_file.split(cfg['file_extension'])[0] + '_distances.h5', 'w') as f:
                    f.create_dataset('distances', data=distances_out)
            else:
                print(f'Distances file already exists for {current_file}, skipping.')


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename)
