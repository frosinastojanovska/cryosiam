import os
import yaml
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    SpatialPad,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType
)

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, PatchIter, MrcWriter
from cryosiam.transforms import ScaleIntensityd, InvertIntensityd
from cryosiam.apps.dense_simsiam_regression import load_backbone_model, load_prediction_model


def main(config_file_path, filename):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    checkpoint_path = cfg['trained_model']

    backbone = load_backbone_model(checkpoint_path)
    prediction_model = load_prediction_model(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    num_output_channels = net_config['parameters']['network']['n_output_channels']
    patch_size = net_config['parameters']['data']['patch_size']
    spatial_dims = net_config['parameters']['network']['spatial_dims']
    os.makedirs(prediction_folder, exist_ok=True)
    if filename:
        files = [filename]
    else:
        files = cfg['test_files']
        if files is None:
            files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x)) and
                     x.endswith(cfg['file_extension'])]
    test_data = []
    for idx, file in enumerate(files):
        test_data.append({'image': os.path.join(test_folder, file),
                          'file_name': os.path.join(test_folder, file)})
    reader = MrcReader(read_in_mem=True)

    transforms = Compose(
        [
            LoadImaged(keys='image', reader=reader),
            EnsureChannelFirstd(keys='image'),
            InvertIntensityd(keys='image'),
            ScaleIntensityd(keys='image'),
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
            out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
            voxel_size = reader.read(test_sample['file_name'][0]).voxel_size
            original_size = test_sample['image'][0][0].shape
            img = pad_transform(test_sample['image'][0])
            patch_dataset = GridPatchDataset(data=[img], patch_iter=patch_iter)
            input_size = list(img[0].shape)
            preds_out = np.zeros([num_output_channels] + input_size, dtype=np.float32)
            loader = DataLoader(patch_dataset, batch_size=cfg['hyper_parameters']['batch_size'], num_workers=2)
            for item in loader:
                img, coord = item[0], item[1].numpy().astype(int)
                z, _ = backbone.forward_predict(img.cuda())
                out = prediction_model(z)
                out = post_pred(out)
                for batch_i in range(img.shape[0]):
                    c_batch = coord[batch_i][1:]
                    o_batch = out[batch_i]
                    # avoid getting patch that is outside of the original dimensions of the image
                    if c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or \
                            c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or \
                            (spatial_dims == 3 and c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                        continue
                    # create slices for the coordinates in the output to get only the middle of the patch
                    # and the separate cases for the first and last patch in each dimension
                    slices = tuple(
                        slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, c[1])
                        if c[1] >= s else slice(c[0] + p // 4, c[1] - p // 4)
                        for c, s, p in zip(c_batch, input_size, patch_size))
                    # create slices to crop the patch so we only get the middle information
                    # and the separate cases for the first and last patch in each dimension
                    slices2 = tuple(
                        slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s))
                        if c[1] >= s else slice(p // 4, 3 * p // 4)
                        for c, s, p in zip(c_batch, input_size, patch_size))
                    preds_out[(slice(0, num_output_channels),) + slices] = o_batch[(slice(0, num_output_channels),)
                                                                                   + slices2]

            preds_out = preds_out[(slice(0, num_output_channels),) + tuple([slice(0, n) for n in original_size])]

            if cfg['scale_prediction']:
                preds_out = (preds_out - preds_out.min()) / (preds_out.max() - preds_out.min())

            if cfg['save_raw_predictions']:
                with h5py.File(out_file.split(cfg['file_extension'])[0] + '_preds.h5', 'w') as f:
                    f.create_dataset('preds', data=preds_out)

            writer = MrcWriter(output_dtype=np.float32, overwrite=True)
            writer.set_metadata({'voxel_size': voxel_size})
            writer.set_data_array(preds_out[0], channel_dim=None)
            writer.write(out_file.split(cfg['file_extension'])[0] + f'{cfg["file_extension"]}')


if __name__ == "__main__":
    parser = parser_helper("Run CryoSiam denoising prediction")
    args = parser.parse_args()
    main(args.config_file, args.filename)
