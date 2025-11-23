import os
import yaml
import h5py
import torch
import numpy as np
from skimage.transform import resize, pyramid_reduce, pyramid_expand
from torch.nn import Softmax, Sigmoid
from torch.utils.data import DataLoader
from skimage.morphology import convex_hull_image
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import (
    Compose,
    EnsureType,
    LoadImaged,
    SpatialPad,
    EnsureTyped,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, PatchIter, MrcWriter
from cryosiam.apps.dense_simsiam_semantic import load_backbone_model, load_prediction_model, get_largest_cc


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
    mask_folder = cfg['mask_folder'] if 'mask_folder' in cfg and cfg['mask_folder'] else None
    num_classes = net_config['parameters']['network']['out_channels']
    threshold = cfg['parameters']['network']['threshold']
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

    writer = MrcWriter()

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

    pad_transform = SpatialPad(spatial_size=patch_size, method='end', mode='constant')
    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0), overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0), overlap=(0, 0.5, 0.5, 0.5))
    post_pred = Compose([EnsureType('numpy', dtype=np.float32, device=torch.device('cpu'))])

    test_dataset = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print('Prediction')
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
            original_size = test_sample['image'][0][0].shape
            img = pad_transform(test_sample['image'][0])
            patch_dataset = GridPatchDataset(data=[img], patch_iter=patch_iter)
            input_size = list(img[0].shape)
            # probs_out = np.zeros(input_size, dtype=np.float32)
            probs_out = np.zeros([num_classes] + input_size, dtype=np.float32)
            distances_out = np.zeros([num_classes] + input_size, dtype=np.float32)
            labels_out = np.zeros(input_size, dtype=np.uint8)
            loader = DataLoader(patch_dataset, batch_size=cfg['hyper_parameters']['batch_size'], num_workers=2)
            for item in loader:
                img, coord = item[0], item[1].numpy().astype(int)
                z, _ = backbone.forward_predict(img.cuda())
                out, d_out = prediction_model(z)
                if num_classes == 1:
                    out = Sigmoid()(out)  # (B, 1, D, H, W)
                else:
                    out = Softmax(dim=1)(out)  # (B, C, D, H, W)
                out = post_pred(out)
                d_out = post_pred(d_out)
                for batch_i in range(img.shape[0]):
                    c_batch = coord[batch_i][1:]
                    o_batch = out[batch_i]
                    d_batch = d_out[batch_i]
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
                    probs_out[(slice(0, num_classes),) + slices] = o_batch[(slice(0, num_classes),) + slices2]
                    distances_out[(slice(0, num_classes),) + slices] = d_batch[(slice(0, num_classes),) + slices2]
                    if num_classes == 1:
                        o_batch = (np.squeeze(o_batch, axis=0) > threshold).astype(np.uint32)  # (D, H, W)
                    else:
                        o_batch[o_batch < threshold] = 0
                        o_batch = np.argmax(o_batch, axis=0)  # (B, D, H, W)
                    labels_out[slices] = o_batch[slices2]

            labels_out = labels_out[tuple([slice(0, n) for n in original_size])]
            probs_out = probs_out[(slice(0, num_classes, ),) + tuple([slice(0, n) for n in original_size])]
            distances_out = distances_out[(slice(0, num_classes, ),) + tuple([slice(0, n) for n in original_size])]

            if mask_folder:
                filename = os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[0] + '_preds.h5'
                with h5py.File(os.path.join(mask_folder, filename), 'r') as f:
                    mask = f['labels'][()].astype(np.int8)
                if mask.shape != original_size:
                    mask = resize(mask, original_size, mode='constant', preserve_range=True).astype(np.int8)
                labels_out = labels_out * mask
                probs_out = probs_out * mask
                distances_out = distances_out * mask

            if 'postprocessing' in cfg['parameters']['network'] and cfg['parameters']['network']['postprocessing']:
                largest_cc = get_largest_cc(probs_out[0], threshold)
                if '3d_postprocessing' in cfg['parameters']['network'] and cfg['parameters']['network'][
                    '3d_postprocessing']:
                    size = largest_cc.shape
                    largest_cc = pyramid_reduce(largest_cc, 4, preserve_range=True)
                    labels_out = convex_hull_image(largest_cc)
                    if spatial_dims == 3:
                        labels_out = pyramid_expand(labels_out, 4, preserve_range=True)[:size[0], :size[1],
                                     :size[2]].astype(int)
                    else:
                        labels_out = pyramid_expand(labels_out, 4, preserve_range=True)[:size[0], :size[1]].astype(int)
                else:
                    for ind in range(largest_cc.shape[0]):
                        labels_out[ind] = convex_hull_image(largest_cc[ind])

            if 'save_internal_files' in cfg and cfg['save_internal_files']:
                with h5py.File(out_file.split(cfg['file_extension'])[0] + '_preds.h5', 'w') as f:
                    f.create_dataset('labels', data=labels_out)
                    f.create_dataset('probs', data=probs_out)
                    f.create_dataset('distances', data=distances_out)
            else:
                if 'save_original_file_extension' in cfg and cfg['save_original_file_extension']:
                    writer.set_data_array(labels_out.astype(np.uint8), channel_dim=None)
                    writer.write(out_file)
                else:
                    with h5py.File(out_file.split(cfg['file_extension'])[0] + '_preds.h5', 'w') as f:
                        f.create_dataset('labels', data=labels_out.astype(np.uint8))


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename)
