import os
import h5py
import yaml
import torch
import starfile
import collections
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, ArrayDataset
from monai.transforms import (
    Compose,
    SpatialPad,
    EnsureType,
    LoadImaged,
    EnsureTyped,
    CenterSpatialCrop,
    EnsureChannelFirst,
    NormalizeIntensityd,
    ScaleIntensityRanged
)

from cryosiam.data import MrcReader
from cryosiam.utils import parser_helper
from cryosiam.networks.nets import SimSiam


def load_prediction_model(checkpoint_path, contrastive=False, device="cuda:0"):
    """Load SimSiam trained model from given checkpoint
    :param checkpoint_path: path to the checkpoint
    :type checkpoint_path: str
    :param device: on which device should the model be loaded, default is cuda:0
    :type device: str
    :return: SimSiam model with laoded trained weights
    :rtype: cryoet_torch.networks.nets.SimSiam
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['hyper_parameters']['backbone_config' if contrastive else 'config']
    model = SimSiam(block_type=config['parameters']['network']['block_type'],
                    n_input_channels=config['parameters']['network']['in_channels'],
                    spatial_dims=config['parameters']['network']['spatial_dims'],
                    num_layers=config['parameters']['network']['num_layers'],
                    num_filters=config['parameters']['network']['num_filters'],
                    no_max_pool=config['parameters']['network']['no_max_pool'],
                    dim=config['parameters']['network']['dim'],
                    pred_dim=config['parameters']['network']['pred_dim'])
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k.replace("_model.", '')  # remove `_model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    device = torch.device(device)
    model.to(device)
    return model


def extract_patches_from_centers(image, regions, box_size):
    patches = []
    labels = []
    max_z, max_y, max_x = image.shape
    filtered_regions = {key: [] for key in regions.keys()}
    for i in range(len(regions['label'])):
        labels.append(regions['label'][i])
        slices = (slice(min(int(regions['centroid-0'][i]) - box_size, 0),
                        min(int(regions['centroid-0'][i]) + box_size, max_z)),
                  slice(min(int(regions['centroid-1'][i]) - box_size, 0),
                        min(int(regions['centroid-1'][i]) + box_size, max_y)),
                  slice(min(int(regions['centroid-2'][i]) - box_size, 0),
                        min(int(regions['centroid-2'][i]) + box_size, max_x)))
        patch = image[slices].copy()
        patches.append(patch)
        for key in regions.keys():
            filtered_regions[key].append(regions[key][i])
    return patches, labels, filtered_regions


def main(config_file_path, filename=None):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if 'trained_model' in cfg and cfg['trained_model'] is not None:
        checkpoint_path = cfg['trained_model']
    else:
        checkpoint_path = os.path.join(cfg['log_dir'], 'model', 'last.ckpt')

    if 'contrastive' in cfg and cfg['contrastive']:
        net = load_prediction_model(checkpoint_path, contrastive=True)
    else:
        net = load_prediction_model(checkpoint_path, contrastive=False)

    test_folder = cfg['data_folder']
    centers_file = cfg['centers_file']
    centers_patch_size = cfg['centers_patch_size']
    prediction_folder = cfg['prediction_folder']
    os.makedirs(prediction_folder, exist_ok=True)
    files = cfg['test_files']
    batch_size = cfg['hyper_parameters']['batch_size']
    if files is None:
        files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x))]
    if 'train_files' in cfg and cfg['train_files'] is not None:
        files += cfg['train_files']
    if filename:
        files = [filename]
    test_data = []
    for idx, file in enumerate(files):
        test_data.append({'image': os.path.join(test_folder, file),
                          'file_name': os.path.join(test_folder, file)})
    reader = MrcReader(read_in_mem=True)
    transforms = Compose(
        [
            LoadImaged(keys=['image'], reader=reader),
            ScaleIntensityRanged(keys=['image'], a_min=cfg['parameters']['data']['min'],
                                 a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
            NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                                divisor=cfg['parameters']['data']['std']),
            EnsureTyped(keys=['image'], data_type='numpy')
        ]
    )
    patch_transforms = Compose([EnsureChannelFirst(channel_dim='no_channel'),
                                SpatialPad(cfg['parameters']['data']['patch_size']),
                                CenterSpatialCrop(roi_size=cfg['parameters']['data']['patch_size']),
                                EnsureType(data_type='tensor')])
    post_pred = Compose([EnsureType('numpy', device=torch.device('cpu'))])

    test_dataset = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    if centers_file.endswith('.star'):
        regions = starfile.read(centers_file)
        if type(regions) is dict:
            regions = regions['particles']
        if 'rlnLabel' not in regions.keys():
            regions.insert(0, 'rlnLabel', range(1, 1 + regions.shape[0]))
            starfile.write(regions, os.path.join(prediction_folder, 'centers_with_labels.star'), overwrite=True)
        regions.rename(columns={'rlnCoordinateZ': 'centroid-0', 'rlnCoordinateY': 'centroid-1',
                                'rlnCoordinateX': 'centroid-2', 'rlnMicrographName': 'tomo',
                                'rlnLabel': 'label', 'rlnArea': 'area'}, errors='ignore', inplace=True)
    else:
        regions = pd.read_csv(centers_file)
        if 'label' not in regions.keys():
            regions.insert(0, 'label', range(1, 1 + len(regions)))
            regions.to_csv(os.path.join(prediction_folder, 'centers_with_labels.csv'), index=False)

    print('Prediction')
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            current_file = test_sample['file_name'][0]
            print(f'Running prediction for file {current_file}')
            out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
            if os.path.exists(out_file.split(cfg['file_extension'])[0] + '_embeds.h5'):
                print('Skipping', out_file)
                continue

            tomo_name = os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[0]
            current_regions = regions[regions['tomo'] == tomo_name]
            if current_regions.shape[0] == 0:
                print(f'No coordinates for file {current_file}')
                continue
            patches, labels, current_regions = extract_patches_from_centers(
                test_sample['image'][0].cpu().detach().numpy(),
                current_regions.to_dict('list'), centers_patch_size)
            print('Extracted patches')
            image_dataset = ArrayDataset(patches, labels=labels, img_transform=patch_transforms)
            embeds = np.zeros([cfg['parameters']['network']['dim']] + [len(patches)])
            tags = np.zeros(len(labels), dtype=np.int64)
            loader = DataLoader(image_dataset, batch_size=batch_size, num_workers=2)
            item_ind = 0
            for item in loader:
                img, label = item[0], item[1]
                if 'contrastive' in cfg and cfg['contrastive']:
                    _, out = net.forward_one(img.cuda())
                else:
                    out = net.encoder(img.cuda())
                embed = post_pred(out)
                for batch_i in range(label.shape[0]):
                    l_batch = label[batch_i]
                    e_batch = embed[batch_i]
                    embeds[:, item_ind] = e_batch
                    tags[item_ind] = l_batch
                    item_ind += 1

            print(f'Saving predictions for file {current_file}')
            with h5py.File(out_file.split(cfg['file_extension'])[0] + '_embeds.h5', 'w') as f:
                f.create_dataset('embeddings', data=embeds)
            with h5py.File(out_file.split(cfg['file_extension'])[0] + '_instance_labels.h5', 'w') as f:
                f.create_dataset('labels', data=tags)

            out_file = out_file.split(cfg['file_extension'])[0]
            current_regions = pd.DataFrame(current_regions)
            if 'area' not in current_regions.columns:
                current_regions['area'] = centers_patch_size ^ 3
            current_regions.drop(columns=['tomo'], inplace=True)
            current_regions.to_csv(out_file + '_instance_regions.csv', index=False)
            current_regions.rename(columns={'centroid-0': 'rlnCoordinateZ', 'centroid-1': 'rlnCoordinateY',
                                            'centroid-2': 'rlnCoordinateX', 'label': 'rlnLabel',
                                            'area': 'rlnArea'},
                                   errors='ignore', inplace=True)
            starfile.write(current_regions, out_file + '_instance_regions.star',
                           overwrite=True)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename)
