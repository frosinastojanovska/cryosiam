import os
import csv
import h5py
import yaml
import torch
import starfile
import numpy as np
import pandas as pd
import collections
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from skimage.morphology import convex_hull_image

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
from cryosiam.apps.simsiam_prediction import load_prediction_model


def extract_patches_from_instances_mask(image, instances_mask, regions=None, min_particle_size=None,
                                        max_particle_size=None, masking=0, expand_labels_size=1):
    patches = []
    labels = []
    instances_mask = expand_labels(instances_mask, distance=expand_labels_size)
    if regions is None:
        regions = regionprops_table(instances_mask, properties=['label', 'area', 'bbox', 'centroid'])
    regions['hull_area'] = [0] * regions['label'].shape[0]
    filtered_regions = {key: [] for key in regions.keys()}
    for i in range(regions['label'].shape[0]):
        if min_particle_size and regions['area'][i] < min_particle_size:
            continue
        if max_particle_size and regions['area'][i] > max_particle_size:
            continue
        labels.append(regions['label'][i])
        mask = instances_mask == regions['label'][i]
        slices = (slice(regions['bbox-0'][i], regions['bbox-3'][i]),
                  slice(regions['bbox-1'][i], regions['bbox-4'][i]),
                  slice(regions['bbox-2'][i], regions['bbox-5'][i]))
        sub_mask = mask[slices]
        patch = image[slices].copy()
        if masking == 1:
            hull_mask = convex_hull_image(sub_mask)
            patch[hull_mask == 0] = 0
        elif masking == 2:
            patch[sub_mask == 0] = 0
        patches.append(patch)
        regions['hull_area'][i] = np.sum(hull_mask) if masking == 1 else 0
        for key in regions.keys():
            filtered_regions[key].append(regions[key][i])
    return patches, labels, filtered_regions


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
        checkpoint_path = os.path.join(cfg['log_dir'], 'model', 'last.ckpt')

    if 'contrastive' in cfg and cfg['contrastive']:
        net = load_prediction_model(checkpoint_path, contrastive=True, device=device)
    else:
        net = load_prediction_model(checkpoint_path, contrastive=False, device=device)

    test_folder = cfg['data_folder']
    instances_mask_folder = cfg['instances_mask_folder']
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
                          'mask': os.path.join(instances_mask_folder, file),
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

    print('Prediction')
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            current_file = test_sample['file_name'][0]
            print(f'Running prediction for file {current_file}')
            out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
            if os.path.exists(out_file.split(cfg['file_extension'])[0] + '_embeds.h5'):
                print('Skipping', out_file)
                continue
            if 'instances' in cfg:
                min_dist = cfg['instances']['min_center_distance']
                max_dist = cfg['instances']['max_center_distance']
                suffix = f'_min-{min_dist}_max-{max_dist}'
            else:
                suffix = ''
            instances_file_name = os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[
                                      0] + f'_instance_preds{suffix}.h5'
            if not os.path.exists(os.path.join(instances_mask_folder, instances_file_name)):
                continue
            with h5py.File(os.path.join(instances_mask_folder, instances_file_name)) as f:
                mask = f['instances'][()]
            regions_file_name = os.path.join(instances_mask_folder,
                                             os.path.basename(test_sample['file_name'][0]).split(cfg['file_extension'])[
                                                 0] + f'_instance_regions{suffix}.csv')
            if os.path.exists(regions_file_name):
                regions = collections.defaultdict(list)
                with open(regions_file_name, newline='') as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        for k, v in row.items():
                            regions[k].append(int(v) if 'label' in k or 'bbox' in k else float(v))
                    for key in regions.keys():
                        regions[key] = np.asarray(regions[key])
            else:
                regions = None
            patches, labels, regions = extract_patches_from_instances_mask(
                test_sample['image'][0].cpu().detach().numpy(),
                mask,
                regions,
                cfg['min_particle_size'],
                cfg['max_particle_size'],
                cfg['masking_type'],
                cfg['expand_labels'])
            print('Extracted patches')
            image_dataset = ArrayDataset(patches, labels=labels, img_transform=patch_transforms)
            embeds = np.zeros([cfg['parameters']['network']['dim']] + [len(patches)])
            tags = np.zeros(len(labels), dtype=np.int64)
            loader = DataLoader(image_dataset, batch_size=batch_size, num_workers=2)
            item_ind = 0
            for item in loader:
                img, label = item[0], item[1]
                if 'contrastive' in cfg and cfg['contrastive']:
                    if device == 'cuda:0':
                        _, out = net.forward_one(img.cuda())
                    else:
                        _, out = net.forward_one(img)
                else:
                    if device == 'cuda:0':
                        out = net.encoder(img.cuda())
                    else:
                        out = net.encoder(img)
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
            with open(out_file + '_instance_regions.csv', 'w') as f:
                w = csv.writer(f)
                w.writerow(list(regions.keys()))
                for i in range(len(regions['label'])):
                    w.writerow([regions[l][i] for l in regions.keys()])

            regions = pd.DataFrame(regions)
            regions.drop(columns=['hull_area'], inplace=True)
            regions.rename(columns={'centroid-0': 'rlnCoordinateZ', 'centroid-1': 'rlnCoordinateY',
                                    'centroid-2': 'rlnCoordinateX', 'bbox-0': 'rlnBbox-0', 'bbox-1': 'rlnBbox-1',
                                    'bbox-2': 'rlnBbox-2', 'bbox-3': 'rlnBbox-3', 'bbox-4': 'rlnBbox-4',
                                    'bbox-5': 'rlnBbox-5', 'label': 'rlnLabel', 'area': 'rlnArea'}, inplace=True)
            starfile.write(regions, out_file + '_instance_regions.star',
                           overwrite=True)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename)
