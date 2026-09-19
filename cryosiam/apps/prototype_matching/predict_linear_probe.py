import os
import h5py
import yaml
import torch
import mrcfile
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd,
    ScaleIntensityRanged, SpatialPad, EnsureChannelFirstd, EnsureTyped)

from cryosiam.utils import parser_helper
from cryosiam.transforms import NumpyToTensord
from cryosiam.data import MrcReader, MrcWriter, PatchIter
from cryosiam.apps.prototype_matching import load_search_model, load_decoder


class FPNLinearProbe(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_channels, n_classes)

    def forward(self, feat_vol):
        C, D, H, W = feat_vol.shape
        flat = feat_vol.reshape(C, -1).T
        logits = self.linear(flat)
        return logits.T.reshape(self.linear.out_features, D, H, W)


def save_probe(head, n_particle_classes, class_names, save_path):
    torch.save({'state_dict': head.state_dict(),
                'in_channels': head.linear.in_features,
                'n_particle_classes': n_particle_classes,
                'n_classes': head.linear.out_features,
                'class_names': class_names}, save_path)
    print(f'Saved linear probe → {save_path}')


def load_probe(save_path, device='cpu'):
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    head = FPNLinearProbe(ckpt['in_channels'], ckpt['n_classes']).to(device)
    head.load_state_dict(ckpt['state_dict'])
    head.eval()
    print(f'Loaded probe: {ckpt["n_particle_classes"]} particle classes  '
          f'{ckpt["class_names"]}')
    return head, ckpt['n_particle_classes'], ckpt['class_names']


def _stitch_slices(c, vol_size, patch_size):
    slices = []
    for i in range(3):
        z0, z1, p, s = c[i][0], c[i][1], patch_size[i], vol_size[i]
        if z0 == 0 and z1 >= s:
            slices.append(slice(0, s))
        elif z0 == 0:
            slices.append(slice(z0, z1 - p // 4))
        elif z1 >= s:
            slices.append(slice(z0 + p // 4, z1))
        else:
            slices.append(slice(z0 + p // 4, z1 - p // 4))
    return slices


def _patch_slices(c, vol_size, patch_size):
    slices = []
    for i in range(3):
        z0, z1, p, s = c[i][0], c[i][1], patch_size[i], vol_size[i]
        if z0 == 0 and z1 >= s:
            slices.append(slice(0, s))
        elif z0 == 0:
            slices.append(slice(0, 3 * p // 4))
        elif z1 >= s:
            slices.append(slice(p // 4, p - (z1 - s)))
        else:
            slices.append(slice(p // 4, 3 * p // 4))
    return slices


def extract_fpn_features(img, search_model, decoder, patch_size,
                         input_size, batch_size, patch_iter, device):
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *patch_size, device=device)
        s_dummy, _ = search_model.get_encoder_features_list(dummy)
        p_dummy = decoder._fpn_body(s_dummy)
        fpn_size = list(p_dummy.shape[-3:])

    out_ch = decoder.out_channels
    stride = [patch_size[i] // fpn_size[i] for i in range(3)]
    fpn_vol = [input_size[i] // stride[i] for i in range(3)]
    feat_vol = np.zeros((out_ch, *fpn_vol), dtype=np.float32)

    loader = DataLoader(
        GridPatchDataset(data=[img], patch_iter=patch_iter),
        batch_size=batch_size, num_workers=2)

    with torch.no_grad():
        for item in loader:
            patches = item[0].to(device)
            coords = item[1].numpy().astype(int)

            s_feats, _ = search_model.get_encoder_features_list(patches)
            p = decoder._fpn_body(s_feats)
            p_np = p.cpu().numpy()

            for b in range(patches.shape[0]):
                c = coords[b][1:]

                fc = [[c[i][0] // stride[i], c[i][1] // stride[i]]
                      for i in range(3)]

                ps = list(p.shape[-3:])
                os_ = _stitch_slices(fc, fpn_vol, ps)
                sl_ = _patch_slices(fc, fpn_vol, ps)

                feat_vol[:, os_[0], os_[1], os_[2]] = \
                    p_np[b][:, sl_[0], sl_[1], sl_[2]]

    return feat_vol, stride


def _load_or_extract_feats(tomo_path, feat_path, search_model, decoder,
                           patch_size, batch_size, patch_iter,
                           pad_transform, tomo_transforms, reuse, device):
    if reuse and os.path.exists(feat_path):
        print(f'  Loading cached FPN features')
        with h5py.File(feat_path, 'r') as hf:
            return hf['feat_vol'][()]

    sample = tomo_transforms({'image': tomo_path, 'file_name': tomo_path})
    img = pad_transform(sample['image'])
    input_size = list(img[0].shape)

    print('  Extracting FPN features...')
    with torch.no_grad():
        feat_vol, stride = extract_fpn_features(
            img=img, search_model=search_model, decoder=decoder,
            patch_size=patch_size, input_size=input_size,
            batch_size=batch_size, patch_iter=patch_iter, device=device)

    with h5py.File(feat_path, 'w') as hf:
        hf.create_dataset('feat_vol', data=feat_vol,
                          compression='gzip', compression_opts=4)
        hf.attrs['stride'] = stride
    return feat_vol


def train_linear_probe(search_model, decoder, scribble_tomo_path,
                       scribble_mask_path, n_particle_classes,
                       tomo_transforms, pad_transform, patch_size,
                       batch_size, patch_iter, prediction_folder,
                       device, background_label=None,
                       n_steps=200, lr=1e-3):
    print(f'\nTraining linear probe on {os.path.basename(scribble_tomo_path)}')

    ref_name = os.path.basename(scribble_tomo_path).split('.')[0]
    feat_path = os.path.join(prediction_folder, f'{ref_name}_ref_feats.h5')
    feat_vol = _load_or_extract_feats(
        scribble_tomo_path, feat_path, search_model, decoder,
        patch_size, batch_size, patch_iter,
        pad_transform, tomo_transforms, reuse=True, device=device)

    feat_t = F.normalize(torch.from_numpy(feat_vol).to(device), dim=0)
    C, D, H, W = feat_t.shape
    all_vox = feat_t.reshape(C, -1).T

    mask = mrcfile.open(scribble_mask_path, permissive=True).data.copy()
    mask_fpn = F.interpolate(
        torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        size=(D, H, W), mode='nearest')[0, 0].long().to(device)
    flat_mask = mask_fpn.flatten()

    particle_mask = (flat_mask > 0)
    if background_label is not None:
        particle_mask = particle_mask & (flat_mask != background_label)

    ann_voxels = all_vox[particle_mask]
    ann_labels = flat_mask[particle_mask] - 1

    n_ann = particle_mask.sum().item()
    print(f'  Annotated particle voxels: {n_ann}')
    for cls in range(n_particle_classes):
        print(f'    class {cls + 1}: {(ann_labels == cls).sum().item()} voxels')

    if background_label is not None:
        bg_mask = flat_mask == background_label
        bg_voxels = all_vox[bg_mask]
        n_bg = bg_mask.sum().item()
        print(f'    background (explicit label {background_label}): {n_bg} voxels')
    else:
        non_annotated = flat_mask == 0
        bg_pool = torch.where(non_annotated)[0]
        n_bg = n_ann
        bg_idx = bg_pool[torch.randperm(len(bg_pool), device=device)[:n_bg]]
        bg_voxels = all_vox[bg_idx]
        print(f'    background (random non-annotated): {n_bg} voxels')

    bg_labels = torch.full((len(bg_voxels),), n_particle_classes,
                           dtype=torch.long, device=device)
    voxels = torch.cat([ann_voxels, bg_voxels])
    labels = torch.cat([ann_labels, bg_labels])
    n_classes = n_particle_classes + 1

    n_per = torch.tensor([(labels == k).sum()
                          for k in range(n_classes)],
                         dtype=torch.float32, device=device)
    weights = n_per.sum() / (n_classes * n_per.clamp(min=1))

    head = FPNLinearProbe(C, n_classes).to(device)
    optimizer = torch.optim.Adam(head.linear.parameters(), lr=lr)

    print(f'  Training {n_classes} classes, {n_steps} steps...')
    for step in range(n_steps):
        logits = head.linear(voxels)
        loss = F.cross_entropy(logits, labels, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            acc = (logits.argmax(1) == labels).float().mean()
            print(f'    step {step + 1:4d}/{n_steps}  '
                  f'loss={loss.item():.4f}  acc={acc:.1%}')

    head.eval()
    return head


def predict_maps(tomo_path, feat_path, search_model, decoder,
                 head, tomo_transforms, pad_transform,
                 patch_size, batch_size, patch_iter,
                 original_size, reuse, device):
    feat_vol = _load_or_extract_feats(
        tomo_path, feat_path, search_model, decoder,
        patch_size, batch_size, patch_iter,
        pad_transform, tomo_transforms, reuse, device)

    feat_t = F.normalize(torch.from_numpy(feat_vol).to(device), dim=0)

    with torch.no_grad():
        logits = head(feat_t)
        logit_maps = logits.cpu().numpy()

    D, H, W = logit_maps.shape[1:]
    if list(original_size) != [D, H, W]:
        ups = []
        for k in range(logit_maps.shape[0]):
            up = F.interpolate(
                torch.from_numpy(logit_maps[k]).unsqueeze(0).unsqueeze(0),
                size=list(original_size),
                mode='trilinear', align_corners=False)[0, 0].numpy()
            ups.append(up)
        logit_maps = np.stack(ups)

    prob_maps = torch.softmax(torch.from_numpy(logit_maps), dim=0).numpy()

    return logit_maps, prob_maps


def main(config_file_path, filename=None):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    with open(config_file_path) as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get('mode', 'predict')
    print(f'Mode: {mode}')

    checkpoint_path = cfg['trained_model']
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_config = checkpoint['hyper_parameters']['config']
    search_model = load_search_model(checkpoint_path, device)
    decoder = load_decoder(checkpoint_path, device)
    search_model.eval()
    decoder.eval()

    data_cfg = model_config['parameters']['data']
    patch_size = data_cfg['patch_size']
    file_ext = cfg['file_extension']
    batch_size = cfg['hyper_parameters']['batch_size']
    reuse = bool(cfg.get('reuse_predictions', True))

    tomo_transforms = Compose([
        LoadImaged(keys='image', reader=MrcReader(read_in_mem=True)),
        EnsureChannelFirstd(keys='image'),
        NumpyToTensord(keys='image'),
        ScaleIntensityRanged(keys='image',
                             a_min=data_cfg['min'], a_max=data_cfg['max'],
                             b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys='image',
                            subtrahend=data_cfg['mean'],
                            divisor=data_cfg['std']),
        EnsureTyped(keys='image', data_type='tensor')])

    patch_iter = PatchIter(patch_size=tuple(patch_size),
                           start_pos=(0, 0, 0),
                           overlap=(0, 0.5, 0.5, 0.5))
    pad_transform = SpatialPad(spatial_size=patch_size,
                               method='end', mode='edge')

    probe_path = cfg['probe_path']
    os.makedirs(os.path.dirname(probe_path), exist_ok=True)
    os.makedirs(cfg['prediction_folder'], exist_ok=True)

    if mode in ('train', 'train_and_predict'):
        n_classes = int(cfg['n_classes'])
        class_names = cfg.get('class_names',
                              [f'class_{i}' for i in range(1, n_classes + 1)])
        background_label = cfg.get('background_label', None)
        if background_label is not None:
            background_label = int(background_label)
            print(f'Background: explicit label {background_label} from scribble mask')
        else:
            print('Background: random non-annotated voxels')

        head = train_linear_probe(search_model=search_model,
                                  decoder=decoder,
                                  scribble_tomo_path=cfg['reference_file'],
                                  scribble_mask_path=cfg['reference_mask'],
                                  n_particle_classes=n_classes,
                                  tomo_transforms=tomo_transforms,
                                  pad_transform=pad_transform,
                                  patch_size=patch_size,
                                  batch_size=batch_size,
                                  patch_iter=patch_iter,
                                  prediction_folder=cfg['prediction_folder'],
                                  device=device,
                                  background_label=background_label,
                                  n_steps=int(cfg.get('n_probe_steps', 200)),
                                  lr=float(cfg.get('probe_lr', 1e-3)))

        save_probe(head, n_classes, class_names, probe_path)

        if mode == 'train':
            print('\nDone. Run with mode: predict to map the embedding space.')
            return

    if mode in ('predict', 'train_and_predict'):
        if mode == 'predict':
            head, n_classes, class_names = load_probe(probe_path, device)

        map_names = list(class_names) + ['background']

        data_folder = cfg['data_folder']
        output_folder = cfg.get('output_folder', cfg['prediction_folder'])
        os.makedirs(output_folder, exist_ok=True)

        map_writer = MrcWriter(output_dtype=np.float32, overwrite=True)
        map_writer.set_metadata({'voxel_size': 1})

        files = cfg.get('test_files') or [
            f for f in os.listdir(data_folder)
            if os.path.isfile(os.path.join(data_folder, f))
               and f.endswith(file_ext)]
        if filename:
            files = [filename]

        test_data = [{'image': os.path.join(data_folder, f),
                      'file_name': os.path.join(data_folder, f)}
                     for f in files]
        test_ds = Dataset(data=test_data, transform=tomo_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=1,
                                 collate_fn=list_data_collate)

        print(f'\n{len(files)} tomogram(s) to map  classes: {class_names}')

        for test_sample in test_loader:
            current_file = test_sample['file_name'][0]
            tomo_name = os.path.basename(current_file).split(file_ext)[0]
            original_size = list(test_sample['image'][0][0].shape)
            feat_path = os.path.join(cfg['prediction_folder'],
                                     f'{tomo_name}_feats.h5')

            print(f'\n{"=" * 55}')
            print(f'Tomogram: {tomo_name}  size: {original_size}')
            print('  Mapping embedding space...')

            logit_maps, prob_maps = predict_maps(tomo_path=current_file,
                                                 feat_path=feat_path,
                                                 search_model=search_model,
                                                 decoder=decoder,
                                                 head=head,
                                                 tomo_transforms=tomo_transforms,
                                                 pad_transform=pad_transform,
                                                 patch_size=patch_size,
                                                 batch_size=batch_size,
                                                 patch_iter=patch_iter,
                                                 original_size=original_size,
                                                 reuse=reuse,
                                                 device=device)

            h5_out = os.path.join(cfg['prediction_folder'],
                                  f'{tomo_name}_probe_maps.h5')
            with h5py.File(h5_out, 'w') as hf:
                for k, name in enumerate(map_names):
                    hf.create_dataset(f'logit_{name}', data=logit_maps[k],
                                      compression='gzip', compression_opts=4)
                    hf.create_dataset(f'prob_{name}', data=prob_maps[k],
                                      compression='gzip', compression_opts=4)
                hf.attrs['class_names'] = map_names
                hf.attrs['n_particle_classes'] = n_classes
                hf.attrs['map_type'] = 'linear_probe_embedding_mapper'
            print(f'  Saved: {os.path.basename(h5_out)}')

            if cfg.get('save_prob_mrc', False):
                for k, name in enumerate(map_names):
                    mrc_out = os.path.join(output_folder,
                                           f'{tomo_name}_{name}_probe_prob.mrc')
                    map_writer.set_data_array(prob_maps[k], channel_dim=None)
                    map_writer.write(mrc_out)

            if cfg.get('save_logit_mrc', False):
                for k, name in enumerate(map_names):
                    mrc_out = os.path.join(output_folder,
                                           f'{tomo_name}_{name}_probe_logit.mrc')
                    map_writer.set_data_array(logit_maps[k], channel_dim=None)
                    map_writer.write(mrc_out)

        print(f'\nDone. Mapped {len(files)} tomogram(s).')


if __name__ == '__main__':
    parser = parser_helper('Few-shot linear probe embedding mapper')
    args = parser.parse_args()
    main(args.config_file, getattr(args, 'filename', None))
