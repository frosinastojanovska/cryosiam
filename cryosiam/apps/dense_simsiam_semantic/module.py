import os
import random
import pickle
import collections

import torch
import torch.nn as nn
import lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from monai.utils import set_determinism
from monai.data.utils import worker_init_fn
from monai.losses import GeneralizedDiceLoss
from monai.data import CacheDataset, Dataset, ITKReader, SmartCacheDataset, ThreadDataLoader, NumpyReader
from monai.transforms import (
    OneOf,
    Compose,
    RandZoomd,
    Identityd,
    LoadImaged,
    SqueezeDimd,
    SpatialPadd,
    EnsureTyped,
    RandScaleIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    ScaleIntensityRanged
)

from cryosiam.data import MrcReader
from cryosiam.data.utils import list_data_collate
from cryosiam.utils import supervised_semantic_train_val_split
from cryosiam.networks.nets import (
    DenseSimSiam,
    SemanticHeads
)
from cryosiam.transforms import ClipIntensityd, RandomLowPassBlurd, RandomGaussianNoised, RandomHighPassSharpend


class SemanticSegmentationModule(pl.LightningModule):
    def __init__(self, config, backbone_config):
        super().__init__()
        self.config = config
        self.checkpoint = torch.load(config['pretrained_model'], weights_only=False)
        self._model_backbone = DenseSimSiam(block_type=backbone_config['parameters']['network']['block_type'],
                                            spatial_dims=backbone_config['parameters']['network']['spatial_dims'],
                                            n_input_channels=backbone_config['parameters']['network'][
                                                'in_channels'],
                                            num_layers=backbone_config['parameters']['network']['num_layers'],
                                            num_filters=backbone_config['parameters']['network']['num_filters'],
                                            fpn_channels=backbone_config['parameters']['network']['fpn_channels'],
                                            no_max_pool=backbone_config['parameters']['network']['no_max_pool'],
                                            dim=backbone_config['parameters']['network']['dim'],
                                            pred_dim=backbone_config['parameters']['network']['pred_dim'],
                                            dense_dim=backbone_config['parameters']['network']['dense_dim'],
                                            dense_pred_dim=backbone_config['parameters']['network'][
                                                'dense_pred_dim'],
                                            include_levels=backbone_config['parameters']['network'][
                                                'include_levels_loss']
                                            if 'include_levels_loss' in backbone_config['parameters'][
                                                'network'] else False,
                                            add_later_conv=backbone_config['parameters']['network'][
                                                'add_fpn_later_conv']
                                            if 'add_fpn_later_conv' in backbone_config['parameters'][
                                                'network'] else False,
                                            decoder_type=backbone_config['parameters']['network']['decoder_type']
                                            if 'decoder_type' in backbone_config['parameters'][
                                                'network'] else 'fpn',
                                            decoder_layers=backbone_config['parameters']['network']['fpn_layers']
                                            if 'fpn_layers' in backbone_config['parameters']['network'] else 2)
        # self._model_backbone.load_state_dict(self.checkpoint['state_dict'])
        new_state_dict = collections.OrderedDict()
        for k, v in self.checkpoint['state_dict'].items():
            name = k.replace('_model.', '')  # remove `_model.`
            new_state_dict[name] = v
        self._model_backbone.load_state_dict(new_state_dict)
        self.backbone_config = backbone_config

        self._model = SemanticHeads(n_input_channels=config['parameters']['network']['dense_dim'],
                                    num_classes=config['parameters']['network']['out_channels'],
                                    spatial_dims=config['parameters']['network']['spatial_dims'],
                                    filters=config['parameters']['network']['filters'],
                                    kernel_size=config['parameters']['network']['kernel_size'],
                                    padding=config['parameters']['network']['padding'])

        self.batch_size = self.config['hyper_parameters']['batch_size']

        self.semantic_loss_function = nn.BCEWithLogitsLoss() if config['parameters']['network'][
                                                                    'out_channels'] == 1 else nn.CrossEntropyLoss()

        self.semantic_loss_function2 = GeneralizedDiceLoss(include_background=True, to_onehot_y=True,
                                                           sigmoid=config['parameters']['network']['out_channels'] == 1,
                                                           softmax=config['parameters']['network']['out_channels'] > 1,
                                                           reduction='mean')
        self.distance_loss_function = nn.MSELoss()

        self.spatial_dims = self.config['parameters']['network']['spatial_dims']
        self.optimizer_name = self.config['hyper_parameters']['optimizer']
        self.lr = self.config['hyper_parameters']['lr']
        self.momentum = self.config['hyper_parameters']['momentum'] \
            if 'momentum' in self.config['hyper_parameters'] else None
        self.weight_decay = self.config['hyper_parameters']['weight_decay']
        self.max_epochs = self.config['hyper_parameters']['max_epochs']

        self.use_noisy_input = self.config['parameters']['transforms']['use_noisy_input'] \
            if 'use_noisy_input' in self.config['parameters']['transforms'] else False

        self.save_hyperparameters()
        if int(self.config['parameters']['gpu_devices']) > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False

    def forward(self, x):
        z, _ = self._model_backbone.forward_predict(x)
        return self._model(z)

    def prepare_data(self):
        data_root = os.path.normpath(self.config['data_folder'])
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        if not os.path.isfile(train_val_path):
            train_files, val_files = supervised_semantic_train_val_split(data_root,
                                                                         seg_root=self.config['labels_folder'],
                                                                         out_root=self.config['temp_dir'],
                                                                         noisy_data_root=self.config[
                                                                             'noisy_data_folder']
                                                                         if 'noisy_data_folder' in self.config else None,
                                                                         files=self.config['train_files'],
                                                                         ratio=self.config['validation_ratio'],
                                                                         val_files=self.config['val_files'],
                                                                         patches_folder=self.config['patches_folder'],
                                                                         file_ext=self.config['file_extension'])
            with open(train_val_path, 'wb') as f:
                pickle.dump({'train_files': train_files, 'val_files': val_files}, f)

    def initialization(self):
        train_val_path = os.path.join(self.config['log_dir'], 'train_val_split.pkl')
        with open(train_val_path, 'rb') as f:
            data = pickle.load(f)
            train_files, val_files = data['train_files'], data['val_files']
        return train_files, val_files

    def setup(self, stage=None):
        # set data
        train_files, val_files = self.initialization()
        if self.trainer.world_size > 1 and self.config['hyper_parameters']['cache_rate'] > 0:
            partition_len = len(train_files) // self.trainer.world_size
            global_rank = self.trainer.global_rank
            train_files = train_files[(partition_len * global_rank):(partition_len * global_rank + partition_len)]
            print(f'World size: {self.trainer.world_size}, global_rank: {global_rank}, '
                  f'start_index: {partition_len * global_rank}')
            partition_len = len(val_files) // self.trainer.world_size
            global_rank = self.trainer.global_rank
            val_files = val_files[(partition_len * global_rank):(partition_len * global_rank + partition_len)]
            print(f'World size: {self.trainer.world_size}, global_rank: {global_rank}, '
                  f'start_index: {partition_len * global_rank}')

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        if self.use_noisy_input:
            keys = ['image', 'noisy_image']
        else:
            keys = ['image']

        image_transforms = [
            RandomLowPassBlurd(keys=['image'], prob=1.0,
                               sigma=self.config['parameters']['transforms']['low_pass_sigma_range']) if
            self.config['parameters']['transforms']['low_pass_sigma_range'] else None,
            RandomHighPassSharpend(keys=['image'], prob=1.0,
                                   sigma=self.config['parameters']['transforms']['high_pass_sigma_range'],
                                   sigma2=self.config['parameters']['transforms']['high_pass_sigma2_range']) if
            self.config['parameters']['transforms']['high_pass_sigma_range'] else None,
            RandomGaussianNoised(keys=['image'], prob=1.0,
                                 sigma=self.config['parameters']['transforms']['noise_sigma_range']) if
            self.config['parameters']['transforms']['noise_sigma_range'] else None,
            Compose([
                RandomLowPassBlurd(keys=['image'], prob=1.0,
                                   sigma=self.config['parameters']['transforms']['low_pass_sigma_range']),
                RandomHighPassSharpend(keys=['image'], prob=1.0,
                                       sigma=self.config['parameters']['transforms']['high_pass_sigma_range'],
                                       sigma2=self.config['parameters']['transforms']['high_pass_sigma2_range']),
                RandomGaussianNoised(keys=['image'], prob=1.0,
                                     sigma=self.config['parameters']['transforms']['noise_sigma_range'])
            ]) if self.config['parameters']['transforms']['combine_transforms'] else None,
            Identityd(keys=['image'])]

        image_transforms = [x for x in image_transforms if x is not None]

        # define the data transforms
        train_transforms = Compose([LoadImaged(keys=keys + ['labels'], reader=MrcReader()),
                                    LoadImaged(keys=['distances'],
                                               reader=NumpyReader(npz_keys='data', channel_dim=0)),
                                    EnsureChannelFirstd(keys=keys + ['labels'], channel_dim='no_channel'),
                                    ClipIntensityd(keys=['distances'], a_min=-5, a_max=5),
                                    ScaleIntensityRanged(keys, a_min=self.config['parameters']['data']['min'],
                                                         a_max=self.config['parameters']['data']['max'], b_min=0,
                                                         b_max=1, clip=True),
                                    SpatialPadd(keys=keys + ['labels', 'distances'],
                                                spatial_size=self.config['parameters']['data']['patch_size']),
                                    OneOf(transforms=image_transforms),
                                    RandZoomd(keys=keys + ['labels', 'distances'], prob=0.8,
                                              min_zoom=self.config['parameters']['transforms']['zoom'][0],
                                              max_zoom=self.config['parameters']['transforms']['zoom'][1],
                                              padding_mode='constant') if 'zoom' in self.config['parameters'][
                                        'transforms'] and self.config['parameters']['transforms'][
                                                                              'zoom'] else Identityd(keys=['image']),
                                    SqueezeDimd(keys=['labels'],
                                                dim=0) if self.config['parameters']['network']['out_channels'] > 1
                                    else Identityd(keys=['labels']),
                                    RandScaleIntensityd(['image'], prob=0.8,
                                                        factors=self.config['parameters']['transforms'][
                                                            'scale_intensity_factors'])
                                    if self.config['parameters']['transforms']['scale_intensity_factors'] else
                                    Identityd(keys=['image']),
                                    NormalizeIntensityd(keys=keys,
                                                        subtrahend=self.config['parameters']['data']['mean'],
                                                        divisor=self.config['parameters']['data']['std']),
                                    EnsureTyped(keys=keys + ['distances'],
                                                data_type='tensor', dtype=torch.float),
                                    EnsureTyped(keys=['labels'], data_type='tensor', dtype=torch.float
                                    if self.config['parameters']['network']['out_channels'] == 1 else torch.long)])

        if self.config['hyper_parameters']['cache_rate'] > 0:
            # cached datasets - 10x faster than regular datasets
            rate = self.config['hyper_parameters']['cache_rate']
            if 'replace_rate' in self.config['hyper_parameters']:
                replace_rate = self.config['hyper_parameters']['replace_rate']
                self.train_ds = SmartCacheDataset(data=train_files, transform=train_transforms,
                                                  replace_rate=replace_rate,
                                                  num_init_workers=1, num_replace_workers=1, shuffle=True,
                                                  seed=self.trainer.global_rank, cache_rate=rate)
                self.val_ds = SmartCacheDataset(data=val_files, transform=train_transforms, replace_rate=replace_rate,
                                                num_init_workers=1, num_replace_workers=1, shuffle=True,
                                                seed=self.trainer.global_rank, cache_rate=rate)
            else:
                self.train_ds = CacheDataset(data=train_files, transform=train_transforms, num_workers=10,
                                             cache_num=len(train_files), cache_rate=rate, copy_cache=False)
                self.val_ds = CacheDataset(data=val_files, transform=train_transforms, num_workers=4,
                                           cache_num=len(val_files), cache_rate=rate, copy_cache=False)
        else:
            self.train_ds = Dataset(data=train_files, transform=train_transforms)
            self.val_ds = Dataset(data=val_files, transform=train_transforms)

    def train_dataloader(self):
        if self.config['hyper_parameters']['cache_rate'] > 0:
            train_loader = ThreadDataLoader(self.train_ds, num_workers=0, batch_size=self.batch_size, shuffle=True,
                                            buffer_size=80, collate_fn=list_data_collate, pin_memory=True,
                                            drop_last=True)
        else:
            train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=list_data_collate,
                                      shuffle=False if self.sync_dist else True, num_workers=10,
                                      persistent_workers=True, worker_init_fn=worker_init_fn,
                                      pin_memory=False, drop_last=True)
        return train_loader

    def val_dataloader(self):
        if self.config['hyper_parameters']['cache_rate'] > 0:
            val_loader = ThreadDataLoader(self.val_ds, num_workers=0, batch_size=self.batch_size, buffer_size=1,
                                          collate_fn=list_data_collate, pin_memory=True)
        else:
            val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=list_data_collate,
                                    num_workers=4, persistent_workers=True, worker_init_fn=worker_init_fn,
                                    pin_memory=False)
        return val_loader

    def configure_optimizers(self):
        lr = self.lr
        if 'unfreeze_backbone' in self.config['parameters']['network'] and self.config['parameters']['network'][
            'unfreeze_backbone']:
            if self.optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(list(self._model_backbone.parameters()) + list(self._model.parameters()),
                                            lr, momentum=self.momentum, weight_decay=self.weight_decay)
            elif self.optimizer_name == 'adam':
                optimizer = torch.optim.Adam(list(self._model_backbone.parameters()) + list(self._model.parameters()),
                                             lr, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.AdamW(list(self._model_backbone.parameters()) + list(self._model.parameters()),
                                              lr, weight_decay=self.weight_decay)
        elif 'unfreeze_decoder' in self.config['parameters']['network'] and self.config['parameters']['network'][
            'unfreeze_decoder']:
            for param in self._model_backbone.encoder.parameters():
                param.requires_grad = False

            if self.optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(
                    list(self._model_backbone.parameters()) + list(self._model.parameters()),
                    lr, momentum=self.momentum, weight_decay=self.weight_decay)
            elif self.optimizer_name == 'adam':
                optimizer = torch.optim.Adam(
                    list(self._model_backbone.parameters()) + list(self._model.parameters()),
                    lr, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.AdamW(
                    list(self._model_backbone.parameters()) + list(self._model.parameters()),
                    lr, weight_decay=self.weight_decay)
        else:
            for param in self._model_backbone.parameters():
                param.requires_grad = False

            if self.optimizer_name == 'sgd':
                optimizer = torch.optim.SGD(self._model.parameters(),
                                            lr, momentum=self.momentum, weight_decay=self.weight_decay)
            elif self.optimizer_name == 'adam':
                optimizer = torch.optim.Adam(self._model.parameters(),
                                             lr, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.AdamW(self._model.parameters(),
                                              lr, weight_decay=self.weight_decay)

        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=self.trainer.estimated_stepping_batches,
                               anneal_strategy='cos')
        lr_schedulers = {'scheduler': scheduler,
                         'monitor': 'train_loss',
                         'interval': 'step',
                         'frequency': 1}
        return [optimizer], [lr_schedulers]

    def training_step(self, batch, batch_idx):
        inputs, labels, distance = batch['image'], batch['labels'], batch['distances']
        if self.use_noisy_input:
            inputs = batch[random.choices(['image', 'noisy_image'], weights=[2, 1], k=1)[0]]
        labels_pred, distance_pred = self.forward(inputs)
        labels_loss = self.semantic_loss_function(labels_pred, labels)
        if 'use_dice_loss' in self.config['parameters']['network'] and self.config['parameters']['network'][
            'use_dice_loss']:
            if self.config['parameters']['network']['out_channels'] > 1:
                labels = torch.unsqueeze(labels, 1)
            labels_loss += self.semantic_loss_function2(labels_pred, labels)
        distances_loss = self.distance_loss_function(distance_pred, distance)

        if self.config['parameters']['network']['distance_prediction']:
            loss = labels_loss + distances_loss
            self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
            self.log('train_labels_loss', labels_loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
            self.log('train_distance_loss', distances_loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
        else:
            loss = labels_loss
            self.log('train_loss', labels_loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, distance = batch['image'], batch['labels'], batch['distances']
        if self.use_noisy_input:
            inputs = batch[random.choices(['image', 'noisy_image'], weights=[1, 1], k=1)[0]]
        labels_pred, distance_pred = self.forward(inputs)
        labels_loss = self.semantic_loss_function(labels_pred, labels)
        if 'use_dice_loss' in self.config['parameters']['network'] and self.config['parameters']['network'][
            'use_dice_loss']:
            if self.config['parameters']['network']['out_channels'] > 1:
                labels = torch.unsqueeze(labels, 1)
            labels_loss += self.semantic_loss_function2(labels_pred, labels)
        distances_loss = self.distance_loss_function(distance_pred, distance)

        if self.config['parameters']['network']['distance_prediction']:
            loss = labels_loss + distances_loss
            self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
            self.log('val_labels_loss', labels_loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
            self.log('val_distance_loss', distances_loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)
        else:
            self.log('val_loss', labels_loss, on_step=False, on_epoch=True, batch_size=self.batch_size,
                     sync_dist=self.sync_dist)

    def on_train_epoch_start(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == 0:
                print('Start smart cache')
                self.trainer.train_dataloader.dataset.start()

    def on_train_epoch_end(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == self.max_epochs:
                print('Shut down smart cache')
                self.trainer.train_dataloader.dataset.shutdown()
            else:
                print('New epoch update smart cache')
                self.trainer.train_dataloader.dataset.update_cache()

    def on_validation_epoch_start(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == 0:
                print('Start val smart cache')
                self.trainer.val_dataloaders.dataset.start()

    def on_validation_epoch_end(self):
        if self.config['hyper_parameters']['cache_rate'] > 0 and 'replace_rate' in self.config['hyper_parameters']:
            if self.current_epoch == self.max_epochs:
                print('Shut down val smart cache')
                self.trainer.val_dataloaders.dataset.shutdown()
            else:
                print('New epoch update val smart cache')
                self.trainer.val_dataloaders.dataset.update_cache()
