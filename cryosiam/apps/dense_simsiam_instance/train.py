import os
import yaml
import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from cryosiam.utils import parser_helper
from cryosiam.apps.dense_simsiam_instance import InstanceSegmentationModule


def main(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    gpu_devices = int(cfg['parameters']['gpu_devices'])
    nodes = int(cfg['parameters']['nodes'])
    is_distributed = gpu_devices > 1 or nodes > 1

    os.makedirs(cfg['log_dir'], exist_ok=True)

    # initialise the LightningModule
    checkpoint = torch.load(cfg['pretrained_model'], weights_only=False)
    backbone_config = checkpoint['hyper_parameters']['config']
    net = InstanceSegmentationModule(cfg, backbone_config)

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(cfg['log_dir'], 'model'),
                                          filename='model_best',
                                          every_n_epochs=cfg['hyper_parameters']['val_interval'],
                                          monitor='val_loss',
                                          save_top_k=1,
                                          save_last=True,
                                          verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    tb_logger = TensorBoardLogger(save_dir=os.path.join(cfg['log_dir']))

    # initialise Lightning's trainer.
    if is_distributed:
        if cfg['hyper_parameters']['cache_rate'] > 0:
            use_distributed_sampler = False
        else:
            use_distributed_sampler = True
        trainer = Trainer(accelerator="gpu", devices=gpu_devices, num_nodes=nodes,
                          strategy=DDPStrategy(find_unused_parameters=True),
                          max_epochs=cfg['hyper_parameters']['max_epochs'],
                          use_distributed_sampler=use_distributed_sampler,
                          logger=tb_logger,
                          check_val_every_n_epoch=cfg['hyper_parameters']['val_interval'],
                          callbacks=[checkpoint_callback, lr_monitor])
    else:
        trainer = Trainer(accelerator="gpu", devices=1,
                          max_epochs=cfg['hyper_parameters']['max_epochs'],
                          logger=tb_logger,
                          check_val_every_n_epoch=cfg['hyper_parameters']['val_interval'],
                          callbacks=[checkpoint_callback, lr_monitor])

    # train
    if cfg['continue_training']:
        trainer.fit(net, ckpt_path=os.path.join(cfg['log_dir'], 'model', 'last.ckpt'))
    else:
        trainer.fit(net)


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
