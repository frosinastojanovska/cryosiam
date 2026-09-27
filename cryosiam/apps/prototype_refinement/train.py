import os
import yaml
import torch
import copy
import json
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from cryosiam.utils import parser_helper
from cryosiam.apps.prototype_refinement import PrototypeRefinementModule


def _load_dense_config(cfg):
    source_path = cfg.get('fine_tune_model') or cfg.get('pretrained_model')
    if not source_path:
        raise ValueError('Specify pretrained_model or fine_tune_model.')
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
    dense = checkpoint.get('hyper_parameters', {}).get('dense_backbone_config')
    if not isinstance(dense, dict) or 'parameters' not in dense:
        raise ValueError('Source checkpoint has no dense_backbone_config.')
    return dense


def _validate_run(cfg, checkpoint_dir, resume_path):
    """Prevent accidentally resuming/overwriting a different experiment."""
    if cfg.get('fine_tune_model') and cfg.get('pretrained_model'):
        raise ValueError(
            'Use fine_tune_model for prototype-bank fine-tuning, or pretrained_model for a new bank; not both.')
    if not cfg.get('class_names'):
        raise ValueError('class_names must not be empty.')
    if cfg.get('resume_checkpoint') and not cfg.get('continue_training', False):
        raise ValueError('Set continue_training: true to resume an existing run.')
    os.makedirs(checkpoint_dir, exist_ok=True)
    marker = os.path.join(cfg['log_dir'], 'supervised_run_manifest.json')
    fingerprint = {key: value for key, value in cfg.items()
                   if key not in ('log_dir', 'continue_training', 'resume_checkpoint',
                                  'checkpoint_monitor', 'checkpoint_mode')}
    if os.path.isfile(marker):
        with open(marker) as stream:
            if json.load(stream) != fingerprint:
                raise ValueError('Existing log_dir belongs to a different supervised experiment. Use a new log_dir.')
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        old = torch.load(resume_path, map_location='cpu', weights_only=False)
        previous = old.get('hyper_parameters', {}).get('config', {})
        previous_active = previous.get('fine_tune_class_names', previous.get('class_names'))
        if list(previous_active or []) != list(cfg['class_names']):
            raise ValueError('Resume class selection differs.')
        if cfg.get('fine_tune_model') != previous.get('fine_tune_model'):
            raise ValueError('Resume fine_tune_model differs.')
        if cfg.get('patches_folder') != previous.get('patches_folder'):
            raise ValueError('Resume patches_folder differs.')
    elif any(os.path.isfile(os.path.join(checkpoint_dir, name)) for name in
             ('last.ckpt', 'model_best.ckpt', 'last_model.ckpt')):
        raise ValueError('Existing checkpoint would be overwritten. Use a fresh log_dir or continue_training: true.')
    if not os.path.isfile(marker):
        with open(marker, 'w') as stream:
            json.dump(fingerprint, stream, indent=2, sort_keys=True)


def main(config_file_path):
    if not torch.cuda.is_available():
        raise RuntimeError('CryoSiam training requires CUDA.')
    with open(config_file_path) as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('Expected a YAML mapping.')
    if not cfg.get('pretrained_model') and not cfg.get('fine_tune_model'):
        raise ValueError('Specify fine_tune_model or pretrained_model.')
    if not cfg.get('log_dir'):
        raise ValueError('Specify log_dir.')
    checkpoint_dir = os.path.join(cfg['log_dir'], 'model')
    resume_path = (cfg.get('resume_checkpoint') or os.path.join(checkpoint_dir, 'last.ckpt')
                   if cfg.get('continue_training', False) else None)
    _validate_run(cfg, checkpoint_dir, resume_path)
    net = PrototypeRefinementModule(copy.deepcopy(cfg), _load_dense_config(cfg))
    if resume_path:
        previous = torch.load(resume_path, map_location='cpu', weights_only=False)
        previous_cfg = previous.get('hyper_parameters', {}).get('config', {})
        if list(previous_cfg.get('class_names', [])) != net.class_names:
            raise ValueError('Checkpoint class layout differs from the current model.')
        if int(previous_cfg.get('k_per_class', 5)) != net.k_per_class:
            raise ValueError('Checkpoint prototype count differs from the current model.')
    monitor = str(cfg.get('checkpoint_monitor', 'val_loss'))
    mode = str(cfg.get('checkpoint_mode', 'min' if monitor.endswith('loss') else 'max')).lower()
    if mode not in ('min', 'max'):
        raise ValueError('checkpoint_mode must be min or max.')
    interval = cfg['hyper_parameters']['val_interval']
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, filename='model_best', every_n_epochs=interval,
        monitor=monitor, mode=mode, save_top_k=1, verbose=True)
    last_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, filename='last_model', every_n_epochs=interval,
        save_top_k=0, save_last=True, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    logger = TensorBoardLogger(save_dir=cfg['log_dir'])
    devices = int(cfg['parameters']['gpu_devices'])
    nodes = int(cfg['parameters']['nodes'])
    common = dict(accelerator='gpu', devices=devices, num_nodes=nodes,
                  max_epochs=cfg['hyper_parameters']['max_epochs'], logger=logger,
                  check_val_every_n_epoch=interval,
                  callbacks=[checkpoint_callback, last_checkpoint, lr_monitor])
    if devices > 1 or nodes > 1:
        common.update(strategy=DDPStrategy(find_unused_parameters=True), use_distributed_sampler=True)
    trainer = Trainer(**common)
    print(f'Fully supervised prototype refinement: {net.active_class_names}')
    trainer.fit(net, ckpt_path=resume_path)


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
