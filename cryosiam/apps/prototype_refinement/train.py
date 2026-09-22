import os
import yaml
import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from cryosiam.utils import parser_helper
from cryosiam.apps.prototype_refinement.module import PrototypeRefinementModule


def _read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _source_checkpoint_path(config):
    if config.get('fine_tune_model'):
        return config['fine_tune_model']
    if config.get('general_model'):
        return config['general_model']
    if config.get('pretrained_model'):
        return config['pretrained_model']
    if config.get('pretrained_dense_simsiam_model'):
        return config['pretrained_dense_simsiam_model']
    raise ValueError(
        'Provide fine_tune_model, general_model/pretrained_model, or '
        'pretrained_dense_simsiam_model.')


def _looks_like_dense_backbone_config(config):
    if not isinstance(config, dict):
        return False

    network = config.get('parameters', {}).get('network', {})
    required = {'block_type',
                'in_channels',
                'spatial_dims',
                'num_layers',
                'num_filters',
                'no_max_pool',
                'fpn_channels',
                'dim',
                'pred_dim',
                'dense_dim',
                'dense_pred_dim'}
    return required.issubset(network)


def _load_dense_backbone_config(config):
    """Recover the DenseSimSiam architecture config from config/checkpoint."""
    configured = config.get('dense_backbone_config')

    if isinstance(configured, dict):
        return configured

    if configured:
        dense_config = _read_yaml(configured)
        if not _looks_like_dense_backbone_config(dense_config):
            raise ValueError(
                f'dense_backbone_config does not contain the expected '
                f'DenseSimSiam network settings: {configured}')
        return dense_config

    source_checkpoint = _source_checkpoint_path(config)
    checkpoint = torch.load(
        source_checkpoint, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})

    dense_config = hparams.get('dense_backbone_config')
    if _looks_like_dense_backbone_config(dense_config):
        return dense_config

    source_config = hparams.get('config')
    if _looks_like_dense_backbone_config(source_config):
        return source_config

    raise RuntimeError(
        'Could not recover dense_backbone_config from the source checkpoint. '
        'Set top-level "dense_backbone_config" in the training YAML to the '
        'original DenseSimSiam YAML file.')


def _requested_devices(config):
    parameters = config.get('parameters', {})
    devices = int(parameters.get('gpu_devices', 1))
    nodes = int(parameters.get('nodes', 1))

    if devices < 1:
        raise ValueError('parameters.gpu_devices must be >= 1.')
    if nodes < 1:
        raise ValueError('parameters.nodes must be >= 1.')
    return devices, nodes


def _resume_checkpoint(config, checkpoint_dir):
    if not bool(config.get('continue_training', False)):
        return None

    explicit = config.get('resume_checkpoint')
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(
                f'resume_checkpoint does not exist: {explicit}')
        return explicit

    last_checkpoint = os.path.join(checkpoint_dir, 'last.ckpt')
    if not os.path.isfile(last_checkpoint):
        raise FileNotFoundError(
            'continue_training=True, but no last checkpoint was found at '
            f'{last_checkpoint}. Set resume_checkpoint explicitly or set '
            'continue_training: false.')
    return last_checkpoint


def main(config_file_path):
    """Unified entrypoint for prototype-refinement training and adaptation."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CryoSiam prototype-refinement training requires a CUDA-enabled '
            'PyTorch installation.')

    config = _read_yaml(config_file_path)
    if not isinstance(config, dict):
        raise ValueError(
            f'Training config must be a YAML mapping: {config_file_path}')
    if not config.get('log_dir'):
        raise ValueError('Set log_dir in the training config.')

    partial_labels = bool(config.get('partial_labels', False))
    learning_mode = ('PU / partial-label learning'
                     if partial_labels
                     else 'standard fully supervised learning')

    os.makedirs(config['log_dir'], exist_ok=True)
    checkpoint_dir = os.path.join(config['log_dir'], 'model')
    os.makedirs(checkpoint_dir, exist_ok=True)

    dense_backbone_config = _load_dense_backbone_config(config)
    net = PrototypeRefinementModule(config, dense_backbone_config)

    default_monitor = ('val_annotated_recall_mean' if partial_labels
                       else 'val_dice_mean')
    monitor = str(config.get('checkpoint_monitor', default_monitor))
    monitor_mode = str(config.get('checkpoint_mode', 'max')).lower()
    if monitor_mode not in ('min', 'max'):
        raise ValueError(
            f'checkpoint_mode must be "min" or "max", got {monitor_mode!r}.')

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename='model_best',
                                          monitor=monitor,
                                          mode=monitor_mode,
                                          save_top_k=1,
                                          save_last=True,
                                          auto_insert_metric_name=False,
                                          verbose=True)
    lr_monitor = LearningRateMonitor(
        logging_interval='step', log_momentum=True)
    tb_logger = TensorBoardLogger(save_dir=config['log_dir'])

    devices, nodes = _requested_devices(config)
    world_size = devices * nodes
    strategy = (DDPStrategy(find_unused_parameters=True)
                if world_size > 1 else 'auto')

    hyper = config.get('hyper_parameters', {})
    max_epochs = int(hyper.get('max_epochs', 1))
    val_interval = int(hyper.get('val_interval', 1))
    if max_epochs < 1:
        raise ValueError('hyper_parameters.max_epochs must be >= 1.')
    if val_interval < 1:
        raise ValueError('hyper_parameters.val_interval must be >= 1.')

    resume_checkpoint = _resume_checkpoint(config, checkpoint_dir)

    print('\nPrototype-refinement training')
    print(f'  config: {config_file_path}')
    print(f'  learning mode: {learning_mode}')
    print(f'  active classes: {config.get("class_names")}')
    print(f'  source checkpoint: {_source_checkpoint_path(config)}')
    print(f'  log dir: {config["log_dir"]}')
    print(f'  checkpoints: {checkpoint_dir}')
    print(f'  monitor: {monitor} ({monitor_mode})')
    print(f'  devices per node: {devices}')
    print(f'  nodes: {nodes}')
    print(f'  strategy: {strategy}')
    print(f'  max epochs: {max_epochs}')
    if resume_checkpoint:
        print(f'  resuming training state from: {resume_checkpoint}')

    trainer = Trainer(accelerator='gpu',
                      devices=devices,
                      num_nodes=nodes,
                      strategy=strategy,
                      max_epochs=max_epochs,
                      check_val_every_n_epoch=val_interval,
                      use_distributed_sampler=True,
                      logger=tb_logger,
                      callbacks=[checkpoint_callback, lr_monitor],
                      default_root_dir=config['log_dir'])

    trainer.fit(net, ckpt_path=resume_checkpoint)

    if trainer.is_global_zero:
        print('\nTraining finished.')
        print(f'  best checkpoint: {checkpoint_callback.best_model_path}')
        print(f'  last checkpoint: {checkpoint_callback.last_model_path}')


if __name__ == '__main__':
    parser = parser_helper('Prototype refinement training / adaptation')
    args = parser.parse_args()
    main(args.config_file)
