import torch
import collections

from cryosiam.networks.nets import (
    DenseSimSiam,
    SemanticHeads
)


def load_backbone_model(checkpoint_path, device="cuda:0"):
    """Load DenseSimSiam trained model from given checkpoint
    :param checkpoint_path: path to the checkpoint
    :type checkpoint_path: str
    :param device: on which device should the model be loaded, default is cuda:0
    :type device: str
    :return: DenseSimSiam model with laoded trained weights
    :rtype: cryosiam.networks.nets.DenseSimSiam
    """
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['hyper_parameters']['backbone_config']
    model_backbone = DenseSimSiam(block_type=config['parameters']['network']['block_type'],
                                  spatial_dims=config['parameters']['network']['spatial_dims'],
                                  n_input_channels=config['parameters']['network']['in_channels'],
                                  num_layers=config['parameters']['network']['num_layers'],
                                  num_filters=config['parameters']['network']['num_filters'],
                                  fpn_channels=config['parameters']['network']['fpn_channels'],
                                  no_max_pool=config['parameters']['network']['no_max_pool'],
                                  dim=config['parameters']['network']['dim'],
                                  pred_dim=config['parameters']['network']['pred_dim'],
                                  dense_dim=config['parameters']['network']['dense_dim'],
                                  dense_pred_dim=config['parameters']['network']['dense_pred_dim'],
                                  include_levels=config['parameters']['network']['include_levels_loss']
                                  if 'include_levels_loss' in config['parameters']['network'] else False,
                                  add_later_conv=config['parameters']['network']['add_fpn_later_conv']
                                  if 'add_fpn_later_conv' in config['parameters']['network'] else False,
                                  decoder_type=config['parameters']['network']['decoder_type']
                                  if 'decoder_type' in config['parameters']['network'] else 'fpn',
                                  decoder_layers=config['parameters']['network']['fpn_layers']
                                  if 'fpn_layers' in config['parameters']['network'] else 2)

    # model_backbone.load_state_dict(checkpoint['state_dict'])
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('_model_backbone.'):
            continue
        name = k.replace("_model_backbone.", '')  # remove `_model_backbone.`
        new_state_dict[name] = v
    model_backbone.load_state_dict(new_state_dict)
    model_backbone.eval()
    device = torch.device(device)
    model_backbone.to(device)
    return model_backbone


def load_prediction_model(checkpoint_path, device="cuda:0"):
    """Load SemanticHeads trained model from given checkpoint
    :param checkpoint_path: path to the checkpoint
    :type checkpoint_path: str
    :param device: on which device should the model be loaded, default is cuda:0
    :type device: str
    :return: InstanceHeads model with loaded trained weights
    :rtype: cryosiam.networks.nets.InstanceHeads
    """
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['hyper_parameters']['config']
    model = SemanticHeads(n_input_channels=config['parameters']['network']['dense_dim'],
                          num_classes=config['parameters']['network']['out_channels'],
                          spatial_dims=config['parameters']['network']['spatial_dims'])

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('_model.'):
            continue
        name = k.replace("_model.", '')  # remove `_model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    device = torch.device(device)
    model.to(device)
    return model
