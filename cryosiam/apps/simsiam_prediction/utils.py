import torch
import collections
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
