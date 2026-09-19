import torch
import collections
from cryosiam.networks.nets import SimSiam


def load_backbone(checkpoint_path, contrastive=False, device='cuda:0'):
    """Load SimSiam trained model from given checkpoint."""
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
        name = k.replace('_model.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(torch.device(device))
    return model, config['parameters']['network']['dim']
