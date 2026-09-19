import collections
import torch

from cryosiam.networks.nets import DenseSimSiam, PrototypeSimilarityFPN


def load_encoder(checkpoint_path, device='cuda:0'):
    """Load DenseSimSiam encoder from PrototypeMatchingModule checkpoint.

    NOTE: the module's attribute is self._backbone (not self._encoder --
    that was the naming in an earlier version of PrototypeMatchingModule),
    so this looks for '_backbone.*' keys in the checkpoint's state_dict.
    Function name kept as load_encoder for backward compatibility with
    existing callers."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    dense_config = checkpoint['hyper_parameters']['dense_backbone_config']
    net_cfg = dense_config['parameters']['network']

    model = DenseSimSiam(block_type=net_cfg['block_type'],
                         n_input_channels=net_cfg['in_channels'],
                         spatial_dims=net_cfg['spatial_dims'],
                         num_layers=net_cfg['num_layers'],
                         num_filters=net_cfg['num_filters'],
                         no_max_pool=net_cfg['no_max_pool'],
                         fpn_channels=net_cfg['fpn_channels'],
                         dim=net_cfg['dim'],
                         pred_dim=net_cfg['pred_dim'],
                         dense_dim=net_cfg['dense_dim'],
                         dense_pred_dim=net_cfg['dense_pred_dim'],
                         decoder=False)

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('_backbone.'):
            continue
        new_state_dict[k.replace('_backbone.', '')] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(torch.device(device))
    return model


def load_decoder(checkpoint_path, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    dense_config = checkpoint['hyper_parameters']['dense_backbone_config']
    net_cfg = dense_config['parameters']['network']
    decoder_cfg = config['parameters']['network']

    decoder_keys = {k.replace('_decoder.', ''): v
                    for k, v in checkpoint['state_dict'].items() if k.startswith('_decoder.')}
    if 'lat5.weight' not in decoder_keys:
        raise RuntimeError(
            'No "_decoder.lat5.weight" found in checkpoint state_dict -- cannot infer '
            'out_channels. Available _decoder.* keys: '
            f'{sorted(decoder_keys.keys())[:10]}')

    embed_dim = decoder_keys['lat5.weight'].shape[0]
    config_embed_dim = decoder_cfg.get('embed_dim', decoder_cfg.get('out_channels'))
    if config_embed_dim is not None and int(config_embed_dim) != embed_dim:
        print(f'  WARNING: config embed_dim/out_channels={config_embed_dim} but the '
              f'checkpoint\'s actual weights imply {embed_dim} -- using the '
              f'checkpoint-derived value, config value ignored.')
    print(f'  Inferred decoder out_channels from checkpoint weights: {embed_dim}')

    model = PrototypeSimilarityFPN(feat_channels=net_cfg['num_filters'],
                                   out_channels=embed_dim,
                                   use_context_head=decoder_cfg.get('use_context_head', False),
                                   predict_at_c3=decoder_cfg.get('predict_at_c3', False))

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('_decoder.'):
            continue
        new_state_dict[k.replace('_decoder.', '')] = v
    miss, unexp = model.load_state_dict(new_state_dict, strict=False)
    if miss:
        print(f'  [decoder] missing:    {miss[:3]}...')
    if unexp:
        print(f'  [decoder] unexpected: {unexp[:3]}...')

    model.eval()
    model.to(torch.device(device))
    return model
