import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.image_size = [512, 512]
_C.input_channels = 3

# encoder 
_C.encoder = CN()
_C.encoder.patch_size = 4
_C.encoder.embed_dim = 96
_C.encoder.depths = [2, 2, 6, 2]
_C.encoder.num_heads = [3, 6, 12, 24]
_C.encoder.mlp_ratio = 4
_C.encoder.qkv_bias = True
_C.encoder.drop_rate = 0
_C.encoder.drop_path_rate = 0.1
_C.encoder.ape = False

# decoder
_C.decoder = CN()
_C.decoder.feat_proj_dim = 320
_C.decoder.embed_dim = 180
_C.decoder.num_classes = 3


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
        
    for k, v in args.__dict__.items():
        config[k] = v
        
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
