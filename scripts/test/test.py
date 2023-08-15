import glob
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import wandb


@hydra.main(version_base=None, config_path='../../conf', config_name='test')
def main(cfg: DictConfig):
    model_paths = glob.glob(os.path.join(cfg.model_glob))
    model_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    if len(model_paths) == 0:
        return

    output_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    with open_dict(cfg):
        cfg.model.proprioception = cfg.proprioception
        cfg.model.image_encoder.device = cfg.device
        cfg.model.action_decoder.device = cfg.device
        cfg.model.action_decoder.layers[0] = len(cfg.views) * cfg.model.image_encoder.output_dim
        if cfg.proprioception == 'with_prop':
            cfg.model.action_decoder.layers[0] += 14 if cfg.rotation_type == 'quat' else 18
        cfg.model.action_decoder.layers[-1] = 14 if cfg.rotation_type == 'quat' else 18
        cfg.model.rotation_type = cfg.rotation_type
        cfg.model.rotation_loss_type = cfg.rotation_loss_type

    with open_dict(cfg):
        cfg.eval.views = cfg.views

    model = hydra.utils.instantiate(cfg.model)
    wandb.login()
    wandb_run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )
    for model_path in model_paths:
        print(model_path)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        print(cfg.eval)
        eval_rate, rgbs = hydra.utils.instantiate(cfg.eval, env=hydra.utils.instantiate(cfg.env), model=model, device=cfg.device)
        rgbs = rgbs[:10]
        wandb_run.log({f'rgb': [wandb.Video(np.stack(rgb), fps=5, format="gif") for rgb in rgbs], 'eval_rate': eval_rate},
                      step=int(os.path.basename(model_path).split('_')[-1].split('.')[0]))

if __name__ == '__main__':
    main()