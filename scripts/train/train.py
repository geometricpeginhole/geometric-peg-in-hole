import json
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn
import torch.nn.functional
import wandb
from torch.utils.data.dataloader import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict


@hydra.main(version_base=None, config_path='../../conf', config_name='train')
def main(cfg: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    torch.manual_seed(cfg.seed)

    with open_dict(cfg):
        cfg.eval.views = cfg.views
        cfg.env.variant = OmegaConf.load(os.path.join('conf', 'env', 'variant',
                                                      f'{os.path.basename(cfg.data_dir).split("_")[-1]}.yaml'))
    with open_dict(cfg):
        assert cfg.rotation_type in ['quat', '6d']
        assert cfg.cache_path is not None
        cfg.dataloader.dataset.data_dir = cfg.data_dir
        cfg.dataloader.dataset.views = cfg.views
        cfg.dataloader.dataset.rotation_type = cfg.rotation_type
        cfg.dataloader.dataset.cache_path = cfg.cache_path
        
    dataloader = hydra.utils.instantiate(cfg.dataloader)
    dataloader_iter = iter(dataloader)
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
    model = hydra.utils.instantiate(cfg.model)
    optim = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    step_tqdm = tqdm.tqdm(range(1, cfg.train.n_steps + 1))
    train_losses = {}
    data_time = 0
    model_time = 0
    print(cfg)

    with open_dict(cfg):
        cfg.wandb.tags = os.path.basename(output_dir).split('-')
        cfg.wandb.group = '-'.join(os.path.basename(output_dir).split('-')[2:4])
    wandb.login()
    wandb_run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )

    for step in step_tqdm:
        data_start = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        rgb, robot_state, label = batch

        # plt.imshow(rgb[0][0, 0].permute(1, 2, 0).cpu().numpy())
        # plt.show()
        # plt.imshow(rgb[1][0, 0].permute(1, 2, 0).cpu().numpy())
        # plt.show()
        # plt.imshow(rgb[2][0, 0].permute(1, 2, 0).cpu().numpy())
        # plt.show()

        model.train()
        rgb = rgb.to(cfg.device, non_blocking=True)
        # rgb = [view.to(cfg.device, non_blocking=True) for view in rgb]
        robot_state = robot_state.to(cfg.device, non_blocking=True)
        label = label.to(cfg.device, non_blocking=True)
        torch.cuda.synchronize()

        data_time = time.time() - data_start
        model_start = time.time()
        step_tqdm.set_postfix({'data_time': data_time, 'model_time': model_time})

        output = model(rgb, robot_state)
        # print(output[0, 0])

        optim.zero_grad()
        losses = model.loss(output, label)
        # print(output, label, loss)
        losses['loss'].backward()
        optim.step()
        model_time = time.time() - model_start
        step_tqdm.set_postfix({'data_time': data_time, 'model_time': model_time})

        # print(losses, train_losses)
        train_losses = {k: (train_losses[k] + [v.item()] if k in train_losses else [v.item()]) for k, v in losses.items()}
        if step % cfg.train.f_log == 0:
            train_losses = {f'train/{k}': np.mean(v) for k, v in train_losses.items()}
            wandb_run.log(train_losses, step=step)
            train_losses = {}

        if step % cfg.train.f_eval == 0:
            eval_acc, rgbs = hydra.utils.instantiate(cfg.eval, env=cfg.env, model=model, device=cfg.device, obs_frames=cfg.dataloader.dataset.obs_frames)
            rgbs = rgbs[:10]
            wandb_run.log({f'rgb': [wandb.Video(np.stack(rgb), fps=10, format="gif") for rgb in rgbs]}, step=step)
            wandb_run.log({'eval/success_rate': eval_acc}, step=step)
            print(f'Eval success rate: {eval_acc}')
        
        if step % cfg.train.f_save == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'bc_model_{step}.pth'))

if __name__ == '__main__':
    main()