import glob
import os
import shutil
import torchvision
import argparse
import numpy as np
import torch
import tqdm
import lib.bc_environment
import cv2
import pybullet_planning as pp
# import models_r3m
import transforms3d as t3d
import lib.bc_dataset
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path='conf/visualize', config_name='base')
def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    dataset = hydra.utils.instantiate(cfg.dataset)
    n_success = 0
    n_total = 0
    for i_eval in tqdm.tqdm(range(cfg.n_evaluations)):
        env.reset()
        i_step = 0
        i_sample = 0
        while i_sample < len(dataset):
            env.update(False)
            if i_step % 24 == 0:
                rgb, robot_state, pred = dataset[i_sample]
                
                if cfg.delta:
                    next_robot_state_pred = lib.bc_environment.apply_delta(robot_state[0], pred[0])
                else:
                    next_robot_state_pred = pred[0]
                env.robot.set_right_eef(next_robot_state_pred[:3], next_robot_state_pred[3:7])
                env.robot.set_left_eef(next_robot_state_pred[7:10], next_robot_state_pred[10:14])
                i_sample += 1
            pp.step_simulation()
            i_step += 1
        
        if env.is_success():
            print(f'Success! {i_eval}')
        else:
            print(f'Failure! {i_eval}')
        n_success += int(env.is_success())
        n_total += 1
    pp.disconnect()
    return n_success / n_total


if __name__ == '__main__':
    main()