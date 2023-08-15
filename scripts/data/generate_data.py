import argparse
import json
import shutil
import pybullet as p
import os
import numpy as np
import pybullet_planning as pp
import cv2
import matplotlib.pyplot as plt
import transforms3d as t3d
import geometric_peg_in_hole.dual_arm_robot
import geometric_peg_in_hole.realsense_camera
import geometric_peg_in_hole.bc_environment
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../../conf', config_name='generate_data')
def main(cfg):
    # input(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    env = hydra.utils.instantiate(cfg.env)

    os.makedirs(os.path.join(output_dir, f'top'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'left_cam'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'right_cam'), exist_ok=True)
    i_sample = 0
    seed = 1000 + cfg.start_idx
    while i_sample < cfg.n_samples:
        env.reset(seed)
        n_steps = 0
        tops = []
        lefts = []
        rights = []
        robot_observations = []
        while not env.is_done():
            env.update(True)
            if n_steps % 24 == 0:
                tops.append(env.camera_top.get_obs()['rgb'])
                lefts.append(env.camera_left.get_obs()['rgb'])
                rights.append(env.camera_right.get_obs()['rgb'])
                robot_observations.append(env.robot.get_obs())

            pp.step_simulation()
            n_steps += 1
        
        if env.is_success():
            os.makedirs(os.path.join(output_dir), exist_ok=True)
            for i_rgb, (top, left, right) in enumerate(zip(tops, lefts, rights)):
                cv2.imwrite(os.path.join(output_dir, f'top/{i_sample+cfg.start_idx:06d}_{i_rgb:06d}.png'), top[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(output_dir, f'left_cam/{i_sample+cfg.start_idx:06d}_{i_rgb:06d}.png'), left[:, :, [2, 1, 0]])
                cv2.imwrite(os.path.join(output_dir, f'right_cam/{i_sample+cfg.start_idx:06d}_{i_rgb:06d}.png'), right[:, :, [2, 1, 0]])
            with open(os.path.join(output_dir, f'{i_sample+cfg.start_idx:06d}_robot.json'), 'w') as f:
                json.dump(robot_observations, f, indent=1)
            print(f'Success {i_sample+cfg.start_idx}')
            i_sample += 1
        else:
            print(f'Failure {i_sample+cfg.start_idx}')
        seed += 1

if __name__ == "__main__":
    main()