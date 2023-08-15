import argparse
import json
import time
import pybullet as p
import pybullet_planning as pp
import os
import numpy as np
import pybullet_planning as pp
import cv2
import matplotlib.pyplot as plt
import transforms3d as t3d
import lib.bc_environment
import hydra


@hydra.main(config_path='conf', config_name='display_obj')
def main(cfg):
    # print run dir
    env = hydra.utils.instantiate(cfg.env)
    for i in range(100):
        pp.step_simulation()
    for i_target, target_id in enumerate(env.target_ids):
        if i_target % 2 == 0:
            pp.set_pose(target_id, ([i_target // 2 * 0.09 - len(env.target_ids) // 4 * 0.09, -0.09 / 2, -0.5], pp.quat_from_euler([np.pi / 2, 0, 0])))
        else:
            pp.set_pose(target_id, ([i_target // 2 * 0.09 - len(env.target_ids) // 4 * 0.09, 0.09 / 2, -0.5], pp.quat_from_euler([np.pi / 2, 0, 0])))
    
    while True:
        pass
        # pp.step_simulation()
        # time.sleep(0.01)


if __name__ == '__main__':
    main()