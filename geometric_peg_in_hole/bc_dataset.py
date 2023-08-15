import time
import torch.utils.data
import glob
import numpy as np
import json
import tqdm
import transforms3d as t3d
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL.Image as Image
import torchvision.transforms.functional as TF


class BCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, delta, obs_frames, act_frames, views, transforms, rotation_type, in_memory, cache_path) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.delta = delta
        self.obs_frames = np.array(obs_frames)
        self.act_frames = np.array(act_frames)
        self.views = views

        self.transforms = transforms
        self.rotation_type = rotation_type
        assert self.rotation_type in ['quat', '6d']
        self.in_memory = in_memory

        self.images = []
        self.obs_indices = []
        self.robot_states = []
        self.actions = []
        json_paths = sorted(glob.glob(self.data_dir + '/*robot.json'))
        for i_sample, json_path in enumerate(json_paths):
            # if i_sample == 100:
            #     break
            with open(json_path, 'r') as f:
                robot_observations = json.load(f)
            sample_start_index = len(self.images)
            for i_obs in range(len(robot_observations)):
                obs_indices = np.clip(self.obs_frames + i_obs, 0, len(robot_observations) - 1)
                self.obs_indices.append(obs_indices + sample_start_index)

                images = []
                i_curr = i_obs
                images.append([])
                for view in self.views:
                    image_path = f'{self.data_dir}/{view}/{i_sample:06d}_{i_curr:06d}.png'
                    images[-1].append(image_path)

                # robot_states.append(get_action(i_curr - 1, robot_observations, self.rotation_type, False))
                right_action = np.concatenate([robot_observations[i_curr]['right_eef_pos'],
                                                robot_observations[i_curr]['right_eef_ori'],])
                left_action = np.concatenate([robot_observations[i_curr]['left_eef_pos'],
                                                robot_observations[i_curr]['left_eef_ori'],])
                
                if rotation_type == 'quat':
                    right_action = np.concatenate([robot_observations[i_curr]['right_eef_pos'],
                                                robot_observations[i_curr]['right_eef_ori']])
                    left_action = np.concatenate([robot_observations[i_curr]['left_eef_pos'],
                                                robot_observations[i_curr]['left_eef_ori']])
                elif rotation_type == '6d':
                    right_action = np.concatenate([robot_observations[i_curr]['right_eef_pos'],
                                                    np.reshape(t3d.quaternions.quat2mat(robot_observations[i_curr]['right_eef_ori'])[:3, :2], -1, 'F')])
                    left_action = np.concatenate([robot_observations[i_curr]['left_eef_pos'],
                                                    np.reshape(t3d.quaternions.quat2mat(robot_observations[i_curr]['left_eef_ori'])[:3, :2], -1, 'F')])
                # print(np.concatenate([right_action, left_action]) - get_action(i_curr - 1, robot_observations, self.rotation_type, False))

                # images = np.stack(images, axis=0).astype(np.float32)
                robot_states = np.concatenate([right_action, left_action]).astype(np.float32)

                actions = []
                actions.append(get_action(i_obs, robot_observations, self.rotation_type, self.delta))
                actions = np.stack(actions, axis=0).astype(np.float32)
                self.images.append(images)
                self.robot_states.append(robot_states)
                self.actions.append(actions)
        self.robot_states = np.stack(self.robot_states, axis=0)
        self.actions = np.stack(self.actions, axis=0)
        images = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=(len(self.images), len(self.images[0]), len(self.images[0][0]), 3, 224, 224))
        print(images.shape)
        for i in tqdm.tqdm(list(range(len(self.images)))):
            for j in range(len(self.images[i])):
                for k in range(len(self.images[i][j])):
                    image = Image.open(self.images[i][j][k]).convert('RGB')
                    image = TF.crop(image, 0, 0, 480, 480)
                    image = TF.resize(image, 224)
                    image = np.array(image)
                    image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.
                    images[i, j, k] = image
        self.images = images
                    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # image, robot_state, action = self.entries[index]
        obs_indices = self.obs_indices[index]
        images = self.images[obs_indices[:, None], 0, [list(range(len(self.views)))]]
        robot_state = self.robot_states[obs_indices]
        action = self.actions[index]
        # print(images.shape, robot_state.shape, action.shape)

        if isinstance(self.transforms, transforms.Compose):
            images = torch.from_numpy(images)
            images = self.transforms(images)
        return images, robot_state, action

def get_action(i_curr, robot_observations, rotation_type, delta):
    i_curr = max(0, min(i_curr, len(robot_observations) - 1))
    i_next = max(0, min(i_curr + 2, len(robot_observations) - 1))
    curr_state = robot_observations[i_curr]
    next_state = robot_observations[i_next]
    world_T_curr_right = t3d.affines.compose(curr_state['right_eef_pos'],
                                                t3d.quaternions.quat2mat(curr_state['right_eef_ori']),
                                                [1, 1, 1])
    world_T_next_right = t3d.affines.compose(next_state['right_eef_pos'],
                                                t3d.quaternions.quat2mat(next_state['right_eef_ori']),
                                                [1, 1, 1])
    world_T_curr_left = t3d.affines.compose(curr_state['left_eef_pos'],
                                            t3d.quaternions.quat2mat(curr_state['left_eef_ori']),
                                            [1, 1, 1])
    world_T_next_left = t3d.affines.compose(next_state['left_eef_pos'],
                                            t3d.quaternions.quat2mat(next_state['left_eef_ori']),
                                            [1, 1, 1])
    if delta:
        curr_T_next_right = np.linalg.inv(world_T_curr_right) @ world_T_next_right
        curr_T_next_left = np.linalg.inv(world_T_curr_left) @ world_T_next_left
        if rotation_type == 'quat':
            right_action = np.concatenate([curr_T_next_right[:3, 3],
                                            t3d.quaternions.mat2quat(curr_T_next_right[:3, :3])])
            left_action = np.concatenate([curr_T_next_left[:3, 3],
                                            t3d.quaternions.mat2quat(curr_T_next_left[:3, :3])])
        elif rotation_type == '6d':
            right_action = np.concatenate([curr_T_next_right[:3, 3], np.reshape(curr_T_next_right[:3, :2], -1, order='F')])
            left_action = np.concatenate([curr_T_next_left[:3, 3], np.reshape(curr_T_next_left[:3, :2], -1, order='F')])
        return np.concatenate([right_action, left_action])
    else:
        if rotation_type == 'quat':
            right_action = np.concatenate([next_state['right_eef_pos'],
                                        next_state['right_eef_ori']])
            left_action = np.concatenate([next_state['left_eef_pos'],
                                        next_state['left_eef_ori']])
        elif rotation_type == '6d':
            right_action = np.concatenate([next_state['right_eef_pos'],
                                            np.reshape(t3d.quaternions.quat2mat(next_state['right_eef_ori'])[:3, :2], -1, order='F')])
            left_action = np.concatenate([next_state['left_eef_pos'],
                                            np.reshape(t3d.quaternions.quat2mat(next_state['left_eef_ori'])[:3, :2], -1, order='F')])
        return np.concatenate([right_action, left_action])
