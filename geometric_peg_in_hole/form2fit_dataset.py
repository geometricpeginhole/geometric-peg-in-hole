import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from form2fit import config
from glob import glob
import os
import imageio
import numpy as np
import transforms3d as t3d
from tqdm import tqdm


class Form2fitDataset(Dataset):
    def __init__(self, task='black-floss', split='train') -> None:
        super().__init__()
        data_dir = os.path.join(config.benchmark_dir, 'train', task, split)
        sample_dirs = glob(os.path.join(data_dir, '*'))
        self.samples = []
        for sample_dir in tqdm(sample_dirs, desc=f'Loading {task} {split} data'):
            image = imageio.imread(os.path.join(sample_dir, 'init_color_height.png')).astype(np.float32)[None, None] / 255.0
            image = np.concatenate([image, image, image], axis=1)
            with open(os.path.join(sample_dir, 'init_pose.txt'), 'r') as f:
                init_pose = np.array([float(x) for x in f.read().split()], dtype=np.float32).reshape(4, 4)
            with open(os.path.join(sample_dir, 'final_pose.txt'), 'r') as f:
                final_pose = np.array([float(x) for x in f.read().split()], dtype=np.float32).reshape(4, 4)
            
            init_pos = init_pose[:3, 3]
            init_rot = np.reshape(init_pose[:3, :2], -1, order='F')
            final_pos = final_pose[:3, 3]
            final_rot = np.reshape(final_pose[:3, :2], -1, order='F')
            action = np.concatenate([init_pos, init_rot, final_pos, final_rot]).astype(np.float32)[None]
            sample = (image, 0, action)
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    