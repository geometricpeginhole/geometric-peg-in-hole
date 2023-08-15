import glob
import os
import imageio
import numpy as np
import cv2
# import models_r3m
import transforms3d as t3d
import lib.bc_dataset
import tqdm

def main():
    datas = glob.glob('data/*')
    datas = list(filter(lambda x: '1000' not in os.path.basename(x), datas))
    datas = list(filter(lambda x: '10000' not in os.path.basename(x), datas))
    datas = list(filter(lambda x: '2' not in os.path.basename(x), datas))
    datas = list(filter(lambda x: '4mm' not in os.path.basename(x), datas))
    print(datas)

    for data in tqdm.tqdm(datas):
        for i_episode in range(10):
            top_paths = glob.glob(os.path.join(data, 'top', f'{i_episode:06d}_*'))
            top_paths = sorted(top_paths, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            left_paths = glob.glob(os.path.join(data, 'left', f'{i_episode:06d}_*'))
            left_paths = sorted(left_paths, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            right_paths = glob.glob(os.path.join(data, 'right', f'{i_episode:06d}_*'))
            right_paths = sorted(right_paths, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            images = []
            for top_path, left_path, right_path in zip(top_paths, left_paths, right_paths):
                top = cv2.imread(top_path)
                left = cv2.imread(left_path)
                right = cv2.imread(right_path)
                image = np.concatenate([top, left, right], axis=1)
                # image = top
                image = image[..., ::-1]
                images.append(image)
            
            # create mp4
            imageio.mimsave(f'visualize/{os.path.basename(data)}_{i_episode}.gif', images, fps=5)


if __name__ == '__main__':
    main()