import glob
import cv2
import os
import numpy as np
import tqdm


if __name__ == '__main__':
    dirs = glob.glob('data/*/top')
    dirs = [dir for dir in dirs if '10000' not in dir]
    for dir in dirs:
        print(dir)
        all_frames = glob.glob(f'{dir}/*.png')
        first_frames = glob.glob(f'{dir}/*000000.png')
        os.makedirs(dir.replace('top', 'left_wrist'), exist_ok=True)
        os.makedirs(dir.replace('top', 'right_wrist'), exist_ok=True)
        for frame in tqdm.tqdm(all_frames):
            first_frame = '_'.join(frame.split('_')[:-1]) + '_000000.png'
            first_frame = cv2.imread(first_frame)
            # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            first_frame_left = first_frame[65:235, 50:220]
            first_frame_right = first_frame[110:280, 280:450]
            out_name = frame.split('/')[-1]
            cv2.imwrite(dir.replace('top', 'left_wrist') + f'/{out_name}', first_frame_left)
            cv2.imwrite(dir.replace('top', 'right_wrist') + f'/{out_name}', first_frame_right)
        # break