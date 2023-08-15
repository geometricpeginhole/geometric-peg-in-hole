import imageio
import glob
import numpy as np
import tqdm


if __name__ == '__main__':
    gifs = glob.glob('viz_eval/*.gif')
    for gif in tqdm.tqdm(gifs):
        images = imageio.mimread(gif)
        images = np.repeat(images, 4, axis=0)
        imageio.mimsave(gif.replace('.gif', '.mp4'), images, codec='libx264', quality=10, fps=20)