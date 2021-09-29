import os
import glob
import numpy as np
import operator
import matplotlib.image as mpimg
import re

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

save_images_dir = 'grid_images'
if not os.path.exists(save_images_dir):
    os.makedirs(save_images_dir)
img_path = 'Desktop/sorted_clustered_images'
files_lst = os.listdir(img_path)
files_lst.sort(key=lambda txt: grep(r"(\d+)\.(\d+)_(\d+)\.png", txt, 1))
print(files_lst)

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h, img_d = images.shape[0], images.shape[1], images.shape[2], images.shape[3]

    grid_w, grid_h = tuple(grid_size)

    grid = np.zeros([grid_h * img_h, grid_w * img_w] + [img_d], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w, :] = images[idx]
    return grid
save_
