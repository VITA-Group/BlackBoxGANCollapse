import os
import glob
import numpy as np
import operator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

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



def main():
    parser = argparse.ArgumentParser(description='Random sampling')
    parser.add_argument('--src_dir', required=True, help='Source directory of images to read from', type=str)
    parser.add_argument('--dst_dir', required=True, help='Destination directory of image grid to save', type=str)
    args, other_args = parser.parse_known_args()

    files = os.listdir(args.src_dir)
    print(files)


    img_lst = []
    for i in range(len(files)):
        img = mpimg.imread(os.path.join(args.src_dir, files[i]))
        img_lst.append(img)

    image_stack = np.stack(img_lst, axis=0)
    print(image_stack.shape)
    image_grid = create_image_grid(image_stack, grid_size=(10,10))
    print(image_grid.shape)
    mpimg.imsave(args.dst_dir, image_grid)

if __name__ == "__main__":
    main()

