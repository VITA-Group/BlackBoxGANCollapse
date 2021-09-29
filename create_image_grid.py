#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import glob
import numpy as np
import operator
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def save_in_image_grid(save_dir, read_dir)
    save_images_dir = 'sampling/monte_carlo_sampling_1m/neighbors/0.2/664058/collision'
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)
    img_lst = []
    for i in range(len(files)):
        img = mpimg.imread(os.path.join(read_dir, files[i]))
        img_lst.append(img)

    image_stack = np.stack(img_lst,axis=0)
    print(image_stack.shape)
    image_grid = create_image_grid(image_stack, grid_size=(10,5))
    print(image_grid.shape)
    mpimg.imsave(os.path.join(save_images_dir, '{}_{}.png'.format(read_dir.split('/')[-1], 
                              read_dir.split('/')[-1])), image_grid)


# In[6]:


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', type=str, required=True, help='source')
    parser.add_argument('--save_dir', type=str, required=True, help='the pause duration for catch action')
    return parser

if __name__ == '__main__':
    parser = default_parser()
    args = parser.parse_args()

    read_dir = args.read_dir
    save_dir = args.save_dir
    save_in_image_grid(save_dir, read_dir)

