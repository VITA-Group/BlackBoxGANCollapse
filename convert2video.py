#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from natsort import natsorted
from moviepy.editor import *
import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[20]:


def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

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


# In[21]:


CC_path = 'reproducibility/CC'
CF_path = 'reproducibility/CF'
FC_path = 'reproducibility/FC'
FF_path = 'reproducibility/FF'
F_path = 'raw_images/ffhq'
C_path = 'raw_images/celebahq'
C_files_list = sorted(os.listdir(C_path))
F_files_list = sorted(os.listdir(F_path))
CC_files_list = sorted(os.listdir(CC_path))
CF_files_list = sorted(os.listdir(CF_path))
FC_files_list = sorted(os.listdir(FC_path))
FF_files_list = sorted(os.listdir(FF_path))


# In[25]:


save_images_dir = 'ffhq_grid_images'
if not os.path.exists(save_images_dir):
    os.makedirs(save_images_dir)
img_F_lst = []
img_CF_lst = []
img_FF_lst = []
for i in range(len(F_files_list)):
    img_F = cropND(mpimg.imread(os.path.join(F_path, F_files_list[i])), (800, 800))
    img_CF = cropND(mpimg.imread(os.path.join(CF_path, CF_files_list[i])), (800, 800))
    img_FF = cropND(mpimg.imread(os.path.join(FF_path, FF_files_list[i])), (800, 800))
    print(img_F.shape)
    print(img_CF.shape)
    print(img_FF.shape)
    img_F_lst.append(img_F)
    img_CF_lst.append(img_CF)
    img_FF_lst.append(img_FF)
image_stack = np.stack(img_F_lst+img_CF_lst+img_FF_lst,axis=0)
print(image_stack.shape)
image_grid = create_image_grid(image_stack, grid_size=(15,3))
print(image_grid.shape)
mpimg.imsave(os.path.join(save_images_dir, 'ffhq%06d.png' % i), image_grid)


# In[25]:


file_list = glob.glob(os.path.join(save_images_dir, '*.png'))  # Get all the pngs in the current directory
#file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
import re
r = re.compile(r'(ffhq\d+\.png)')
file_list_sorted = sorted(file_list, key=lambda x:r.search(x).group(1))
print(file_list_sorted)
fps = 1
clips = [ImageClip(m).set_duration(1)
         for m in file_list_sorted]
concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("ffhq.mp4", fps=fps)

