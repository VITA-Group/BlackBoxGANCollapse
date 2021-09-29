#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from natsort import natsorted
from moviepy.editor import *
import numpy as np
import operator

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


# In[6]:


CC_path_1 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_CELEBAHQ/1'
CC_path_01 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_CELEBAHQ/0.1'
CC_path_001 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_CELEBAHQ/0.01'
CC_path_0001 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_CELEBAHQ/0.001'
#CC_path_00001 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_CELEBAHQ/0.0001'

CF_path_1 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_FFHQ/1'
CF_path_01 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_FFHQ/0.1'
CF_path_001 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_FFHQ/0.01'
CF_path_0001 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_FFHQ/0.001'
#CF_path_00001 = 'opt_w_subspace/generated_images/CELEBAHQ_pretrained/invert_FFHQ/0.0001'


FC_path_1 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_CELEBAHQ/1'
FC_path_01 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_CELEBAHQ/0.1'
FC_path_001 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_CELEBAHQ/0.01'
FC_path_0001 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_CELEBAHQ/0.001'
#FC_path_00001 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_CELEBAHQ/0.0001'

FF_path_1 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_FFHQ/1'
FF_path_01 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_FFHQ/0.1'
FF_path_001 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_FFHQ/0.01'
FF_path_0001 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_FFHQ/0.001'
#FF_path_00001 = 'opt_w_subspace/generated_images/FFHQ_pretrained/invert_FFHQ/0.0001'

F_path = 'ffhq-selected-combined'
C_path = 'celebahq-selected-combined'
C_files_list = os.listdir(C_path)
F_files_list = os.listdir(F_path)


# In[10]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

save_images_dir = 'celebahq_grid_images'
if not os.path.exists(save_images_dir):
    os.makedirs(save_images_dir)
for i in range(len(F_files_list)):
    img_C = cropND(mpimg.imread(os.path.join(F_path, F_files_list[i])), (800, 800))
    img_CC_1 = cropND(mpimg.imread(os.path.join(CF_path_1, F_files_list[i])), (800, 800))
    img_CC_01 = cropND(mpimg.imread(os.path.join(CF_path_01, F_files_list[i])), (800, 800))
    img_CC_001 = cropND(mpimg.imread(os.path.join(CF_path_001, F_files_list[i])), (800, 800))
    img_CC_0001 = cropND(mpimg.imread(os.path.join(CF_path_0001, F_files_list[i])), (800, 800))
    #img_CF_00001 = cropND(mpimg.imread(os.path.join(CF_path_00001, F_files_list[i])), (800, 800))
    
    
    img_FC_1 = cropND(mpimg.imread(os.path.join(FF_path_1, F_files_list[i])), (800, 800))
    img_FC_01 = cropND(mpimg.imread(os.path.join(FF_path_01, F_files_list[i])), (800, 800))
    img_FC_001 = cropND(mpimg.imread(os.path.join(FF_path_001, F_files_list[i])), (800, 800))
    img_FC_0001 = cropND(mpimg.imread(os.path.join(FF_path_0001, F_files_list[i])), (800, 800))
    #img_FF_00001 = cropND(mpimg.imread(os.path.join(FF_path_00001, F_files_list[i])), (800, 800))
    
    image_stack = np.stack((img_F,img_CF_1,img_CF_01,img_CF_001,img_CF_0001,
                            img_F,img_FF_1,img_FF_01,img_FF_001,img_FF_0001),axis=0)
    print(image_stack.shape)
    image_grid = create_image_grid(image_stack, grid_size=(5,2))
    print(image_grid.shape)
    mpimg.imsave(os.path.join(save_images_dir, 'celebahq%06d.png' % i), image_grid)


# In[ ]:


#base_dir = os.path.realpath("./images")
#print(base_dir)

#gif_name = 'pic'

file_list = glob.glob(os.path.join(save_images_dir, '*.png'))  # Get all the pngs in the current directory
#file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
import re
r = re.compile(r'(ffhq\d+\.png)')
file_list_sorted = sorted(file_list, key=lambda x:r.search(x).group(1))
print(file_list_sorted)


# In[8]:


fps = 1
clips = [ImageClip(m).set_duration(1)
         for m in file_list_sorted]
concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("ffhq.mp4", fps=fps)

