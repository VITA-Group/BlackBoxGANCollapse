#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%

import os
import glob
import numpy as np
import operator

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

#%%


# In[26]:


read_dir = '/home/wuzhenyu_sjtu/Desktop/visuals/Robs/81308/81308'
files = os.listdir(read_dir)
print(files)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize


save_images_dir = '/home/wuzhenyu_sjtu/Desktop/visuals/Robs/81308'
if not os.path.exists(save_images_dir):
    os.makedirs(save_images_dir)
img_lst = []
j = 0
for i in range(12*12):
    if i%12>=2 and i%12<=9 and i//12>=2 and i//12<=9:
        img = np.zeros([128,128,3], dtype=np.float32)
    else:  
        img = mpimg.imread(os.path.join(read_dir, files[j]))
        img = resize(img, (128,128))
        j += 1
    img_lst.append(img)


# In[23]:


print(len(img_lst))


# In[ ]:


image_stack = np.stack(img_lst,axis=0)
print(image_stack.shape)
image_grid = create_image_grid(image_stack, grid_size=(12,12))
print(image_grid.shape)
image_grid[2*128 : 10*128, 2*128 : 10*128, :] = mpimg.imread('/home/wuzhenyu_sjtu/Desktop/visuals/Robs/81308/81308.png')
mpimg.imsave(os.path.join(save_images_dir, 'R_obs.png'), image_grid)


# In[31]:


import sys
from PIL import Image

images = [Image.open(x) for x in ['/home/wuzhenyu_sjtu/Desktop/visuals/Robs/81308/R_obs.png',
                                  '/home/wuzhenyu_sjtu/Desktop/visuals/Rref/80368/R_ref.png']]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)+64
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]+64

new_im.save('collapse_sanity.png')

