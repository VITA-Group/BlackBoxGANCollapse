#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pickle
import os
from shutil import copyfile

races_dict = pickle.load(open('fname_race_dict.pkl', 'rb'))
#genders_dict = pickle.load(open('fname_gender_dict.pkl', 'rb'))
#ages_dict = pickle.load(open('fname_age_dict.pkl', 'rb'))


# In[ ]:


white_count, black_count, asian_count = 0, 0, 0
white_lst, black_lst, asian_lst = [], [], []
for fname, race in races_dict.items():
    if race == 'Black':
        black_count += 1
        black_lst.append(fname.split('/')[-1])
    elif race == 'White':
        white_count += 1
        white_lst.append(fname.split('/')[-1])
    else:
        asian_count += 1
        asian_lst.append(fname.split('/')[-1])

print(black_count, len(black_lst))
print(white_count, len(white_lst))
print(asian_count, len(asian_lst))
for fname in black_lst:
    print(fname.split('/')[-1])


# In[23]:


if not os.path.exists('black'):
    os.makedirs('black')
if not os.path.exists('white'):
    os.makedirs('white')
if not os.path.exists('asian'):
    os.makedirs('asian')

for black in black_lst:
    copyfile(os.path.join('/mnt/ilcompf5d1/user/zwu/stylegan/datasets/ffhq-dataset/thumbnails128x128', black), os.path.join('black', black))
for white in white_lst:
    copyfile(os.path.join('/mnt/ilcompf5d1/user/zwu/stylegan/datasets/ffhq-dataset/thumbnails128x128', white), os.path.join('white', white))
for asian in asian_lst:
    copyfile(os.path.join('/mnt/ilcompf5d1/user/zwu/stylegan/datasets/ffhq-dataset/thumbnails128x128', asian), os.path.join('black', asian))    

