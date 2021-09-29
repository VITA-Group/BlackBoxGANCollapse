#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pickle
import numpy as np
import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
import glob
from operator import add
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

def plot_idx_distribution():
    iml_idx_dict = {}
    with open('celeba/identity_CelebA.txt', 'r') as handle:
        for line in handle:
            iml, idx = line.strip().split()
            iml_idx_dict[int(iml[:-4])] = int(idx)

    imh_idx_dict = {}
    with open('celeba/image_list.txt', 'r') as handle:
        next(handle)
        for line in handle:
            imh, _, iml, _, _ = line.strip().split()
            imh_idx_dict[int(imh)] = iml_idx_dict[int(iml[:-4])]
    
    print(imh_idx_dict)

    flipped = {}
    for k,v in imh_idx_dict.items():
        if v not in flipped:
            flipped[v] = [k]
        else:
            flipped[v].append(k)
    
    count_duplicate_lst = []
    for v in flipped.values():
        count_duplicate_lst.append(len(v))
    count_duplicate_lst.sort()
    print(count_duplicate_lst)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    get_ipython().run_line_magic('matplotlib', 'inline')
    bins = np.linspace(0, 30, 30)
    plt.hist([count_duplicate_lst], normed=True, 
             histtype='step', cumulative=False, bins=bins, color=['b'], label=['# repeated identities'])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count')
    plt.savefig('distribution.png')  # should before plt.show method
    return flipped


def merge_files_idx(flipped):
    import shutil
    src_dir = 'img_resized'
    for idx, ims in flipped.items():
        dst_dir = os.path.join(src_dir, str(idx))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for im in ims:
            print('{:6d}.png'.format(im))
            shutil.move(os.path.join(src_dir, '{:06d}.png'.format(im)), os.path.join(dst_dir, '{:06d}.png'.format(im)))
            


# In[8]:


flipped = plot_idx_distribution()
merge_files_idx(flipped)


# In[2]:


def compute_embds_matrix(path, M):
    pkls = glob.glob(os.path.join(path, "*.pkl"))
    print(pkls)
    A_lst = [] 
    for pkl in pkls:
        print(pkl)
        with open(pkl, 'rb') as handle:
            samples = pickle.load(handle)
            keys = list(samples.keys())
            keys.sort(key=lambda txt: grep(r"(\d+)\.png", txt, 1))
            samples = [samples[key] for key in keys]
            chunks = [normalize(np.asarray(samples[i:i+M]), axis=1, norm='l2') for i in range(0, len(samples), M)]
            print(chunks[0].shape)
            A_lst.extend(chunks)
    return A_lst

def compute_nearest_neighbors(A_lst, epsilon, N):
    neighbors_count_lstoflst = []
    final_neighbors_count_lstoflst = []
    for i in range(N):
        print('i={}'.format(i))
        Ai = A_lst[i]
        Bi = np.transpose(Ai)
        AiBi = np.matmul(Ai, Bi)
        np.fill_diagonal(AiBi, 1)
        AiBi = np.arccos(AiBi) / math.pi
        np.fill_diagonal(AiBi, np.inf)
        AiBi = AiBi - np.ones(AiBi.shape)*epsilon
        neighbors_count = list(np.sum(AiBi <= 0, axis=1))
        neighbors_count_lstoflst.append(neighbors_count)
        for j in range(i):
            print('j={}'.format(j))
            Aj = A_lst[j]
            AjBi = np.matmul(Aj, Bi)
            np.fill_diagonal(AjBi, 1)
            AjBi = np.arccos(AjBi) / math.pi
            np.fill_diagonal(AjBi, np.inf)
            AjBi = AjBi - np.ones(AjBi.shape)*epsilon
            neighbors_count = list(np.sum(AjBi <= 0, axis=1))
            neighbors_count_lstoflst[j] = list(map(add, neighbors_count_lstoflst[j], neighbors_count))
            AiBj = np.transpose(AjBi)
            neighbors_count = list(np.sum(AiBj <= 0, axis=1))
            neighbors_count_lstoflst[i] = list(map(add, neighbors_count_lstoflst[i], neighbors_count))
        final_neighbors_count_lstoflst.append(list(itertools.chain(*neighbors_count_lstoflst)))
    return final_neighbors_count_lstoflst


# In[7]:


def plot_distribution_comparison():
    final_neighbors_count_lst = []
    color_lst = ['y','c','r','m','g','b','k','purple','navy','orangered']
    epsilon_lst = [0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.2,0.22,0.24]
    for epsilon in epsilon_lst:
        with open('training_final_neighbors_count_lstoflst_{}.pkl'.format(epsilon), 'rb') as fp:
            final_neighbors_count = pickle.load(fp)
        final_neighbors_count_lst.append(final_neighbors_count[2])
    bins = np.linspace(0, 100, 100)
    plt.hist(final_neighbors_count_lst, normed=True, 
             histtype='step', cumulative=True, bins=bins, color=color_lst[:len(epsilon_lst)], 
             label=[str(epsilon) for epsilon in epsilon_lst])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count of nearest neighbors among training data')
    plt.savefig('Training nearest neighbors.png', dpi=600)


# In[8]:


path = 'save_embds_pkls/training'
M = 10000
N = 3
A_lst = compute_embds_matrix(path, M)
epsilon_lst = [0.2]
for epsilon in epsilon_lst:
    final_neighbors_count_lstoflst = compute_nearest_neighbors(A_lst, epsilon, N)
    with open('training_final_neighbors_count_lstoflst_{}.pkl'.format(epsilon), 'wb') as fp:
        pickle.dump(final_neighbors_count_lstoflst, fp)

