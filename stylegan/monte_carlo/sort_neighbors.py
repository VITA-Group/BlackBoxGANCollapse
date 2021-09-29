import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pylab as pl

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

def compute_embds_matrix(path, M):
    pkls = os.listdir(path)
    pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
    print(pkls)
    A_lst = [] 
    for pkl in pkls:
        print(pkl)
        with open(os.path.join(path, pkl), 'rb') as handle:
            samples = pickle.load(handle)
            keys = list(samples.keys())
            keys.sort(key=lambda txt: grep(r"(\d+)\.png", txt, 1))
            samples = [samples[key] for key in keys]
            chunks = [normalize(np.asarray(samples[i:i+M]), axis=1, norm='l2') for i in range(0, len(samples), M)]
            print(chunks[0].shape)
            print(len(chunks))
            A_lst.extend(chunks)
    return A_lst


M = 10000
N = 100
path = 'save_embds_pkls/random_sampling'
epsilon = 0.2
A_lst = compute_embds_matrix(path, M)

with open('final_neighbors_count_lstoflst_{}.pkl'.format(epsilon), 'rb') as fp:
    final_neighbors_count_lstoflst = pickle.load(fp)
for final_neighbors_count_lst in final_neighbors_count_lstoflst:
    print(len(final_neighbors_count_lst))
    print(min(final_neighbors_count_lst))
    print(max(final_neighbors_count_lst))
    print(np.argmax(final_neighbors_count_lst))

final_neighbors_count_lst = final_neighbors_count_lstoflst[99]
print(max(final_neighbors_count_lst))
final_neighbors_count_lst = np.asarray(final_neighbors_count_lst)
indices = np.argpartition(final_neighbors_count_lst, -100)[-100:]
print(indices)
indices = np.asarray(indices)
print(final_neighbors_count_lst[indices])
A = np.concatenate(A_lst, axis=0)
AT = np.transpose(A)


group = np.ones(indices.shape[0])*(-1)
for i in range(indices.shape[0]):
    veci = A[indices[i]]
    min_dist = 1
    for j in range(i):
        vecj = A[indices[j]]
        if np.dot(veci, vecj) < min_dist:
            min_dist = np.dot(veci, vecj)
            group[i] = j
    if min_dist > 0.2:
        group[i] = i

print(group)
from shutil import copyfile

group_lst = []
for i in range(group.shape[0]):
    group_id = group[i]
    if group_id not in group_lst:
        group_lst.append(group_id)
    for j in range(len(group_lst)):
        if group_id == group_lst[j]:
            group[i] = int(j)

for i in range(indices.shape[0]):
    ind = indices[i]
    src_dir = '{}_{}'.format((ind // 10000)*10000, (ind // 10000 + 1)*10000)
    src_dir = os.path.join('/mnt/ilcompf5d1/user/zwu/stylegan-encoder/random_sampling/png', src_dir, '{}.png'.format(ind))
    dst_dir = os.path.join('/mnt/ilcompf5d1/user/zwu/stylegan-encoder/nearest_neighbors', 'sorted_clustered_images')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_dir = os.path.join(dst_dir, '{}_{}.png'.format(group[i],ind))
    copyfile(src_dir, dst_dir)