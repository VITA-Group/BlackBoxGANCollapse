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
%matplotlib inline


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
    %matplotlib inline
    bins = np.linspace(0, 30, 30)
    plt.hist([count_duplicate_lst], normed=True,
             histtype='step', cumulative=False, bins=bins, color=['b'], label=['# repeated identities'])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count')

def merge_files_idx():
    import shutil
    src_dir = 'img_resized'
    for idx, ims in flipped.items():
        dst_dir = os.path.join(src_dir, str(idx))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for im in ims:
            print('{:6d}.png'.format(im))
            shutil.move(os.path.join(src_dir, '{:06d}.png'.format(im)), os.path.join(dst_dir, '{:06d}.png'.format(im)))

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
