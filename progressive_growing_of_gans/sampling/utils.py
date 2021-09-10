import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
from operator import add
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import argparse

# url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl

def normalize_by_row(arr):
    row_sums = np.sqrt((arr * arr).sum(axis=1))
    new_arr = arr / row_sums[:, np.newaxis]
    return new_arr

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(ind))

def compute_embds_matrix(path, M, N=None):
    print(path)
    pkls = []
    for root, dirs, files in os.walk(path):
        if len(files) != 0:
            pkls.extend([os.path.join(root, file) for file in files if file.endswith('.pkl')])
    #pkls = os.listdir(path)
    pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
    if N is not None:
        pkls = pkls[:N]
    print(pkls)
    A_lst = []
    for pkl in pkls:
        print(pkl)
        with open(pkl, 'rb') as handle:
            samples = pickle.load(handle)
            # keys = list(samples.keys())
            # keys.sort(key=lambda txt: grep(r"(\d+)\.png", txt, 1))
            # samples = [samples[key] for key in keys]
            chunks = [normalize(np.asarray(samples[i:i + M]), axis=1, norm='l2') for i in range(0, len(samples), M)]
            print(chunks[0].shape)
            print(len(chunks))
            A_lst.extend(chunks)
    return A_lst

def plot_distribution(save_path, neighbors_count_dict, epsilon, M, N):
    # %matplotlib inline
    # plt.style.use('seaborn-deep')
    bins = np.linspace(0, 100, 100)
    plt.hist([neighbors_count_dict[int(M*N)]], normed=True, histtype='step', cumulative=True, bins=bins, color=['r'], label=[str(int(M*N))])
    plt.legend(loc='upper right')
    plt.ylabel('Prob')
    plt.xlabel('Count of neighbors within {} distance'.format(epsilon))
    plt.savefig(os.path.join(save_path,'Sampling neighbors within {} distance.png'.format(epsilon)), dpi=600)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_Gs(model_path):
    pkl = os.listdir(model_path)[0]
    with open(os.path.join(model_path, pkl), 'rb') as file:
        print(file)
        G, D, Gs = pickle.load(file)
        return Gs

# def load_Gs_from_url(url):
#     _Gs_cache = dict()
#     if url not in _Gs_cache:
#         with dnnlib.util.open_url(url, cache_dir='../cache') as f:
#             _G, _D, Gs = pickle.load(f)
#         _Gs_cache[url] = Gs
#     return _Gs_cache[url]