import pickle
import numpy as np
import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
import pickle
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pylab as pl

def atoi(text):
    return int(text) if text.isdigit() else text

def normalize_by_row(arr):
    row_sums = np.sqrt((arr*arr).sum(axis=1))
    new_arr = arr / row_sums[:, np.newaxis]
    return new_arr

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

def compute_embds_matrix(path, M, transpose=False):
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
            if transpose:
                chunks = [np.transpose(normalize(np.asarray(samples[i:i+M]), axis=1, norm='l2'))
                        for i in range(0, len(samples), M)]
            else:
                chunks = [normalize(np.asarray(samples[i:i+M]), axis=1, norm='l2')
                        for i in range(0, len(samples), M)]
            print(chunks[0].shape)
            print(len(chunks))
            A_lst.extend(chunks)
    return A_lst

def compute_nearest_neighbors(A_lst, B_lst, epsilon, N):
    A = np.concatenate(A_lst, axis=0)
    neighbors_count_final = [0] * A.shape[0]
    for i in range(N):
        print('i={}'.format(i))
        Bi = B_lst[i]
        print(Bi.shape)
        ABi = np.matmul(A, Bi)
        ABi = np.arccos(ABi) / math.pi
        ABi = ABi - np.ones(ABi.shape)*epsilon
        neighbors_count = list(np.sum(ABi <= 0, axis=1))
        neighbors_count_final = list(map(add, neighbors_count_final, neighbors_count))
    return neighbors_count_final

def plot_distribution_comparison():
    final_neighbors_count_lst = []
    color_lst = ['y','c','r','m','g','b','k','purple','navy','orangered']
    epsilon_lst = [0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26]
    for epsilon in epsilon_lst:
        with open('training_sampling_neighbors_count_final_{}.pkl'.format(epsilon), 'rb') as fp:
            neighbors_count_final = pickle.load(fp)
        final_neighbors_count_lst.append(neighbors_count_final)

    bins = np.linspace(0, 100, 100)
    plt.hist(final_neighbors_count_lst, normed=True, histtype='step', cumulative=True, bins=bins,
             color=color_lst[:len(epsilon_lst)], label=[str(epsilon) for epsilon in epsilon_lst])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count of nearest neighbors between training and sampled data')
    plt.savefig('Training sampling nearest neighbors between training and sampled data.png', dpi=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling nearest neighbors')
    parser.add_argument('--start', help='Start of the distance hard threshold', type=float)
    parser.add_argument('--end', help='End of the distance hard threshold', type=float)
    parser.add_argument('--step_size', help='Step size of the distance hard threshold', type=float)
    args, other_args = parser.parse_known_args()

    path_A = 'save_embds_pkls/training'
    path_B = 'save_embds_pkls/random_sampling'
    M = 5000
    N = 200
    A_lst = compute_embds_matrix(path_A, M)
    B_lst = compute_embds_matrix(path_B, M, transpose=True)
    final_neighbors_count_lstoflst = []
    epsilon_lst = []
    for epsilon in list(pl.frange(args.start,args.end,args.step_size)):
        neighbors_count_final = compute_nearest_neighbors(A_lst, B_lst, epsilon, N)
        with open('training_sampling_neighbors_count_final_{}.pkl'.format(epsilon), 'wb') as fp:
            pickle.dump(neighbors_count_final, fp)
        final_neighbors_count_lstoflst.append(neighbors_count_final)
        epsilon_lst.append(epsilon)
