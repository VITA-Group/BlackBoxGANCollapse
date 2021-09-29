import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
from operator import add
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import argparse
import pylab as pl

def normalize_by_row(arr):
    row_sums = np.sqrt((arr*arr).sum(axis=1))
    new_arr = arr / row_sums[:, np.newaxis]
    return new_arr

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

def plot_distribution(final_neighbors_count_lstoflst, epsilon):
    #%matplotlib inline
    #plt.style.use('seaborn-deep')
    bins = np.linspace(0, 100, 100)
    plt.hist([final_neighbors_count_lstoflst[0], final_neighbors_count_lstoflst[4], final_neighbors_count_lstoflst[9],
              final_neighbors_count_lstoflst[19], final_neighbors_count_lstoflst[59], final_neighbors_count_lstoflst[79],
              final_neighbors_count_lstoflst[99]], normed=True,
             histtype='step', cumulative=True, bins=bins, color=['y','c','r','m','g','b','k'], label=['10000','50000','100000','200000','600000','800000','1000000'])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count of nearest neighbors within {} distance'.format(epsilon))
    plt.savefig('Sampling nearest neighbors within {} distance.png'.format(epsilon), dpi=600)

def plot_distribution_comparison():
    final_neighbors_count_lst = []
    color_lst = ['y','c','r','m','g','b','k','purple','navy','orangered']
    epsilon_lst = [0.08,0.09,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.2]
    for epsilon in epsilon_lst:
        with open ('final_neighbors_count_lstoflst_{}.pkl'.format(epsilon), 'rb') as fp:
            final_neighbors_count_lstoflst = pickle.load(fp)
        final_neighbors_count_lst.append(final_neighbors_count_lstoflst[99])
    bins = np.linspace(0, 100, 100)
    plt.hist(final_neighbors_count_lst, normed=True,
         histtype='step', cumulative=True, bins=bins, color=color_lst, label=[str(epsilon) for epsilon in epsilon_lst])
    plt.legend(loc='upper right')
    plt.ylabel('Prob');
    plt.xlabel('Count of nearest neighbors among sampled data')
    plt.savefig('Sampling nearest neighbors.png', dpi=600)

#if __name__ == "__main__":
with open('final_neighbors_count_lstoflst_{}.pkl'.format(0.2), 'rb') as fp:
    final_neighbors_count_lstoflst = pickle.load(fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling nearest neighbors')
    parser.add_argument('--start', help='Start of the distance hard threshold', type=float)
    parser.add_argument('--end', help='End of the distance hard threshold', type=float)
    parser.add_argument('--step_size', help='Step size of the distance hard threshold', type=float)
    args, other_args = parser.parse_known_args()

    M = 10000
    N = 100
    path = 'save_embds_pkls/random_sampling'
    A_lst = compute_embds_matrix(path, M)
    for epsilon in list(pl.frange(args.start,args.end,args.step_size)):
        final_neighbors_count_lstoflst = compute_nearest_neighbors(A_lst, epsilon, N)
        with open('final_neighbors_count_lstoflst_{}.pkl'.format(epsilon), 'wb') as fp:
            pickle.dump(final_neighbors_count_lstoflst, fp)
        plot_distribution(final_neighbors_count_lstoflst, epsilon)
