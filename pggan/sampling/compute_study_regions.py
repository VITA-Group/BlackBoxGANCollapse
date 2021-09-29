import pickle
import itertools
import os
import math
from operator import add
import numpy as np
import argparse
from utils import compute_embds_matrix, plot_distribution
import random

def count_neighbors_set(A_lst, epsilon, M, N):
    neighbors_count_lstoflst = []
    neighbors_count_dict = {}
    for i in range(N):
        print('i={}'.format(i))
        Ai = A_lst[i]
        Bi = np.transpose(Ai)
        AiBi = np.matmul(Ai, Bi)
        np.fill_diagonal(AiBi, 1)
        AiBi = np.arccos(AiBi) / math.pi
        np.fill_diagonal(AiBi, np.inf)
        AiBi = AiBi - np.ones(AiBi.shape) * epsilon
        neighbors_count = list(np.sum(AiBi <= 0, axis=1))
        neighbors_count_lstoflst.append(neighbors_count)
        for j in range(i):
            print('j={}'.format(j))
            Aj = A_lst[j]
            AjBi = np.matmul(Aj, Bi)
            np.fill_diagonal(AjBi, 1)
            AjBi = np.arccos(AjBi) / math.pi
            np.fill_diagonal(AjBi, np.inf)
            AjBi = AjBi - np.ones(AjBi.shape) * epsilon
            neighbors_count = list(np.sum(AjBi <= 0, axis=1))
            neighbors_count_lstoflst[j] = list(map(add, neighbors_count_lstoflst[j], neighbors_count))
            AiBj = np.transpose(AjBi)
            neighbors_count = list(np.sum(AiBj <= 0, axis=1))
            neighbors_count_lstoflst[i] = list(map(add, neighbors_count_lstoflst[i], neighbors_count))
        neighbors_count_dict[(i+1)*M] = np.asarray(list(itertools.chain(*neighbors_count_lstoflst)))
    return neighbors_count_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute neighbors with specified threshold')
    parser.add_argument('--saved_sampling_path', required=True, help='The path for reading embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The size of each tiled embedding matrix', type=int)
    parser.add_argument('--S', default=10, help='The size of S', type=int)
    parser.add_argument('--T', default=1000, help='The size of T', type=int)
    parser.add_argument('--K', default=100, help='The number of anchor points in Rref', type=int)
    parser.add_argument('--P', default=1, help='The number of paralleled job in obtained the samples', type=int)
    parser.add_argument('--theta', default=0.25, help='The threshold value of distance when counting neighbors', type=float)
    args, other_args = parser.parse_known_args()

    M = args.M
    S = args.S
    T = args.T
    K = args.K
    P = args.P

    D = args.M * args.T // args.P  # The number of samples collected by each paralleled job
    path = args.saved_sampling_path
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, S)
    if not os.path.exists(os.path.join(path, 'neighbors')):
        os.makedirs(os.path.join(path, 'neighbors'))

    theta = args.theta

    neighbors_count_dict = count_neighbors_set(A_lst, theta, M, S)
    final_neighbors_count = neighbors_count_dict[int(M*S)]
    print(max(final_neighbors_count))

    Rref_indices = random.sample(range(M * S), K)
    pos = -100
    Robs_indices = np.argpartition(final_neighbors_count, pos)[pos:]

    Rref_anchors_dct, Robs_anchors_dct = {}, {}

    for ind in Robs_indices:
        pkl_dir = os.path.join(os.path.join(path, 'embds'), '{}_{}'.format((ind // D)*D, (ind // D + 1)*D),
                               '{}_{}.pkl'.format((ind // M)*M, (ind // M + 1)*M))
        with open(pkl_dir, 'rb') as handle:
            pkl = pickle.load(handle)
        vec = pkl[ind % M]
        Robs_anchors_dct[ind] = vec / np.linalg.norm(vec)

    with open(os.path.join(path, 'neighbors', 'Robs_anchors_dct_{}.pkl'.format(theta)), 'wb') as handle:
        pickle.dump(Robs_anchors_dct, handle)

    i = 1
    for ind in Rref_indices:
        pkl_dir = os.path.join(os.path.join(path, 'embds'), '{}_{}'.format((ind // D) * D, (ind // D + 1) * D),
                               '{}_{}.pkl'.format((ind // M) * M, (ind // M + 1) * M))
        with open(pkl_dir, 'rb') as handle:
            pkl = pickle.load(handle)
        vec = pkl[ind % M]
        Rref_anchors_dct[ind] = vec / np.linalg.norm(vec)

        if i % 10 == 0:
            with open(os.path.join(path, 'neighbors', 'Rref_anchors_dct_{}_{}.pkl'.format(theta, i//10)), 'wb') as handle:
                pickle.dump(Rref_anchors_dct, handle)
                Rref_anchors_dct = {}
        i += 1