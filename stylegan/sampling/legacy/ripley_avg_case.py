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
import random
from utils import compute_embds_matrix, str2bool

def monte_carlo(A_lst, I0, N, d):
    Count = 0
    for i in range(N):
        #print('i={}'.format(i))
        Ai = A_lst[i]
        #print(I0)
        AiT = np.transpose(Ai)
        #print(np.matmul(I0, AiT))
        theta_mat = np.arccos(np.matmul(I0, AiT)) / math.pi
        theta_mat = theta_mat - np.ones(theta_mat.shape)*d
        Count += np.sum(theta_mat <= 0)
        #Pr += np.sum(np.exp(1-np.arccos(np.matmul(I0, AiT)) / math.pi))
    return Count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling nearest neighbors')
    parser.add_argument('--start', required=True, help='Start of the distance hard threshold', type=float)
    parser.add_argument('--end', required=True, help='End of the distance hard threshold', type=float)
    parser.add_argument('--step_size', required=True, help='Step size of the distance hard threshold', type=float)
    parser.add_argument('--job_id', required=True, help='The id of the submitted job', type=str)
    parser.add_argument('--T_path', required=True, help='The path of T', type=str)
    parser.add_argument('--S_path', required=True, help='The path of S', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=5, help='The number of tiled matrix', type=int)
    parser.add_argument('--K', default=10, help='The number of anchor points', type=int)
    parser.add_argument('--random_anchor', required=True, help='Whether we should get the anchor points by randomly sampling', type=str2bool)
    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    path = os.path.join(args.S_path, 'embds')
    anchor_pt_dct = {}
    if args.random_anchor:
        indices = random.sample(range(M * 2), args.K)
    else:
        percentile_lst = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #percentile_lst = [0]
        indices = []
        for percentile in percentile_lst:
            with open(os.path.join(args.S_path, 'neighbors', 'clustered_indices_{}.pkl'.format(percentile)), 'rb') as handle:
                indices.append(pickle.load(handle))
                print('Loading indices from saved pickle file')
                print(indices)

    for i in indices:
        pkl_dir = os.path.join(path, '{}_{}'.format((i // 50000)*50000, (i // 50000 + 1)*50000),
                               '{}_{}.pkl'.format((i // 10000)*10000, (i // 10000 + 1)*10000))
        with open(pkl_dir, 'rb') as handle:
            pkl = pickle.load(handle)
        vec = pkl[i % 10000]
        anchor_pt_dct[i] = vec / np.linalg.norm(vec)

    ripley_dir = os.path.join(args.S_path, 'ripley')
    if not os.path.exists(ripley_dir):
        os.makedirs(ripley_dir)
    A_lst = compute_embds_matrix(os.path.join(args.T_path, 'embds'), M)
    file = open(os.path.join(ripley_dir, 'ripley_{}.txt'.format(args.job_id)), 'w')
    for d in list(np.arange(args.start, args.end, args.step_size)):
        for k,v in anchor_pt_dct.items():
            print(d)
            v = v / np.linalg.norm(v)
            v = v[np.newaxis,:]
            count = monte_carlo(A_lst, v, N, d)
            #Pr = (monte_carlo(A_lst, v, N)-10000000)/((np.e-1)*10000000)
            result = '{}:\t{}:{}'.format(k, d, count)
            print(result)
            file.write(result+'\n')
    file.close()
