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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def normalize_by_row(arr):
    row_sums = np.sqrt((arr*arr).sum(axis=1))
    new_arr = arr / row_sums[:, np.newaxis]
    return new_arr

def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

def compute_embds_matrix(path, M):
    pkls = [file for file in os.listdir(path) if file.endswith('pkl')]
    pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
    #print(pkls)
    A_lst = []
    for pkl in pkls:
        #print(pkl)
        with open(os.path.join(path, pkl), 'rb') as handle:
            samples = pickle.load(handle)
            #keys = list(samples.keys())
            #keys.sort(key=lambda txt: grep(r"(\d+)\.png", txt, 1))
            #samples = [samples[key] for key in keys]
            chunks = [normalize(np.asarray(samples[i:i+M]), axis=1, norm='l2') for i in range(0, len(samples), M)]
            #print(chunks[0].shape)
            #print(len(chunks))
            A_lst.extend(chunks)
    return A_lst

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
    parser.add_argument('--sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--indices_path', required=True, help='The path of the indices', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=100, help='The number of tiled matrix', type=int)
    parser.add_argument('--K', default=100, help='The number of anchor points', type=int)
    parser.add_argument('--random_anchor', required=True, help='Whether we should get the anchor points by randomly sampling', type=str2bool)
    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    anchor_pt_dct = {}
    if args.random_anchor:
        indices = random.sample(range(M * N // 10), args.K)
    else:
        with open(os.path.join(args.indices_path, 'neighbors', 'clustered_indices.pkl'), 'rb') as handle:
            indices = pickle.load(handle)
            print('Loading indices from saved pickle file')
            print(indices)

    for i in indices:
        pkl_dir = os.path.join(os.path.join(args.indices_path, 'embds', '0_1000000'), '{}_{}.pkl'.format((i // 10000)*10000, (i // 10000 + 1)*10000))
        with open(pkl_dir, 'rb') as handle:
            pkl = pickle.load(handle)
        vec = pkl[i % 10000]
        anchor_pt_dct[i] = vec / np.linalg.norm(vec)

    path = os.path.join(args.sampling_path, 'embds', '0_1000000')
    ripley_dir = os.path.join(args.sampling_path, 'ripley_{}'.format(args.indices_path))
    if not os.path.exists(ripley_dir):
        os.makedirs(ripley_dir)
    A_lst = compute_embds_matrix(path, M)
    file = open(os.path.join(ripley_dir, 'ripley_{}.txt'.format(args.job_id)), 'w')
    for d in list(pl.frange(args.start,args.end,args.step_size)):
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
