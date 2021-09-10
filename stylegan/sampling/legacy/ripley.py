import pickle
import os
import math
import numpy as np
import argparse
import pylab as pl
import random
from utils import compute_embds_matrix, str2bool


def count_neighbors_pt(A_lst, I0, N, d):
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
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    parser.add_argument('--K', default=100, help='The number of anchor points', type=int)
    parser.add_argument('--random_anchor', required=True, help='Whether we should get the anchor points by randomly sampling', type=str2bool)
    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    path = os.path.join(args.sampling_path, 'embds')
    anchor_pt_dct = {}
    if args.random_anchor:
        indices = random.sample(range(M * N), args.K)
    else:
        with open(os.path.join(args.sampling_path, 'neighbors', 'clustered_indices.pkl'), 'rb') as handle:
            indices = pickle.load(handle)
            print('Loading indices from saved pickle file')
            print(indices)

    for i in indices:
        pkl_dir = os.path.join(path, '{}_{}'.format((i // 1000000)*1000000, (i // 1000000 + 1)*1000000),
                               '{}_{}.pkl'.format((i // 10000)*10000, (i // 10000 + 1)*10000))
        with open(pkl_dir, 'rb') as handle:
            pkl = pickle.load(handle)
        vec = pkl[i % 10000]
        anchor_pt_dct[i] = vec / np.linalg.norm(vec)

    ripley_dir = os.path.join(args.sampling_path, 'ripley')
    if not os.path.exists(ripley_dir):
        os.makedirs(ripley_dir)
    A_lst = compute_embds_matrix(path, M)
    file = open(os.path.join(ripley_dir, 'ripley_{}.txt'.format(args.job_id)), 'w')
    for d in list(pl.frange(args.start,args.end,args.step_size)):
        for k,v in anchor_pt_dct.items():
            print(d)
            v = v / np.linalg.norm(v)
            v = v[np.newaxis,:]
            count = count_neighbors_pt(A_lst, v, N, d)
            result = '{}:\t{}:{}'.format(k, d, count)
            print(result)
            file.write(result+'\n')
    file.close()
