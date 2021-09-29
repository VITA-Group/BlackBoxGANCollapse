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
    parser.add_argument('--job_id', default=None, help='The id of the submitted job', type=str)
    parser.add_argument('--saved_sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    parser.add_argument('--K', default=100, help='The number of anchor points', type=int)
    parser.add_argument('--random_anchor', required=True, help='Whether we should get the anchor points by randomly sampling', type=str2bool)
    parser.add_argument('--theta', default=0.25, help='The threshold value of distance when counting neighbors')

    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    theta = args.theta
    path = args.saved_sampling_path

    if args.random_anchor:
        with open(os.path.join(path, 'neighbors', 'Rref_anchors_dct_{}.pkl'.format(theta)), 'rb') as handle:
            anchor_pts_dct = pickle.load(handle)
    else:
        with open(os.path.join(path, 'neighbors', 'Robs_anchors_dct_{}.pkl'.format(theta)), 'rb') as handle:
            anchor_pts_dct = pickle.load(handle)

    ripley_dir = os.path.join(path, 'ripley')
    if not os.path.exists(ripley_dir):
        os.makedirs(ripley_dir)
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M)

    if args.random_anchor:
        file = open(os.path.join(ripley_dir, 'ripley_Rref_{}_{}.txt'.format(theta, args.job_id)), 'w')
    else:
        file = open(os.path.join(ripley_dir, 'ripley_Robs_{}.txt'.format(theta)), 'w')
    for d in list(pl.frange(args.start,args.end,args.step_size)):
        for k,v in anchor_pts_dct.items():
            print(d)
            v = v / np.linalg.norm(v)
            v = v[np.newaxis,:]
            count = count_neighbors_pt(A_lst, v, N, d)
            result = '{}:\t{}:{}'.format(k, d, count)
            print(result)
            file.write(result+'\n')
    file.close()
