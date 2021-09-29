import pickle
import os
import numpy as np
import argparse
import random
import math
from utils import compute_embds_matrix, str2bool
from math import log10
import re

def monte_carlo(A_lst, I0, N, d):
    exp_sim = 0
    for i in range(N):
        #print('i={}'.format(i))
        Ai = A_lst[i]
        #print(I0)
        AiT = np.transpose(Ai)
        #print(np.matmul(I0, AiT))
        dist_mat = np.arccos(np.clip(np.matmul(I0, AiT), -1.0, 1.0)) / math.pi
        sim_mat = (np.exp(np.maximum(0,np.ones(dist_mat.shape)*d - dist_mat))-np.ones(dist_mat.shape)) / (pow(math.e,d)-1)

        exp_sim += np.sum(sim_mat)
        #Pr += np.sum(np.exp(1-np.arccos(np.matmul(I0, AiT)) / math.pi))
    return 1 / -log10(exp_sim / (N*A_lst[0].shape[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the expected similarity by Monte Carlo Sampling')
    parser.add_argument('--saved_sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    parser.add_argument('--random_anchor', required=True, help='Whether we should get the anchor points by randomly sampling', type=str2bool)
    parser.add_argument('--theta', default=0.25, help='The threshold value of distance when counting neighbors')
    parser.add_argument('--job_id', default=None, help='The id of the submitted job', type=str)

    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    theta = args.theta
    path = args.saved_sampling_path

    if args.random_anchor:
        if args.job_id is None:
            anchor_pts_dct = {}
            regex = re.compile('Rref.*')
            for root, dirs, files in os.walk(os.path.join(path, 'neighbors')):
                for file in files:
                    if regex.match(file):
                        with open(os.path.join(path, 'neighbors', file), 'rb') as handle:
                            anchor_pts_dct.update(pickle.load(handle))
        else:
            with open(os.path.join(path, 'neighbors', 'Rref_anchors_dct_{}_{}.pkl'.format(theta, args.job_id)), 'rb') as handle:
                anchor_pts_dct = pickle.load(handle)
    else:
        with open(os.path.join(path, 'neighbors', 'Robs_anchors_dct_{}.pkl'.format(theta)), 'rb') as handle:
            anchor_pts_dct = pickle.load(handle)

    monte_carlo_dir = os.path.join(path, 'monte_carlo_sampling')
    if not os.path.exists(monte_carlo_dir):
        os.makedirs(monte_carlo_dir)
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)
    if args.random_anchor:
        fp = open(os.path.join(monte_carlo_dir, 'monte_carlo_sampling_ref.txt'), 'w')
    else:
        fp = open(os.path.join(monte_carlo_dir, 'monte_carlo_sampling_obs.txt'), 'w')
        #for N in [1, 10, 50, 100, 500, 1000]:
    for N in [1000]:
        for d in [0.3]:
            for k,v in anchor_pts_dct.items():
                print(d)
                v = v / np.linalg.norm(v)
                v = v[np.newaxis,:]
                exp_sim = monte_carlo(A_lst, v, N, d)
                result = '{}:{}:\t{}:{}'.format(N, k, d, exp_sim)
                print(result)
                fp.write(result+'\n')
    fp.close()