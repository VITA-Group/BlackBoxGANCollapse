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
    # To avoid math domain error
    exp_sim = 0
    if N < 1:
        pos = int(N * 10000)
        N = 1
    else:
        pos = 10000
    for i in range(N):
        # print('i={}'.format(i))
        # Ai = A_lst[i][:6000,:]
        Ai = A_lst[i][:pos, :]
        AiT = np.transpose(Ai)
        # print(np.matmul(I0, AiT))
        dist_mat = np.arccos(np.clip(np.matmul(I0, AiT), -1.0, 1.0)) / math.pi
        sim_mat = (np.exp(np.maximum(0, np.ones(dist_mat.shape) * d - dist_mat)) - np.ones(dist_mat.shape)) / (
        pow(math.e, d) - 1)

        exp_sim += np.sum(sim_mat)
        # print(exp_sim)
        # Pr += np.sum(np.exp(1-np.arccos(np.matmul(I0, AiT)) / math.pi))
    try:
        MCCS = 1 / -log10(exp_sim / (N * A_lst[0].shape[0]))
    except ValueError:
        MCCS = 0
    return MCCS


def compute_MCCS(obs_anchor_pts_dct, ref_anchor_pts_dct, N_range, theta, A_lst):
    fp_ref = open(os.path.join(monte_carlo_dir, 'monte_carlo_sampling_{}_ref.txt'.format(theta)), 'w')
    fp_obs = open(os.path.join(monte_carlo_dir, 'monte_carlo_sampling_{}_obs.txt'.format(theta)), 'w')

    for k, v in obs_anchor_pts_dct.items():
        for N in N_range:
            v = v / np.linalg.norm(v)
            v = v[np.newaxis, :]
            exp_sim = monte_carlo(A_lst, v, N, theta)
            result = '{}:{}:\t{}:{}'.format(int(N * 10000), k, theta, exp_sim)
            print(result)
            fp_obs.write(result + '\n')
    fp_obs.close()

    for k, v in ref_anchor_pts_dct.items():
        for N in N_range:
            v = v / np.linalg.norm(v)
            v = v[np.newaxis, :]
            exp_sim = monte_carlo(A_lst, v, N, theta)
            result = '{}:{}:\t{}:{}'.format(int(N * 10000), k, theta, exp_sim)
            print(result)
            fp_ref.write(result + '\n')
    fp_ref.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the expected similarity by Monte Carlo Sampling')
    parser.add_argument('--saved_sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    # parser.add_argument('--random_anchor', required=True, help='Whether we should get the anchor points by randomly sampling', type=str2bool)
    # parser.add_argument('--theta', default=0.25, help='The threshold value of distance when counting neighbors')
    parser.add_argument('--job_id', default=None, help='The id of the submitted job', type=str)

    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    path = args.saved_sampling_path

    monte_carlo_dir = os.path.join(path, 'monte_carlo_sampling')
    if not os.path.exists(monte_carlo_dir):
        os.makedirs(monte_carlo_dir)
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)
    N_range = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
               1, 10, 20, 50, 100, 200, 500, 1000]
    theta_range = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for theta in theta_range:
        if args.job_id is None:
            ref_anchor_pts_dct = {}
            regex = re.compile('Rref.*')
            for root, dirs, files in os.walk(os.path.join(path, 'neighbors')):
                for file in files:
                    if regex.match(file):
                        with open(os.path.join(path, 'neighbors', file), 'rb') as handle:
                            print(os.path.join(path, 'neighbors', file))
                            ref_anchor_pts_dct.update(pickle.load(handle))
            sampled_keys = random.sample(list(ref_anchor_pts_dct.keys()), 10)
            ref_anchor_pts_dct = {key: ref_anchor_pts_dct[key] for key in sampled_keys}
        else:
            with open(os.path.join(path, 'neighbors', 'Rref_anchors_dct_{}_{}.pkl'.format(theta, args.job_id)),
                      'rb') as handle:
                print(os.path.join(path, 'neighbors', 'Rref_anchors_dct_{}_{}.pkl'.format(theta, args.job_id)))
                ref_anchor_pts_dct = pickle.load(handle)
        with open(os.path.join(path, 'neighbors', 'Robs_anchors_dct_{}.pkl'.format(theta)), 'rb') as handle:
            obs_anchor_pts_dct = pickle.load(handle)
        compute_MCCS(obs_anchor_pts_dct, ref_anchor_pts_dct, N_range, theta, A_lst)