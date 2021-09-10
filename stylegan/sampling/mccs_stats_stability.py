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
    if N < 1:
        size = int(N * A_lst[0].shape[0])
        Ai = A_lst[0][:size]
        # print(I0)
        AiT = np.transpose(Ai)
        # print(np.matmul(I0, AiT))
        dist_mat = np.arccos(np.clip(np.matmul(I0, AiT), -1.0, 1.0)) / math.pi
        sim_mat = (np.exp(np.maximum(0, np.ones(dist_mat.shape) * d - dist_mat)) - np.ones(dist_mat.shape)) / (
                    pow(math.e, d) - 1)

        exp_sim += np.sum(sim_mat)
    else:
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
    try:
        MCCS = 1 / -log10(exp_sim / (N*A_lst[0].shape[0]))
        return MCCS
    except ValueError:
        print(exp_sim)
        print(N*A_lst[0].shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the expected similarity by Monte Carlo Sampling')
    parser.add_argument('--saved_sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    parser.add_argument('--K', default=5000, help='The number of anchor points', type=int)
    # parser.add_argument('--theta', default=0.25, help='The threshold value of distance when counting neighbors')
    parser.add_argument('--P', default=1, help='The number of paralleled job in obtaining the samples', type=int)
    parser.add_argument('--job_id', default=None, help='The id of the submitted job', type=str)

    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    K = args.K
    # theta = args.theta
    D = args.M * args.N // args.P  # The number of samples collected by each paralleled job

    path = args.saved_sampling_path

    anchor_inds = random.sample(range(M * N), K)
    anchor_inds.sort()

    anchors_embds_dct, anchors_latents_dct = {}, {}

    count = 0
    opened = -1
    for ind in anchor_inds:
        if opened != (ind // M)*M:
            pkl_dir = os.path.join(os.path.join(path, 'embds'), '{}_{}'.format((ind // D)*D, (ind // D + 1)*D),
                               '{}_{}.pkl'.format((ind // M)*M, (ind // M + 1)*M))
            with open(pkl_dir, 'rb') as handle:
                pkl = pickle.load(handle)
                embds = pkl[ind % M]
                anchors_embds_dct[ind] = embds / np.linalg.norm(embds)
            opened = (ind // M)*M
        else:
            embds = pkl[ind % M]
            anchors_embds_dct[ind] = embds / np.linalg.norm(embds)

        npy_dir = os.path.join(os.path.join(path, 'latents'), '{}_{}'.format((ind // D) * D, (ind // D + 1) * D),
                               '{}_{}.npy'.format((ind // M) * M, (ind // M + 1) * M))
        handle = np.load(npy_dir)
        anchors_latents_dct[ind] = handle[ind % M]
        #if count % 10000 == 0:
        print(count)
        count += 1

    with open(os.path.join(path, 'anchors', 'anchors_embds_dct.pkl'), 'wb') as handle:
        pickle.dump(anchors_embds_dct, handle)
    with open(os.path.join(path, 'anchors', 'anchors_latents_dct.pkl'), 'wb') as handle:
        pickle.dump(anchors_latents_dct, handle)

    with open(os.path.join(path, 'anchors', 'anchors_embds_dct.pkl'), 'rb') as handle:
        anchor_pts_dct = pickle.load(handle)

    monte_carlo_dir = os.path.join(path, 'monte_carlo_sampling_4')
    if not os.path.exists(monte_carlo_dir):
        os.makedirs(monte_carlo_dir)
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)
    # for Nv in [1, 10, 100]:
    for Nv in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for theta in [0.35, 0.4]:
            fp = open(os.path.join(monte_carlo_dir, 'monte_carlo_sampling_anchors_{}_{}.txt'.format(str(Nv), str(theta))), 'w')
            for k, v in anchor_pts_dct.items():
                print(theta)
                v = v / np.linalg.norm(v)
                v = v[np.newaxis,:]
                exp_sim = monte_carlo(A_lst, v, Nv, theta)
                result = '{}:{}:\t{}:{}'.format(Nv, k, theta, exp_sim)
                print(result)
                fp.write(result+'\n')
            fp.close()