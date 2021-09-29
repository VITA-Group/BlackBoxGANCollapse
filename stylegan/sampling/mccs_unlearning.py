import pickle
import os
import numpy as np
import argparse
import random
import math
from utils import compute_embds_matrix, str2bool
from math import log10
import re
from sklearn.preprocessing import normalize

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
    parser.add_argument('--read_anchors_path', required=True, help='The path of the embedding of the anchor images for reading', type=str)
    args, other_args = parser.parse_known_args()

    M = args.M
    N = args.N
    path = args.saved_sampling_path

    with open(args.read_anchors_path, 'rb') as handle:
        samples = pickle.load(handle)
        # keys = list(samples.keys())
        # keys.sort(key=lambda txt: grep(r"(\d+)\.png", txt, 1))
        # samples = [samples[key] for key in keys]
        anchor_pts = normalize(samples, axis=1, norm='l2')
        print(anchor_pts.shape)

    monte_carlo_dir = os.path.join(path, 'unlearning')
    if not os.path.exists(monte_carlo_dir):
        os.makedirs(monte_carlo_dir)
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)
    fp = open(os.path.join(monte_carlo_dir, '{}.txt'.format(args.read_anchors_path.split('/')[-1])), 'w')
    d = 0.4
    for i in range(anchor_pts.shape[0]):
        v = anchor_pts[i]
        v = v / np.linalg.norm(v)
        v = v[np.newaxis,:]
        exp_sim = monte_carlo(A_lst, v, 10, d)
        result = '{}:\t{}:{}'.format(10, d, exp_sim)
        print(result)
        fp.write(result+'\n')
        if i==10000:
            break
    fp.close()