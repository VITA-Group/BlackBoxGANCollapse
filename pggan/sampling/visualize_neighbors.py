import pickle
import itertools
import os
import math
from sklearn.preprocessing import normalize
import re
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pylab as pl


def grep(pat, txt, ind):
    r = re.search(pat, txt)
    return int(r.group(1))

def compute_embds_matrix(path, M, N):
    pkls = []
    for root, dirs, files in os.walk(path):
        if len(files) != 0:
            pkls.extend([os.path.join(root, file) for file in files if file.endswith('.pkl')])
    #pkls = os.listdir(path)
    pkls.sort(key=lambda txt: grep(r"(\d+)_(\d+)\.pkl", txt, 1))
    pkls = pkls[:N]
    print(pkls)
    A_lst = []
    for pkl in pkls:
        print(pkl)
        with open(pkl, 'rb') as handle:
            samples = pickle.load(handle)
            # keys = list(samples.keys())
            # keys.sort(key=lambda txt: grep(r"(\d+)\.png", txt, 1))
            # samples = [samples[key] for key in keys]
            chunks = [normalize(np.asarray(samples[i:i + M]), axis=1, norm='l2') for i in range(0, len(samples), M)]
            print(chunks[0].shape)
            print(len(chunks))
            A_lst.extend(chunks)
    return A_lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize nearest neighbors')
    parser.add_argument('--start', required=True, help='Start of the distance threshold for neighbors', type=float)
    parser.add_argument('--end', required=True, help='End of the distance threshold for neighbors', type=float)
    parser.add_argument('--step_size', required=True, help='Step size of the epsilon', type=float)
    #parser.add_argument('--resolution', help='resolution of the trained model', type=int)
    parser.add_argument('--path', required=True, help='The path for reading embeddings', type=str)

    args, other_args = parser.parse_known_args()

    M = 10000
    N = 10
    #path = os.path.join(args.path, str(args.resolution))
    path = args.path
    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)

    for epsilon in list(pl.frange(args.start, args.end, args.step_size)):
        with open(os.path.join(path, 'neighbors', 'final_neighbors_count_lstoflst_{}.pkl'.format(epsilon)), 'rb') as fp:
            final_neighbors_count_lstoflst = pickle.load(fp)

        # final_neighbors_count_lst = final_neighbors_count_lstoflst[0]
        final_neighbors_count_lst = final_neighbors_count_lstoflst[N-1]
        print(max(final_neighbors_count_lst))
        final_neighbors_count_lst = np.asarray(final_neighbors_count_lst)
        indices = np.argpartition(final_neighbors_count_lst, -1)[-1:]
        print(indices)
        indices = np.asarray(indices)
        with open(os.path.join(args.path, 'neighbors', 'clustered_indices.pkl'), 'wb') as handle:
            pickle.dump(list(indices),handle)
        print(final_neighbors_count_lst[indices])
        # A = np.concatenate(A_lst, axis=0)
        # AT = np.transpose(A)
        from shutil import copyfile

        k = 0
        embds_lst = []
        for ind in indices:
            # vec = A[ind]
            # dist_arr = np.matmul(vec, AT)
            # dist_arr = np.arccos(dist_arr) / math.pi
            # index_lst = list(np.nonzero(dist_arr <= epsilon)[0])
            # pair_lst = [(index, dist_arr[index]) for index in index_lst]
            # pair_lst = sorted(pair_lst, key=lambda x: x[1])
            # i = 0
            # for index, _ in pair_lst:
            #     latents_dir = os.path.join('latents', '{}_{}'.format((index // 10000) * 10000, (index // 10000 + 1) * 10000))
            #     src = os.path.join(path, latents_dir, '{}_{}.npy'.format((index // 10000) * 10000, (index // 10000 + 1) * 10000))
            #     dst = os.path.join(path, 'neighbors', str(epsilon), str(ind), 'collision', 'neighbors_latents')
            #     if not os.path.exists(dst):
            #         os.makedirs(dst)
            #     dst = os.path.join(dst, '{}_{}.npy'.format(index, i))
            #     latents = np.load(src)
            #     np.save(dst, latents[index-(index // 10000) * 10000])
            #     i += 1

            # cluster
            #latents_dir = os.path.join('latents', '{}_{}'.format((ind // 10000) * 10000, (ind // 100000 + 1) * 10000))
            latents_dir = os.path.join('latents', '1000000_2000000')
            src = os.path.join(path, latents_dir, '{}_{}.npy'.format((ind // 10000) * 10000 + 1000000, (ind // 10000 + 1) * 10000 + 1000000))
            dst = os.path.join(path, 'neighbors', str(epsilon), 'clustered_latents')
            if not os.path.exists(dst):
                os.makedirs(dst)
            dst = os.path.join(dst, '{}_{}.npy'.format(ind, k))
            latents = np.load(src)
            np.save(dst, latents[ind % 10000])

            # # cluster
            # embds_dir = os.path.join('embds', '0_1000000')
            # src = os.path.join(path, embds_dir, '{}_{}.pkl'.format((ind // 10000) * 10000, (ind // 10000 + 1) * 10000))
            # dst = os.path.join(path, 'neighbors', str(epsilon), 'clustered_embds')
            # if not os.path.exists(dst):
            #     os.makedirs(dst)
            # dst = os.path.join(dst, '{}_{}.pkl'.format(ind, k))
            # embd = pickle.load(open(src, 'rb'))
            # embds_lst.append(embd[ind % 10000])
            # pickle.dump(embd[ind % 10000], open(dst, 'wb'))

            # # cluster
            # images_dir = os.path.join('images', '{}_{}'.format((ind // 10000) * 10000, (ind // 10000 + 1) * 10000))
            # src = os.path.join(home_dir, images_dir, '{}.png'.format(ind))
            # dst = os.path.join(home_dir, 'neighbors', str(epsilon), 'clustered_images')
            # if not os.path.exists(dst):
            #     os.makedirs(dst)
            # dst = os.path.join(dst, '{}_{}.png'.format(ind, k))
            # copyfile(src, dst)
            k += 1
        pickle.dump(np.asarray(embds_lst), open(os.path.join(path, 'neighbors', str(epsilon), 'clustered_embds.pkl'), 'wb'))