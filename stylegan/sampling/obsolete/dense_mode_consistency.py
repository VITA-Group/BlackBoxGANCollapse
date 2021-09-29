import os,sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import operator
import argparse
from utils import load_Gs, compute_embds_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from monte_carlo_sampling import monte_carlo

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

if __name__ == "__main__":

    tflib.init_tf()

    parser = argparse.ArgumentParser(description='Drawing images from latent codes')
    parser.add_argument('--saved_sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    parser.add_argument('--theta', required=True, help='The threshold value of distance when counting neighbors', type=float)
    parser.add_argument('--S_size', required=True, help='The size of S', type=str)

    args, other_args = parser.parse_known_args()

    path = args.saved_sampling_path
    M, N = args.M, args.N
    theta = args.theta

    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)

    with open(os.path.join('monte_carlo_sampling_10m_celebahq', 'neighbors_{}'.format(args.S_size), 'Robs_anchors_dct_{}.pkl'.format(theta)), 'rb') as handle:
        anchor_pts_dct = pickle.load(handle)

    ind2mccs_dct = {}
    for k, v in anchor_pts_dct.items():
        print(k)
        v = v / np.linalg.norm(v)
        v = v[np.newaxis, :]
        mccs = monte_carlo(A_lst, v, N, theta)
        ind2mccs_dct[k] = mccs
    ind2mccs_tuples = list(ind2mccs_dct.items())
    print(sorted(ind2mccs_tuples, key=lambda pair:pair[1]))
    ind2rank_dct = {ind: i for i, (ind, _) in enumerate(sorted(ind2mccs_tuples, key=lambda pair:pair[1]))}
    print(ind2rank_dct)

    save_path = 'monte_carlo_sampling_10m_celebahq/neighbors_{}/R_obs'.format(args.S_size)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Gs = load_Gs('../pretrained_models/celebahq')

    for k, v in anchor_pts_dct.items():
        print(k)
        image = Gs.run(np.expand_dims(v, axis=0), None, **synthesis_kwargs)
        print(image.shape)
        image = np.squeeze(image)
        image = PIL.Image.fromarray(image, 'RGB')
        dst = os.path.join(save_path, '{}_{}_{}.png'.format(ind2rank_dct[k], k, ind2mccs_dct[k]))
        image.save(dst, 'PNG')
        print(dst)