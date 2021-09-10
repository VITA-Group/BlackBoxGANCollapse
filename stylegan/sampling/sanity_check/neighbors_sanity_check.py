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
import math
from monte_carlo_sampling import monte_carlo

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

def find_distance_anchor(A_lst, I0, N):
    dist_lst = []
    for i in range(N):
        #print('i={}'.format(i))
        Ai = A_lst[i]
        #print(I0)
        AiT = np.transpose(Ai)
        #print(np.matmul(I0, AiT))
        theta_mat = np.arccos(np.matmul(I0, AiT)) / math.pi
        dist_lst.extend(theta_mat.flatten().tolist())
    return dist_lst

def gan_inference(Gs, latent, image_save_path, index):
    image = Gs.run(np.expand_dims(latent, axis=0), None, **synthesis_kwargs)
    print(image.shape)
    image = np.squeeze(image)
    image = PIL.Image.fromarray(image, 'RGB')
    dst = os.path.join(image_save_path, '{}.png'.format(index))
    image.save(dst, 'PNG')
    print(dst)

if __name__ == "__main__":

    tflib.init_tf()

    parser = argparse.ArgumentParser(description='Drawing images from latent codes')
    parser.add_argument('--saved_sampling_path', required=True, help='The path of the saved embeddings', type=str)
    parser.add_argument('--M', default=10000, help='The dimension of the tiled matrix', type=int)
    parser.add_argument('--N', default=1000, help='The number of tiled matrix', type=int)
    parser.add_argument('--S', default=10, help='The size of S', type=int)
    parser.add_argument('--T', default=1000, help='The size of T', type=int)
    parser.add_argument('--P', default=1, help='The number of paralleled job in obtained the samples', type=int)
    parser.add_argument('--theta', required=True, help='The threshold value of distance when counting neighbors', type=float)
    parser.add_argument('--region', required=True, help='The region for sanity check, either Robs or Rref', type=str)
    # parser.add_argument('--S_size', required=True, help='The size of S', type=str)

    args, other_args = parser.parse_known_args()

    path = args.saved_sampling_path
    M, N, S, T, P = args.M, args.N, args.S, args.T, args.P
    theta = args.theta

    D = M * T // P  # The number of samples collected by each paralleled job

    A_lst = compute_embds_matrix(os.path.join(path, 'embds'), M, N)

    region = args.region

    with open(os.path.join('monte_carlo_sampling_10m_celebahq', 'neighbors', '{}_anchors_embds_dct_{}.pkl'.format(region, theta)), 'rb') as handle:
        anchor_embds_dct = pickle.load(handle)

    anchor2neighbors_dct = {}
    for anchor_ind, embd in anchor_embds_dct.items():
        print(anchor_ind)
        embd = embd / np.linalg.norm(embd)
        embd = embd[np.newaxis, :]
        dist_lst = find_distance_anchor(A_lst, embd, N)
        pos = 100
        neighbors_inds = np.argpartition(dist_lst, pos)[:pos]
        anchor2neighbors_dct[anchor_ind] = []
        for ind in neighbors_inds:
            npy_dir = os.path.join(os.path.join(path, 'latents'), '{}_{}'.format((ind // D) * D, (ind // D + 1) * D),
                               '{}_{}.npy'.format((ind // M) * M, (ind // M + 1) * M))
            handle = np.load(npy_dir)
            anchor2neighbors_dct[anchor_ind].append(handle[ind % M])

    save_path = 'monte_carlo_sampling_10m_celebahq/neighbors/visuals/{}'.format(region)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    Gs = load_Gs('../pretrained_models/celebahq')


    with open(os.path.join('monte_carlo_sampling_10m_celebahq', 'neighbors', '{}_anchors_latents_dct_{}.pkl'.format(region, theta)), 'rb') as handle:
        anchor_latents_dct = pickle.load(handle)
    for anchor_ind, anchor_latent in anchor_latents_dct.items():
        print(anchor_ind)
        image_save_path = os.path.join(save_path, str(anchor_ind))
        if not os.path.exists(os.path.join(image_save_path, str(anchor_ind))):
            os.makedirs(os.path.join(image_save_path, str(anchor_ind)))
        gan_inference(Gs, anchor_latent, image_save_path, str(anchor_ind))
        index = 0
        for neighbor_latent in anchor2neighbors_dct[anchor_ind]:
            gan_inference(Gs, neighbor_latent, os.path.join(image_save_path, str(anchor_ind)), index)
            index += 1